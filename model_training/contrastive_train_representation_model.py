import os
import sys
import yaml
import torch
import albumentations as A
import numpy as np
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import nni
from sklearn.cluster import AgglomerativeClustering

sys.path.append("..")
from app.representation import InceptionResnetV1
from data.albu_transforms import get_dynamic_transforms, get_standard_transforms

'''
Define Utility functions
'''
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
    
def count_params(params):
    return sum(p.numel() for p in params)

def get_all_embeddings(dataset, model, device):
    # convenient function from pytorch-metric-learning to get all embeddings
    model.eval()
    with torch.no_grad():
        tester = testers.BaseTester()
        embeddings, labels = tester.get_all_embeddings(dataset, model)
    return embeddings.to(device), labels.to(device)

def custom_clustering(x, nmb_clusters):
    x_np = x.cpu().numpy()
    clusterer = AgglomerativeClustering(n_clusters=nmb_clusters)
    preds = clusterer.fit_predict(x_np)
    return torch.tensor(preds)


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def freeze_backbone_except(model, layer_names):
    """
    Freeze all parameters in model except those whose names contain any string in layer_names.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layer_names):
            param.requires_grad = True
        else:
            param.requires_grad = False

    
# ─────────────────────────────────────────────────────────────────────
#  Train / Test functions
# ─────────────────────────────────────────────────────────────────────
def train(model, train_loader, optimizer, loss_fn, epoch_idx, writer, device, miner=None):
    total_loss = 0
    for batch_idx, (views, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch_idx}")):
        # views is a tuple of two augmented images
        # Stack along a new dimension so that views shape becomes (batch_size, n_views, C, H, W)
        views = torch.stack(views, dim=1).to(device)
        labels = labels.to(device)
        
        batch_size, n_views, C, H, W = views.shape
        # Merge batch and view dimensions for forward pass
        images = views.view(batch_size * n_views, C, H, W)
        embeddings = model(images)  # shape: (batch_size*n_views, feature_dim)
        # Repeat labels for each view: (batch_size,) -> (batch_size*n_views,)
        labels_expanded = labels.repeat_interleave(n_views)
        
        # SupConLoss expects (batch_size, n_views, feature_dim)
        loss = loss_fn(embeddings, labels_expanded)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 20 == 0:
            #print(f"Epoch {epoch_idx}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            writer.add_scalar("Loss/train", loss.item(), epoch_idx * len(train_loader) + batch_idx)

    return total_loss / len(train_loader)


def test(model, test_loader, accuracy_calculator, loss_fn, epoch_idx, writer, device, miner=None):
    test_dataset = test_loader.dataset  # Access the dataset used by the DataLoader
    test_embeddings, test_labels = get_all_embeddings(test_dataset, model, device)  # Pass the dataset, not the DataLoader
    if test_labels.dim() > 1:
        test_labels = test_labels.squeeze(1) # Ensure labels are the correct shape

    #test_loss = loss_fn(test_embeddings, test_labels)

    # Compute accuracy
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels)
    precision_at_1 = accuracies.get("precision_at_1", 0)
    mean_avg_precision = accuracies.get("mean_average_precision", 0)
    ami = accuracies.get("AMI", 0)

    # Log metrics to TensorBoard
    writer.add_scalar("Precision@1/test", precision_at_1, epoch_idx)
    writer.add_scalar("Mean Average Precision/test", mean_avg_precision, epoch_idx)
    writer.add_scalar("AMI", ami, epoch_idx)
    #writer.add_scalar("Supervised Contrastive Loss/test", test_loss, epoch_idx)

    return precision_at_1, mean_avg_precision, ami, 0.0 #test_loss


# ─────────────────────────────────────────────────────────────────────
# Define dataset classes
# ─────────────────────────────────────────────────────────────────────
class AlbumentationsImageDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)  # Load image
        sample = np.array(sample)  # Convert PIL Image to NumPy array

        if self.transform:
            sample = self.transform(image=sample)["image"]  # Pass explicitly as `image`
        return sample, target
    
class ContrastiveImageDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(sample)
        if self.transform:
            # Create two independent augmentations for each image
            view1 = self.transform(image=sample)["image"]
            view2 = self.transform(image=sample)["image"]
            return (view1, view2), target
        return sample, target


if __name__=="__main__":
    '''
    Set hyperparameters
    '''
    # 1) Load YAML config (dataset paths + default hyperparameters)
    config_path = "contrastive_config.yaml"
    config = load_config(config_path)

    # 2) Ask NNI for a set of hyperparameters. If NNI is not running, it returns {}
    try:
        nni_params = nni.get_next_parameter()
    except:
        nni_params = {}

    # 3) Override defaults from config with any NNI‐provided values
    hp = {}
    default_hp = config["hyperparameters"]
    for key, default_val in default_hp.items():
        hp[key] = nni_params.get(key, default_val)

    # Explicitly set each hyperparameter from hp:
    batch_size        = int(hp["batch_size"])
    learning_rate     = float(hp["learning_rate"])
    num_epochs        = int(hp["num_epochs"])
    weight_decay      = float(hp["weight_decay"])
    temperature       = float(hp["temperature"])
    lr_min            = float(config["scheduler"]["eta_min"])
    # Handle first_stage_epochs specially
    first_stage_epochs = hp["first_stage_epochs"]
    if isinstance(first_stage_epochs, str) and first_stage_epochs == "num_epochs":
        first_stage_epochs = num_epochs
    else:
        first_stage_epochs = int(first_stage_epochs)

    # 4) Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5) Set up TensorBoard
    log_dir = config["logging"]["log_dir_prefix"] + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    # 6) Create metric‐learning objects
    reducer = reducers.AvgNonZeroReducer()
    distance = distances.CosineSimilarity()
    regularizer = regularizers.RegularFaceRegularizer()
    miner = miners.BatchEasyHardMiner(
        pos_strategy=hp["miner_pos_strategy"],
        neg_strategy=hp["miner_neg_strategy"],
        allowed_pos_range=(hp["miner_allowed_pos_low"], hp["miner_allowed_pos_high"]),
        allowed_neg_range=(hp["miner_allowed_neg_low"], hp["miner_allowed_neg_high"]),
        distance=distances.CosineSimilarity() if hp["miner_distance"] == "cosine" else distances.LpDistance(p=2)
    )
    base_loss = losses.NTXentLoss(temperature=temperature)
    loss_fn = losses.CrossBatchMemory(
        loss=base_loss,
        embedding_size=512,
        memory_size=hp.get("memory_size", 2048),
        miner=miner
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision", "AMI"), kmeans_func=custom_clustering, k=5)


    '''
    Set Datasets
    '''
    ds_cfg = config["datasets"]

    # 7) Build data transforms
    train_transform = A.Compose([get_dynamic_transforms(size=(160,160)),
                                 get_standard_transforms()])
    
    test_transform = get_standard_transforms()

    # 8a) Build all train datasets by iterating through the list of folders
    train_dirs = ds_cfg["train_dirs"]
    train_datasets: list[ContrastiveImageDataset] = []
    for train_path in train_dirs:
        if not os.path.isdir(train_path):
            raise ValueError(f"Train directory does not exist: {train_path}")
        train_datasets.append(
            ContrastiveImageDataset(train_path, transform=train_transform)
        )
    # Concatenate them into one big training set
    train_dataset = ConcatDataset(train_datasets)

    # 8b) Load exactly one test dataset from test_dir
    test_path = ds_cfg["test_dir"]
    if not os.path.isdir(test_path):
        raise ValueError(f"Test directory does not exist: {test_path}")
    test_dataset = AlbumentationsImageDataset(test_path, transform=test_transform)

    # 9) Build DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["dataloader"]["num_workers"])
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config["dataloader"]["num_workers"])

    '''
    Training portion
    '''
    # 10) Load pre-trained embedding model
    pretrained_weights = config["pretrained"]["weights_path"]
    embed_model = InceptionResnetV1(
        weights_filename=pretrained_weights,
        device=device,
        training=True
    )

    # If multiple GPUs available, wrap in DataParallel
    #if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #    print(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
    #    embed_model = torch.nn.DataParallel(embed_model)

    embed_model = embed_model.to(device)

    # 11) Create optimizer & LR scheduler
    optimizer = torch.optim.Adam(
        list(embed_model.parameters()) + list(loss_fn.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)

    # 12) Compute initial test accuracy before training
    initial_precision, initial_map, initial_ami, initial_loss = test(embed_model, test_loader, accuracy_calculator, loss_fn, 0, writer, device)
    nni.report_final_result(initial_precision.item() if isinstance(initial_precision, torch.Tensor) else float(initial_precision))
    print(f"[Epoch 0] Test Precision@1: {initial_precision:.4f}, mAP: {initial_map:.4f}, AMI: {initial_ami:.4f}, loss: {initial_loss:.4f}")

    # Initialize best prec@1 tracking
    best_precision = initial_precision
    best_model_state = None
    prec1s = []

    # 13) Training loop with two-stage fine-tuning
    for epoch in range(1, num_epochs + 1):
        if epoch <= first_stage_epochs:  # First stage: Freeze the embedding model, train only the final layer
            # Adjust layer names depending on your model’s architecture.
            freeze_backbone_except(embed_model, layer_names=["last_linear", "last_bn"])
            print("Stage 1 trainable params:", count_params(filter(lambda p: p.requires_grad, embed_model.parameters())))
            print("Stage 1 trainable loss params:", count_params(filter(lambda p: p.requires_grad, loss_fn.parameters())))
        else:  # Second stage: Unfreeze everything
            set_requires_grad(embed_model, True)
            print("Full Model Stage trainable params:", count_params(filter(lambda p: p.requires_grad, embed_model.parameters())))
            print("Full Model Stage loss params:", count_params(filter(lambda p: p.requires_grad, loss_fn.parameters())))

        train_loss = train(embed_model, train_loader, optimizer, loss_fn, epoch, writer, device)
        print(f"Epoch {epoch}, Average Training Loss: {train_loss:.4f}")

        prec1, map_score, ami_score, test_loss = test(embed_model, test_loader, accuracy_calculator, loss_fn, epoch, writer, device)
        print(f"[Epoch {epoch}] Test Precision@1: {prec1:.4f}, mAP: {map_score:.4f}, AMI: {ami_score:.4f}, Loss: {test_loss:.4f}")
        nni.report_intermediate_result(prec1.item() if isinstance(prec1, torch.Tensor) else float(prec1))
        prec1s.append(prec1.item() if isinstance(prec1, torch.Tensor) else float(prec1))

        if prec1 > best_precision:
            best_precision = prec1
            best_model_state = embed_model.state_dict()
            print(f"Epoch {epoch}: New best Precision@1 achieved at: {best_precision:.4f}")
            output_filename = f"{datetime.now().strftime('%Y%m%d')}_contrastive_facenet.pt"
            torch.save(best_model_state, output_filename)


        scheduler.step()  # Step the learning rate scheduler
        print("--------------------------------------------------------")

    # 14) After all epochs, report final best to NNI and save
    nni.report_final_result(np.max(prec1s))
    if best_model_state is not None:
        output_filename = f"{datetime.now().strftime('%Y%m%d')}_contrastive_facenet.pt"
        torch.save(best_model_state, output_filename)
        print(f"Best model saved as {output_filename}")
    else:
        print("No model improvement observed; saving the final model state.")
        output_filename = f"{datetime.now().strftime('%Y%m%d')}_contrastive_facenet.pt"
        torch.save(embed_model.state_dict(), output_filename)
        print(f"Final model saved as {output_filename}")

    # Close TensorBoard writer
    writer.close()