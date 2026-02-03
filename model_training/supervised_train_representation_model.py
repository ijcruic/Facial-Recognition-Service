import os
import sys
import yaml
import torch
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CyclicLR
from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import nni
from sklearn.cluster import AgglomerativeClustering

sys.path.append("..")
from app.representation import InceptionResnetV1
from data.albu_transforms import get_dynamic_transforms, get_standard_transforms


# ─────────────────────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────────────────────
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
def train(model, train_loader, optimizer, scheduler, loss_fn, epoch_idx, writer, device):
    total_loss = 0
    for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        
        embeddings = model(data)

        # For supervised loss, like ArcFace
        loss = loss_fn(embeddings, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # For cyclicLR, need to step here
        scheduler.step()
        
        total_loss += loss.item()
        if (batch_idx % 50 == 0) and (batch_idx >0):
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            writer.add_scalar("Loss/train", loss.item(), epoch_idx * len(train_loader) + batch_idx)

    return total_loss / len(train_loader)


def test(model, test_loader, accuracy_calculator, loss_fn, epoch_idx, writer, device):
    test_dataset = test_loader.dataset  # Access the dataset used by the DataLoader
    test_embeddings, test_labels = get_all_embeddings(test_dataset, model, device)  # Pass the dataset, not the DataLoader
    if test_labels.dim() > 1:
        test_labels = test_labels.squeeze(1) # Ensure labels are the correct shape

    # test_loss = loss_fn(test_embeddings, test_labels)

    # Compute accuracy
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels)
    precision_at_1 = accuracies.get("precision_at_1", 0)
    mean_avg_precision = accuracies.get("mean_average_precision", 0)
    ami = accuracies.get("AMI", 0)

    # Log metrics to TensorBoard
    writer.add_scalar("Precision@1/test", precision_at_1, epoch_idx)
    writer.add_scalar("Mean Average Precision/test", mean_avg_precision, epoch_idx)
    writer.add_scalar("AMI", ami, epoch_idx)
    #writer.add_scalar("ArFace Loss/test", test_loss, epoch_idx)

    return precision_at_1, mean_avg_precision, ami, #test_loss


# ─────────────────────────────────────────────────────────────────────
#  Dataset wrappers
# ─────────────────────────────────────────────────────────────────────
class AlbumentationsImageDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)  # Load image
        sample = np.array(sample)  # Convert PIL Image to NumPy array

        if self.transform:
            sample = self.transform(image=sample)["image"]  # Pass explicitly as `image`
        return sample, target


if __name__ == "__main__":
    # 1) load supervised config
    config_path = "supervised_config.yaml"
    config = load_config(config_path)

    # 2) Ask NNI for a set of hyperparameters. If NNI is not running, it returns {}
    try:
        nni_p = nni.get_next_parameter()
    except:
        nni_p = {}

    # 3)  Override defaults from config with any NNI‐provided valuess
    hp = {}
    for k, dv in config["hyperparameters"].items():
        hp[k] = nni_p.get(k, dv)

    # Explicitly set each hyperparameter from hp:
    batch_size        = int(hp["batch_size"])
    learning_rate     = float(hp["learning_rate"])
    num_epochs        = int(hp["num_epochs"])
    first_stage_epochs = int(hp["first_stage_epochs"])
    weight_decay      = float(hp["weight_decay"])
    lr_min            = float(config["scheduler"]["eta_min"])

    # 4) Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5) Set up TensorBoard
    log_dir = config["logging"]["log_dir_prefix"] + datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    # 6) Create metric‐learning objects
    reducer = reducers.MeanReducer()
    distance = distances.CosineSimilarity()
    regularizer = regularizers.RegularFaceRegularizer()
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision", "AMI"), kmeans_func=custom_clustering, k=5)
    miner = miners.BatchEasyHardMiner(distance=distance)

    # SubCenterArcFace requires num_classes, margin, scale, embedding_size
    # We will set margin/scale via config
    # determine num_classes by summing classes across train dirs
    train_dirs = config["datasets"]["train_dirs"]
    total_classes = 0
    for d in train_dirs:
        total_classes += len(os.listdir(d))
    loss_fn = losses.SubCenterArcFaceLoss(
        num_classes=total_classes,
        weight_regularizer=regularizer,
        reducer=reducer,
        margin=config["loss"]["margin"],
        scale=config["loss"]["scale"],
        embedding_size=config["model"]["embedding_size"]
    ).to(device)


    # 7) Build data transforms
    train_transform = A.Compose([get_dynamic_transforms(size=(160,160)),
                                 get_standard_transforms()])
    
    test_transform = get_standard_transforms()

    train_datasets: list[AlbumentationsImageDataset] = []
    for train_path in train_dirs:
        if not os.path.isdir(train_path):
            raise ValueError(f"Train directory does not exist: {train_path}")
        train_datasets.append(
            AlbumentationsImageDataset(train_path, transform=train_transform)
        )
    # Concatenate them into one big training set
    train_dataset = ConcatDataset(train_datasets)
       
    # 8b) Load exactly one test dataset from test_dir
    test_path = config["datasets"]["test_dir"]
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
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Found {torch.cuda.device_count()} GPUs. Using DataParallel.")
        embed_model = torch.nn.DataParallel(embed_model)

    embed_model = embed_model.to(device)

    # 11) Create optimizer & LR scheduler
    optimizer = torch.optim.Adam(
        list(embed_model.parameters()) + list(loss_fn.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = CyclicLR(optimizer, base_lr=lr_min, max_lr=learning_rate, step_size_up=len(train_loader)*5, mode="triangular2", cycle_momentum=False)

    # 12) Compute initial test accuracy before training
    initial_precision, initial_map, initial_ami = test(embed_model, test_loader, accuracy_calculator, loss_fn, 0, writer, device)
    print("Initial Test Accuracy (Pre-trained Model):")
    print(f"[Epoch 0] Test Precision@1: {initial_precision:.4f}, mAP: {initial_map:.4f}, AMI: {initial_ami:.4f}")

    # Initialize best loss tracking
    best_ami = initial_ami
    best_model_state = None

    # 13) Training loop with two-stage fine-tuning
    for epoch in range(1, num_epochs + 1):
        if epoch <= first_stage_epochs:  # First stage: Freeze the embedding model, train only loss function
            embed_model.eval()
            set_requires_grad(embed_model, False)
            set_requires_grad(loss_fn, True)
            print("Stage 1 trainable params:", count_params(filter(lambda p: p.requires_grad, embed_model.parameters())))
            print("Stage 1 trainable loss params:", count_params(filter(lambda p: p.requires_grad, loss_fn.parameters())))
        elif (epoch > first_stage_epochs) and (epoch <=first_stage_epochs+10):
            embed_model.train()
            freeze_backbone_except(embed_model, layer_names=['last_linear', 'last_bn'])
            set_requires_grad(loss_fn, True)
            print("Stage 2 trainable params:", count_params(filter(lambda p: p.requires_grad, embed_model.parameters())))
            print("Stage 2 trainable loss params:", count_params(filter(lambda p: p.requires_grad, loss_fn.parameters())))
        else:  # Second stage: Unfreeze everything
            embed_model.train()
            set_requires_grad(embed_model, True)
            set_requires_grad(loss_fn, True)
            print("Full Model Stage trainable params:", count_params(filter(lambda p: p.requires_grad, embed_model.parameters())))
            print("Full Model Stage loss params:", count_params(filter(lambda p: p.requires_grad, loss_fn.parameters())))

        train_loss = train(embed_model, train_loader, optimizer, scheduler, loss_fn, epoch, writer, device)
        print(f"Epoch {epoch}, Average Training Loss: {train_loss:.4f}")

        prec1, map_score, ami_score = test(embed_model, test_loader, accuracy_calculator, loss_fn, epoch, writer, device)
        print(f"[Epoch {epoch}] Test Precision@1: {prec1:.4f}, mAP: {map_score:.4f}, AMI: {ami_score:.4f}")
        nni.report_intermediate_result(ami_score.item() if isinstance(ami_score, torch.Tensor) else float(ami_score))

        if ami_score >best_ami:
            best_ami = ami_score
            best_model_state = embed_model.state_dict()
            print(f"Epoch {epoch}: New best AMI achieved at: {best_ami:.4f}")
            output_filename = f"{datetime.now().strftime('%Y%m%d')}_supervised_facenet.pt"
            torch.save(best_model_state, output_filename)

        #scheduler.step()  # Step the learning rate scheduler
        print("--------------------------------------------------------")

    # 14) After all epochs, report final best to NNI and save
    nni.report_final_result(best_ami.item() if isinstance(best_ami, torch.Tensor) else float(best_ami))
    if best_model_state is not None:
        output_filename = f"{datetime.now().strftime('%Y%m%d')}_supervised_facenet.pt"
        torch.save(best_model_state, output_filename)
        print(f"Best model saved as {output_filename}")
    else:
        print("No model improvement observed; saving the final model state.")
        output_filename = f"{datetime.now().strftime('%Y%m%d')}_supervised_facenet.pt"
        torch.save(embed_model.state_dict(), output_filename)
        print(f"Final model saved as {output_filename}")

    # Close TensorBoard writer
    writer.close()