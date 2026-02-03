import torch
import albumentations as A
import numpy as np
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import clip

import sys
sys.path.append("..")
from app.representation import InceptionResnetV1

'''
Trying to leverage CLIP
'''
#clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
#embed_model = clip_model.visual  # This gives you the image encoder only


'''
Set global training variables
'''
BATCH_SIZE = 1024
NUM_EPOCHS = 200
# Use CUDA if available, else fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN_MODEL_WEIGHTS = "../models_weights/representation/20180402-114759-facenet.pt"

# Set dataset paths
LFW_DIR = "../data/lfw-deepfunneled"
CASIA_DIR = "../data/CASIA-maxpy-clean"  
CELEBA_DIR = "../data/CelebA" 
CFP_DIR = "../data/CFP_dataset"

'''
Set up logging to monitor training
'''
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_dir=log_dir)

'''
Training data setup
'''
# Define data augmentation functions for train and test MODIFIED for CLIP
train_transform = A.Compose([
    #A.RandomResizedCrop(size=(160, 160), scale=(0.8, 1.0)),  
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=0.5),
    A.OpticalDistortion(distort_limit=0.05, p=0.3),
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.1),
    A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), p=0.3),
    #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ToTensorV2(),
])

test_transform = A.Compose([
    #A.Resize(160, 160),
    A.Resize(224, 224),
    #A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    A.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ToTensorV2(),
])

# Load datasets and apply transforms
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


# Train dataset(s) accounting for overlap in people between different datasets
# Load datasets
train_dataset = ConcatDataset([ContrastiveImageDataset(LFW_DIR, transform=train_transform),
                               ContrastiveImageDataset(CELEBA_DIR, transform=train_transform)
])

# load in a test dataset
test_dataset = AlbumentationsImageDataset(CFP_DIR, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


'''
Create metric learning objects
'''
# Specify reducer to caluclate loss across all pairs or triplets in a batch
reducer = reducers.AvgNonZeroReducer()

# Specify a distance metric, if desired (most loss and calculators have a default, so check this before setting)
distance = distances.CosineSimilarity()

# Specify Regularizers, if desired
regularizer = regularizers.RegularFaceRegularizer()

# Specify the loss function
loss_fn = losses.SupConLoss()

# Specify an accuracy calcualtor for evaluating performance on the test
accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision"), k=5)

'''
Define Training and Test Functions
'''
def train(model, train_loader, optimizer, loss_fn, epoch, miner=None):
    model.train()
    total_loss = 0
    for batch_idx, (views, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        # views is a tuple of two augmented images
        # Stack along a new dimension so that views shape becomes (batch_size, n_views, C, H, W)
        views = torch.stack(views, dim=1).to(DEVICE)
        labels = labels.to(DEVICE)
        
        batch_size, n_views, C, H, W = views.shape
        # Merge batch and view dimensions for forward pass
        images = views.view(batch_size * n_views, C, H, W)
        #embeddings = model(images)  # shape: (batch_size*n_views, feature_dim)
        #added this bc CLIP embeddings benefit from L2 normalization 
        # because contrastive losses (like SupConLoss) assume embeddings lie on a unit hypersphere 
        # — this helps with training stability and semantic alignment.
        embeddings = model(images)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        # Repeat labels for each view: (batch_size,) -> (batch_size*n_views,)
        labels = labels.repeat_interleave(n_views)
        
        # SupConLoss expects (batch_size, n_views, feature_dim)
        loss = loss_fn(embeddings, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)

    return total_loss / len(train_loader)


# convenient function from pytorch-metric-learning to get all embeddings
def get_all_embeddings(dataset, model):
    model.eval()
    with torch.no_grad():
        tester = testers.BaseTester()
        embeddings, labels = tester.get_all_embeddings(dataset, model)
    return embeddings, labels


def test(test_loader, model, accuracy_calculator, loss_fn, epoch, miner=None):
    test_dataset = test_loader.dataset  # Access the dataset used by the DataLoader
    test_embeddings, test_labels = get_all_embeddings(test_dataset, model)  # Pass the dataset, not the DataLoader
    if test_labels.dim() > 1:
        test_labels = test_labels.squeeze(1) # Ensure labels are the correct shape

    test_loss = loss_fn(test_embeddings, test_labels)

    # Compute accuracy
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, test_labels)
    precision_at_1 = accuracies.get("precision_at_1", 0)
    mean_avg_precision = accuracies.get("mean_average_precision", 0)

    print(f"Test set accuracy (Precision@1): {precision_at_1:.4f}")
    print(f"Test set accuracy (Mean Average Precision): {mean_avg_precision:.4f}")
    print(f"Test set loss: {test_loss:.4f}")

    # Log metrics to TensorBoard
    writer.add_scalar("Precision@1/test", precision_at_1, epoch)
    writer.add_scalar("Mean Average Precision/test", mean_avg_precision, epoch)
    writer.add_scalar("Supervised Contrastive Loss/test", test_loss, epoch)

    return precision_at_1


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


if __name__=="__main__":
    '''
    Training portion
    '''
    # Instantiate embedding model
    #embed_model = InceptionResnetV1(weights_filename=PRETRAIN_MODEL_WEIGHTS, device=DEVICE, training=True)
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    embed_model = clip_model.visual  # Use vision encoder only

    # Define optimizer, include loss function if using a supervised function
    optimizer = torch.optim.Adam(
        list(embed_model.parameters()), 
        lr=3e-4, weight_decay=1e-5
    )

    # Define learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-9)

    # Compute initial test accuracy before training
    print("Initial Test Accuracy (Pre-trained Model):")
    initial_precision = test(test_loader, embed_model, accuracy_calculator, loss_fn, 0)

    # Initialize best loss tracking
    best_precision = initial_precision
    best_model_state = None

    # Training loop with two-stage fine-tuning
    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch < 50:  # First stage: Freeze the embedding model, train only the final layer
            # Adjust layer names depending on your model’s architecture.
            freeze_backbone_except(embed_model, layer_names=["last_linear", "last_bn"])
        else:  # Second stage: Unfreeze everything
            set_requires_grad(embed_model, True)

        train_loss = train(embed_model, train_loader, optimizer, loss_fn, epoch)
        print(f"Epoch {epoch}, Average Training Loss: {train_loss:.4f}")

        current_precision = test(test_loader, embed_model, accuracy_calculator, loss_fn, epoch)
        if current_precision > best_precision:
            best_precision = current_precision
            best_model_state = embed_model.state_dict()
            print(f"Epoch {epoch}: New best precision@1: {best_precision:.4f} achieved.")
            output_filename = f"{datetime.now().strftime('%Y%m%d')}_contrastive_facenet.pt"
            torch.save(best_model_state, output_filename)


        scheduler.step()  # Step the learning rate scheduler
        print("--------------------------------------------------------")

    # Save the best model
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