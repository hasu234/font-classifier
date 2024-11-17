import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from preprocessing import correct_skew, select_text_area, binarize_image
import torchvision.transforms as transforms
from fontClassifier import FontDataset, ResNet

def get_args():
    parser = argparse.ArgumentParser(description="Font Classification Training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    return parser.parse_args()

args = get_args()

# preprocessing function
def preprocess_data(img):
    img = binarize_image(img)
    img, _ = correct_skew(img)
    img = select_text_area(img)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img

# Define transformations for data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

# Define dataset path
dataset_path = args.dataset_path

# Load the dataset without any transforms initially
original_dataset = FontDataset(root_dir=dataset_path, preprocess_func=preprocess_data)

# Split the dataset into train (80%), validation (10%), and test (10%) sets
train_size = int(0.8 * len(original_dataset))
val_size = int(0.1 * len(original_dataset))
test_size = len(original_dataset) - train_size - val_size

train_indices, val_indices, test_indices = random_split(range(len(original_dataset)), [train_size, val_size, test_size])

# Create subsets with different transforms
train_dataset = FontDataset(root_dir=dataset_path, preprocess_func=preprocess_data, transform=train_transforms)
val_dataset = FontDataset(root_dir=dataset_path, preprocess_func=preprocess_data, transform=test_transforms)
test_dataset = FontDataset(root_dir=dataset_path, preprocess_func=preprocess_data, transform=test_transforms)

# Subset datasets
train_dataset.data = [original_dataset.data[i] for i in train_indices.indices]
train_dataset.labels = [original_dataset.labels[i] for i in train_indices.indices]
val_dataset.data = [original_dataset.data[i] for i in val_indices.indices]
val_dataset.labels = [original_dataset.labels[i] for i in val_indices.indices]
test_dataset.data = [original_dataset.data[i] for i in test_indices.indices]
test_dataset.labels = [original_dataset.labels[i] for i in test_indices.indices]

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the training loop with validation and saving the best model
def train(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_accuracy = 0.0  # To track the best validation accuracy
    best_model_wts = None  # To store the best model weights

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            running_loss += loss.item()

        train_accuracy = 100 * correct_preds / total_preds
        avg_train_loss = running_loss / len(train_loader)

        # Validation step
        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.size(0)

        val_accuracy = 100 * correct_preds / total_preds

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_wts = model.state_dict()

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    
    # Evaluate on the test set
    model.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    test_accuracy = 100 * correct_preds / total_preds
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Save the best model to a file
    torch.save(best_model_wts, 'best_resnet_model.pth')
    print(f"Best model saved with validation accuracy: {best_val_accuracy:.2f}%")

# Instantiate the model
model = ResNet(num_classes=4)

# Train the model
train(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.001)

# run the training script
# python train.py --dataset_path path/to/dataset