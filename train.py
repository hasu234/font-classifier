import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


def compute_metrics(y_true, y_pred, average='macro'):
    """
    Compute accuracy, precision, recall, and F1 score.
    """
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average=average) * 100
    recall = recall_score(y_true, y_pred, average=average) * 100
    f1 = f1_score(y_true, y_pred, average=average) * 100
    return accuracy, precision, recall, f1

def train(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.001, patience=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Add weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Reduce LR every 3 epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_f1 = 0.0  # To track the best validation F1 score
    best_model_wts = None  # To store the best model weights
    early_stop_counter = 0  # Early stopping counter

    for epoch in range(num_epochs):
        # Training step
        model.train()
        running_loss = 0.0
        y_true_train, y_pred_train = [], []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

            running_loss += loss.item()

        # Compute training metrics
        train_accuracy, train_precision, train_recall, train_f1 = compute_metrics(y_true_train, y_pred_train)

        # Validation step
        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        # Compute validation metrics
        val_accuracy, val_precision, val_recall, val_f1 = compute_metrics(y_true_val, y_pred_val)

        # Save the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = model.state_dict()
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Precision: {train_precision:.2f}%, Recall: {train_recall:.2f}%, F1: {train_f1:.2f}%")
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%, "
              f"Precision: {val_precision:.2f}%, Recall: {val_recall:.2f}%, F1: {val_f1:.2f}%")

        # Early stopping
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

        # Step the scheduler
        scheduler.step()

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    # Evaluate on the test set
    model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())

    test_accuracy, test_precision, test_recall, test_f1 = compute_metrics(y_true_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Test Precision: {test_precision:.2f}%, Recall: {test_recall:.2f}%, F1 Score: {test_f1:.2f}%")

    # Save the best model to a file
    torch.save(best_model_wts, 'best_resnet_model.pth')
    print(f"Best model saved with validation F1 Score: {best_val_f1:.2f}%")

# Instantiate the model
model = ResNet(num_classes=4)

# Train the model
train(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.001, patience=3)

# run the training script
# python train.py --dataset_path path/to/dataset