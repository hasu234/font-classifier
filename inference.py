import torch
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
from preprocessing import binarize_image, correct_skew, select_text_area
from fontClassifier import ResNet

def get_args():
    parser = argparse.ArgumentParser(description="Font Classification Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    return parser.parse_args()

args = get_args()

# class labels
class_names = ['AbyssinicaSIL-Regular', 'AdventPro-Italic[wdth,wght]', 'AdventPro[wdth,wght]', 'BalsamiqSans-Italic']

# Load the saved model
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the image to match input size and transformations used during training
def preprocess_image(img):
    img = binarize_image(img)
    img, _ = correct_skew(img)
    img = select_text_area(img)
    # Ensure the output is a PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img

preprocess = transforms.Compose([
        transforms.Resize((64, 256)),  # Ensure image size matches input size (64x256)
        transforms.ToTensor()
    ])

# Inference function
def predict(image_path, model, class_names):
    img = Image.open(image_path)
    image = preprocess_image(img)
    image = preprocess(image)  # Apply preprocessing transformations
    image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Make the prediction
    with torch.no_grad():  # No need to track gradients during inference
        output = model(image)
    
    # Get the predicted class label (index)
    _, predicted_class = torch.max(output, 1)
    
    # Get the class name
    predicted_label = class_names[predicted_class.item()]
    
    print(f'Predicted label: {predicted_label}')
    return predicted_label

# Load the model (assuming the model architecture is defined)
model = ResNet(num_classes=4)
model = load_model(model, 'best_resnet_model.pth')

# Predict on a single image
image_path = args.image_path
predicted_label = predict(image_path, model, class_names)

# Output: Predicted label: AbyssinicaSIL-Regular

# run by python inference.py --image_path image.jpg