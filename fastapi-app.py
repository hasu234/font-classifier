import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from preprocessing import binarize_image, correct_skew, select_text_area
from fontClassifier import ResNet

app = FastAPI()

# Class labels
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
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img

preprocess = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor()
])

# Load the model
model = ResNet(num_classes=4)
model = load_model(model, 'modelWeight/best_resnet_model.pth')

# Prediction function
def predict(image: Image.Image, model, class_names):
    image = preprocess_image(image)
    image = preprocess(image)  # Apply preprocessing transformations
    image = image.unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Make the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    predicted_label = class_names[predicted_class.item()]
    return predicted_label

# FastAPI endpoint for image prediction
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Ensure the file is an image
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are supported.")

    try:
        # Open the uploaded image
        image = Image.open(file.file)
        predicted_label = predict(image, model, class_names)
        return JSONResponse(content={"predicted_label": predicted_label})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)