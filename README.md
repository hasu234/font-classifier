# Table of Content
- [Table of Content](#table-of-content)
- [Dependencies](#dependencies)
- [Setting up  conda Environment](#setting-up--conda-environment)
- [Training on your data](#training-on-your-data)
- [Inference Pipeline](#inference-pipeline)
  - [Running on Huggingface Space](#running-on-huggingface-space)
  - [Running inference script](#running-inference-script)
  - [Running the FastAPI App](#running-the-fastapi-app)
  - [Running on Docker](#running-on-docker)
  - [Running from Docker Hub](#running-from-docker-hub)
- [Training Log](#training-log)

# Dependencies

* Python 3.9
* opencv-python
* Pillow
* scikit-learn
* tqdm
* torch
* torchvision
* wandb
* fastapi
* uvicorn

# Setting up  conda Environment

* Clone the repository by running 
```
git clone https://github.com/hasu234/font-classifier.git
```
* Change the current directory to font-classifier
```
cd font-classifier
```
* Create a conda environment 
```
conda create -n font python=3.9
```
* Activate the environment 
```
conda activate font
```
* Install the required library from ```requirements.txt``` by running 
```
pip install -r requirements.txt
```

# Training on your data
To train on your dataset, make sure you have a data folder having the same folder hierarchy like below
```
├── dataset
│   ├── class1
│   │   ├──image1.jpg
│   │   ├──image2.jpg
│   ├── class2
│   │   ├──image1.jpg
│   │   ├──image2.jpg
│   ├── class3
│   │   ├──image1.jpg
│   │   ├──image2.jpg
│   ├── class4
│   │   ├──image1.jpg
│   │   ├──image2.jpg
```
or make some changes on ```train.py``` according to your dataset directory.
* Make sure you are in the project directory and run the ```train.py``` script with the folder directory of your dataset
```
python train.py --dataset_path /path/to/dataset_directory
```
# Inference Pipeline
There are 5 different way to infer the classifier. When running ```python``` script from terminal, make sure to activate the conda environment we have created earlier. 

## Running on Huggingface Space
Go to [this](https://huggingface.co/spaces/hasu234/font-classifier) link and upload the font image. You will see the Font Name below the image. 

## Running inference script
From the project directory, run the ```inference.py``` script from the terminal/command prompt specifying the test image location. 
```
python inference.py --image_path path/to/image.jpg
```
The predicted font name will be displayed to the terminal.

## Running the FastAPI App
From the project directory, run the ```fastapi-app.py``` script from terminal. 
```
python fastapi-app.py
```
The project should be running on ```localhost:8000``` port. By entering through ```http://localhost:8000/docs``` you could notice a single POST request with /predict/ endpoint. You just have to click ```Try it out```, upload an image, and click the ```Execute``` button. It should return a reponse like this
```json
{
  "predicted_label": "BalsamiqSans-Italic"
}
```

## Running on Docker
From the project directory, run the following command to build and run the classifier. Make sure the docker daemon is running.
* Build the Docker image by running  (First time only)
```
docker build -t font-classifier .
```
* Run the Docker Container 
```
docker run -p 8000:8000 font-classifier
```
The FastAPI app should run on ```localhost:8000``` port. Follow the previous FastAPI instruction to predict the font label.

## Running from Docker Hub
Make sure the docker daemon is running. This step doesn't need the GitHub repository to be on local. 
* Pull the Docker Image grom Docker Hub
```
docker pull hasu23/font-classifier
```
* Run the Docker Container
```
docker run -p 8000:8000 font-classifier
```

The FastAPI app should run on ```localhost:8000``` port. Follow the previous FastAPI instruction to predict the font label.

# Training Log
The training log can be accessed from [here](https://wandb.ai/hasmot23-organization/font-classifier?nw=nwuserhasmot23)