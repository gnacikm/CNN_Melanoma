# Short Intro
In this repo we investigate various deep Convolutional Neural Networks (CNNs) in order to classify the images of Melanoma/Non-melanoma Cancer.  We will build and tune a simple CNN and also investigate Deep CNNs by transfer learning. 

# Package Requirenments
To install all necessary packages to run this repo please run
pip install -r requirements.txt

# Structure
The structure of this repo is the following:

## Notebooks
All JupyterNotebooks can be found in notebooks folder. 
There is a certain order that we suggest to run them. 

## Models
All classes for CNN models are defined in models folder. 
All useful scripts are defined in utils folder. 

## Models structure and weights
All models (structure, weights) are stored in saved_models folder. We keep weights in h5 format, we also store ONNX model format.

## Data
In order to download the data that is used in the repo please follow the intructions from 1.GettingData.ipynb. The data will be stored in data folder. 

## Kaggle 
You will need a kaggle folder with kaggle json to access Kaggle Api. Kaggle json can be generated directly from your kaggle progile. 

## Logs
The folder logs contains the logs for the TensorBoard. See Notebook 1. 
