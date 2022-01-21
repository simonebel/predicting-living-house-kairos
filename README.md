# predicting-living-house-kairos
This repository is part of the process for Kairos Fire

<br>

# Approach 

The aims of this project is to predict the living locations of users. We have a dataset which contains measures of the location of the users in the time. 
* We first explore the data to get an intuition on how we should handle the data. 
* Then we use DBSCAN to predict the locations of the users. 
* Finally we take the denser cluster for each user and estimate the center of it to get the prediction  

<br>

# How to use this project ? 

* Clone this projet 

* Create the virtual environnement 

    `
    conda env create -f environment.yml
    `

* Activate the environnement 

    `
    conda activate kairos
    `

* Launch Jupyter lab 

    `
    jupyter lab
    `

<br>

# Folder 

| Folder | Content |
|:--------:|:---------:|
| [data](./data) | This folder contains dumps of dataframe used during this project |
| [location_prediction](./location_prediction) | This is our main module it contains two files : [data_visualisation](./location_prediction/data_visualisation.py) which contains the functions used for the exploration of the data and [model](./location_prediction/model.py) which contains the functions used to get the prediction |
| [static](./static) | This folder contains some images and the [report](./static/report).|
| [0.Exploratory_Data_Analysis.ipynb](./0.Exploratory_Data_Analysis.ipynb)  | The notebook containing the exploration of the data |
| [1.Model.ipynb](./1.Model.ipynb) | The notebook containing the approach to create the model and get the prediction  |
| README.md | This file |
| [env.yml](./env.yml)| A yaml containing the virtual environnment |