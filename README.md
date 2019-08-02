# Disaster Response Pipeline


## 1. Project Overview

In this project, I'll apply data engineering to analyze disaster data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> to build a model for an API that classifies disaster messages.


This project will includes three main steps:

- Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure 
- Machine Learning Pipeline to train a model able to classify text message in categories
- Web App to show model results in real time.




## 2. Running

### 2.1. Data Cleaning

**Go to the project directory** and the run the following command:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
this line will do ETL Pipeline to extract data from source, clean data and save them in a proper databse structure


### 2.2. Training Classifier

After the data cleaning process, run this command **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

this line will do Machine Learning Pipeline to train a model able to classify text message in categories



### 2.3. Starting the web app

**Go the app directory** and run the following command:


```bat
python run.py
```
Go to http://0.0.0.0:3001/
