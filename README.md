# Disaster Response Pipeline Project

### Table of Contents

1. [Project Motivation](#motivation)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Acknowledgements](#acknowledgements)


### Project Motivation: <a name="motivation"></a>
The goal of this project is to practise using ETL pipeline, supervised Machine Learning and Python-based web app to solve a real-world problem. In this project, we look at a dataset of pre-labeled natural-disaster-related tweets provided by Figure Eight and train a model to predict the situation described in a inputted message/tweet.

The trained model is saved in a pickle file and used in a Python flask web-app, where user can input message and get the predicted natural-disaster-related labels.

### Installation: <a name="installation"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Description: <a name="files"></a>

   - **process_data.py**: where ETL pipeline is implemented to fetch data from two .csv files and store the cleaned data in a SQLite databse
   
   - **train_classifier.py**: where Machine Learning pipeline is implemented to train a classifier model and store it in a pickle file for web-app use
   
   - **run.py**: where the backend of the Flask web-app is implemented
   
### Acknowledgements<a name="acknowledgements"></a>

Credit to Figure Eight for the data and Udacity for the template. For further information, please consult Udacity website. 
