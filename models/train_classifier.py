import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import nltk
import pickle

def load_data(database_filepath):
    """
    This function loads the dataframe from specified database
    Input arguments: database_filepath
    Returns: X, Y, category_names
    """    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('processed', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names= Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
    This function tokenizes the given text and returns cleaned and lemmatized tokens
    Input arguments: text
    Returns: clean_tokens
    """  
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    This function builds a machine learning pipeline
    Input arguments: none
    Returns: pipeline
    """  
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function test model with test data and prints out classification reports
    Input arguments: model, X_test, Y_test, category_names
    Returns: none
    """  
    
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns= category_names)

    for column in Y_pred_df:
        print("Column: " + column)
        print(classification_report(Y_test[column], Y_pred_df[column]))
        print("-"*70)

def save_model(model, model_filepath):
    """
    This function saves the ML model in a pickle file
    Input arguments: model, model_filepath
    Returns: none
    """  
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """
    The main function train and saves a ML classifier
    """  
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()