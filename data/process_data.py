import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    This functions loads the messages .csv file and categories .csv file and merge them into one dataframe
    Input Arguments: messages_filepath, categories_filepath
    Returns: merged dataframe df
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories)
    return df

def clean_data(df):
    
    """
    This functions takes the categories column and convert it to 36 seperate columns with binary values
    and drop duplicates in the end.
    Input Arguments: original dataframe df
    Returns: converted dataframe df
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # replace the original categories column with the new `categories` dataframe
    df = df.drop(columns = ['categories'])
    df = pd.concat([df, categories],axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    This functions takes the dataframe and save it to the pointed file path
    Input Arguments: df, database_filename
    Returns: none
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('processed', engine, index=False)  

    
def main():
    """
    The main function loads, cleans and saves the data to the database
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()