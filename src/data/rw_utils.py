from sqlalchemy import create_engine
import pandas as pd
from pandas import DataFrame
from os.path import dirname
import warnings
import pickle

warnings.filterwarnings(action='ignore')
DATA_PATH = dirname(dirname(dirname(__file__))) + "/data/"


def read_from_db(table_name: str = "longform", db_name: str = "news.db") -> DataFrame:
    """
    Read data from db and returns dataFrame

    :param table_name: table name, :type str
    :param db_name: db name, type: str
    :return: DataFrame
    """
    con = create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()
    df = pd.read_sql_table(f"{table_name}", con)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def write_to_db(table_name: str, data: DataFrame, db_name: str = "news.db"):
    """
    Write data to db

    :param table_name: table name, :type str
    :param data: DataFrame to save
    :param db_name: db name, type: str
    :return: None
    """
    con = create_engine(f'sqlite:///{DATA_PATH}{db_name}').connect()
    data.to_sql(f'{table_name}', con=con)
    print(f"Data is wrote to path {DATA_PATH}, with name {table_name}")


def read_from_csv(csv_name: str, sep: str = ",") -> DataFrame:
    """
    This method read data from csv file and  returns DataFrame

    :param sep: csv seperator, :type str
    :param csv_name: name of the csv, :type str
    :return: DataFrame
    """
    df = pd.read_csv(f"{DATA_PATH}{csv_name}", sep=sep)
    print(f"Data is read. Len of the data {len(df)} and columns {df.columns}")
    return df


def write_to_csv(csv_name: str, data: DataFrame):
    """
    This method write data from csv file and  returns DataFrame

    :param data: data to save, :type str
    :param csv_name: name of the csv, :type str
    :return: None
    """
    data.to_csv(f"{DATA_PATH}{csv_name}", index=False)
    print(f"Data is wrote to path {DATA_PATH}, with name {csv_name}")


def load_model(path: str):
    """
    This method loads the model

    :param path: path of the mode, :type str
    :return:
    """
    with open(path, 'rb') as file:
        pickle_model = pickle.load(file)

    return pickle_model
