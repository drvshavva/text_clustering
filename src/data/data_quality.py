import pandas as pd
from pandas import DataFrame
from os.path import dirname

RESULTS_PATH = dirname(dirname(dirname(__file__))) + "/results/files/"


def create_data_quality_report(pandas_df: DataFrame, label_name: str) -> DataFrame:
    """
    This method prints data quality analysis results

    :param pandas_df: dataFrame, :type DataFrame
    :param label_name: label column name, :type str
    :return: DataFrame
    """

    len_df = len(pandas_df)
    unique_classes = pandas_df[label_name].unique()
    columns = list(pandas_df.columns)
    file = open(f"{RESULTS_PATH}data_quality.txt", "w")
    file.write("#################################################\n")
    file.write("############ DATA QUALITY RESULT ################\n")
    file.write("#################################################\n\n")
    file.write(f"Number of sample in data set:{len_df}.\n")
    file.write(f"Number of classes in data set: {len(unique_classes)} and "
               f"they are: {unique_classes}.\n")
    file.write(f"Columns in data set:{columns}.\n\n")
    file.write(f"{pandas_df.info()}.\n\n")
    file.write("\n\n############## SUMMARY STATISTICS ###############\n\n")
    file.write(f"{pandas_df.describe()}.\n\n")

    file.write("\n\n############## NULL PERCENTAGES #################\n\n")
    percent_missing = pandas_df.isnull().sum() * 100 / len(pandas_df)
    missing_value_df = pd.DataFrame({'column_name': pandas_df.columns,
                                     'percent_missing': percent_missing})
    for i, column in enumerate(pandas_df.columns):
        file.write(f"Column: {column}  percent of null values:  %{percent_missing[i]}.\n")

    file.close()

    file1 = open(f"{RESULTS_PATH}data_quality.txt", "r")
    lines = file1.readlines()
    for line in lines:
        print(f"{line.strip()}")

    return missing_value_df
