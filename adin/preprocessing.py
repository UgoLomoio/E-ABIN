import pandas as pd
import numpy as np

def count_unnamed(data):
  count = 0
  for gene in data.columns:
    if "Unnamed" in str(gene):
      #print(gene)
      count+=1
  return count
    
def replace_nan_with_mode(dataframe, target_name="Target"):
    mode_df = dataframe.groupby(target_name).agg(lambda x: pd.Series.mode(x).iloc[0] if not x.mode().empty else np.nan).reset_index()
    columns = dataframe.columns[:-1]
    # Replace NaN values in each value column with the corresponding mode
    for j, col in enumerate(columns):
        if j+1 in np.arange(1, len(columns), 1000):
            print("\r", j+1, "/", len(columns), end = "")
        dataframe[col] = dataframe[col].fillna(mode_df[col])

    # Drop the mode columns as they are no longer needed
    #dataframe = dataframe.drop(columns=[f'{col}_mode' for col in columns])
    dataframe = dataframe.drop(columns=["Target"])
    return dataframe

def single_dataframe(data, group):
   dataframe = data.join(group)
   return dataframe


def check_if_preprocessing_is_needed(dataframe):
   
    needed = False
    if dataframe.isnull().any().any():
       needed = True
    if count_unnamed(dataframe) > 0:
       needed = True
    return needed


def preprocess(dataframe):
   
    needed = check_if_preprocessing_is_needed(dataframe)
    if needed:
        if dataframe.isnull().any().any():
            print("Found None gene expression values, replacing them with a target-based mode approach.")
            dataframe = replace_nan_with_mode(dataframe)
        unnamed = count_unnamed(dataframe)
        if unnamed > 0:
            raise Exception("Input gene expression data might be corrupted. Found {} unnamed genes.".format(unnamed))
        
    return dataframe