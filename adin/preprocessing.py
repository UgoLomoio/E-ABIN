import pandas as pd
import numpy as np

def count_unnamed(data):
  count = 0
  for gene in data.columns:
    if "Unnamed" in str(gene):
      #print(gene)
      count+=1
  return count
    

def replace_nan_with_mode_and_rename(dataframe, ann_df, target_name="Target", need_replace = True):
    
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    mode_df = dataframe.groupby(target_name).agg(lambda x: pd.Series.mode(x).iloc[0] if not x.mode().empty else np.nan).reset_index()
    columns = dataframe.columns[:-1]
    rename = {}
    
    # Replace NaN values in each value column with the corresponding mode
    for j, col in enumerate(columns):
        if j+1 in np.arange(1, len(columns), 1000):
            print("\r", j+1, "/", len(columns), end = "")
        
        deleted_col = False

        result = ann_df[ann_df['Name'] == col]
        if not result.empty:
            gene_name = result['Symbol'].values[0]
            if pd.isna(gene_name):
                deleted_col = True
                del dataframe[col]
                #print("The gene associated with", col, "is unknown")
            else:
                rename[col] = gene_name
                #print(f'The gene associated with {col} is {gene_name}')
        else:
            gene_name = col 
 
        if not deleted_col:
            if need_replace:
                if dataframe[col].isna().any().sum() > 0:
                    for target_class in mode_df[target_name]:
                        mode_value = mode_df.loc[mode_df[target_name] == target_class, col].values[0]

                        if pd.notna(mode_value):
                            print(f"Filling NA column {gene_name} (class {target_class}): {dataframe[dataframe[target_name] == target_class][col].isna().sum()} NaNs to be filled with {mode_value}")
                        
                            # Fill NaN values for the specific target class
                            dataframe.loc[dataframe[target_name] == target_class, col] = dataframe.loc[dataframe[target_name] == target_class, col].fillna(mode_value)

                            if dataframe[dataframe[target_name] == target_class][col].isna().any():  # Check if NaNs still exist
                                print(f"Column {col} still has NaN values after filling for class {target_class}.")

            #print(f'{col} not found in the annotation file.')
    
    dataframe.rename(columns = rename, inplace = True)
    dataframe.drop(columns=["Target"], inplace = True)
   
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


def preprocess(dataframe, ann_df, need_rename = True):
   
    needed = check_if_preprocessing_is_needed(dataframe)
    if needed:
        if dataframe.isnull().any().any():
            print("Found None gene expression values, replacing them with a target-based mode approach.")
            need_replace = True
        dataframe = replace_nan_with_mode_and_rename(dataframe, ann_df, need_replace = need_replace)
        unnamed = count_unnamed(dataframe)
        if unnamed > 0:
            raise Exception("Input gene expression data might be corrupted. Found {} unnamed genes.".format(unnamed))
        
    return dataframe