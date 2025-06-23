import platform

if platform.system() == "linux":
    import cudf.pandas
    cudf.pandas.install()

from numba.core.target_extension import Target
import pandas as pd
import numpy as np

def count_unnamed(data):
  count = 0
  for gene in data.columns:
    if "Unnamed" in str(gene):
      #print(gene)
      count+=1
  return count
    

def replace_nan_with_mode_and_rename(dataframe, ann_df, target_name="Target", need_rename = True):
    
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')
    mode_df = dataframe.groupby(target_name).agg(lambda x: pd.Series.mode(x).iloc[0] if not x.mode().empty else np.nan).reset_index()
    columns = dataframe.columns[:-1]
    rename = {}
    
    # Replace NaN values in each value column with the corresponding mode
    for j, col in enumerate(columns):
        if j+1 in np.arange(1, len(columns), 1000):
            print("\r", j+1, "/", len(columns), end = "")
        
        col = str(col)

        deleted_col = False

        ann_df.columns = [c.upper().replace("_", " ") for c in list(ann_df.columns)]
    
        if "ID" in ann_df.keys():
            ann_df['ID'] = ann_df['ID'].astype(str)
            result = ann_df[ann_df['ID'] == col]
        elif "NAME" in ann_df.keys():
            ann_df['NAME'] = ann_df['NAME'].astype(str)
            result = ann_df[ann_df['NAME'] == col]
        else:
            print(f"Warning: Error while performing renaming to gene symbol. Couldn't find ID or NAME column.")

        if not result.empty:
            if 'SYMBOL' in result.keys():
                gene_name = result['SYMBOL'].values[0]
                #print(1, gene_name)
            elif 'GENE SYMBOL' in result.keys():
                gene_name = result['GENE SYMBOL'].values[0]
                #print(2, gene_name)
            elif 'GENE NAME' in result.keys():
                #print(3, gene_name)
                gene_name = result['GENE NAME'].values[0]
            else:
                gene_name = np.nan
                print(f"Warning: Error while performing renaming to gene symbol. Annotation dataframe columns: {ann_df.columns}")

            if pd.isna(gene_name):
                deleted_col = True
                del dataframe[col]
               #print("The gene associated with", col, "is unknown")
            else:
                rename[col] = gene_name
               #print(f'The gene associated with {col} is {gene_name}')
        else:
            print("Dataframe 'result' is empty")
            gene_name = col 
 
        if not deleted_col:
            #if need_rename:
                if dataframe[col].isna().any().sum() > 0:
                    for target_class in mode_df[target_name]:
                        mode_value = mode_df.loc[mode_df[target_name] == target_class, col].values[0]

                        if pd.notna(mode_value):
                            count_nans = dataframe[dataframe[target_name] == target_class][col].isna().sum()
                            if count_nans > 0:
                                print(f"Filling NA column {gene_name} (class {target_class}): {count_nans} NaNs to be filled with {mode_value}")
                        
                            # Fill NaN values for the specific target class
                            dataframe.loc[dataframe[target_name] == target_class, col] = dataframe.loc[dataframe[target_name] == target_class, col].fillna(mode_value)

                            if dataframe[dataframe[target_name] == target_class][col].isna().any():  # Check if NaNs still exist
                                print(f"Column {col} still has NaN values after filling for class {target_class}.")

            #print(f'{col} not found in the annotation file.')
    
    if need_rename:
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

def rename(dataframe, ann_df):
    
    columns = dataframe.columns[:-1]
    rename = {}
    
    # Replace NaN values in each value column with the corresponding mode
    for j, col in enumerate(columns):
        if j+1 in np.arange(1, len(columns), 1000):
            print("\r", j+1, "/", len(columns), end = "")
        
        deleted_col = False
        result = ann_df[ann_df['ID'] == col]#Name
        if not result.empty:
            if 'Symbol' in result.keys():
                gene_name = result['Symbol'].values[0]
            elif 'Gene Symbol' in result.keys():
                gene_name = result['Gene Symbol'].values[0]
            if pd.isna(gene_name):
                deleted_col = True
                del dataframe[col]
                #print("The gene associated with", col, "is unknown")
            else:
                rename[col] = gene_name
                #print(f'The gene associated with {col} is {gene_name}')
        else:
            gene_name = col 
 

    dataframe.rename(columns = rename, inplace = True)
    dataframe.drop(columns=["Target"], inplace = True)
   
    return dataframe



def preprocess(dataframe, ann_df, rate_drop = 0.01, need_rename = True):
   
    needed = check_if_preprocessing_is_needed(dataframe)
    if needed:
        print("Handle null values and renaming")
        if dataframe.isnull().any().any():
            print("Found None gene expression values, replacing them with a target-based mode approach.")
            need_rename = True

        num_nan_drop = int(rate_drop * dataframe.shape[0])
        if num_nan_drop == 0:
            num_nan_drop = 1
        nan_columns = dataframe.isna().sum()
        if "Target" in nan_columns.keys():
            print("Warning: Target column has NaN values!!")
            del nan_columns["Target"]

        columns_to_drop = [col for col, count in nan_columns.items() if count >= num_nan_drop]
        n = len(columns_to_drop)
        print(f"Dropping {n} columns with {num_nan_drop} NAN values or higher")
        dataframe = dataframe.drop(columns_to_drop, axis=1)
        dataframe = replace_nan_with_mode_and_rename(dataframe, ann_df, need_rename = need_rename)
        unnamed = count_unnamed(dataframe)
        if unnamed > 0:
            raise Exception("Input gene expression data might be corrupted. Found {} unnamed genes.".format(unnamed))
    else:
        print("Renaming")
        if need_rename:
            dataframe = rename(dataframe, ann_df)
            unnamed = count_unnamed(dataframe)
            if unnamed > 0:
                raise Exception("Input gene expression data might be corrupted. Found {} unnamed genes.".format(unnamed))
    return dataframe