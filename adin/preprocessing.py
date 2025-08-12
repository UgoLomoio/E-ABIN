import platform
from tqdm import tqdm

if platform.system() == "linux":
    import cudf.pandas
    cudf.pandas.install()

import pandas as pd
import numpy as np

def count_unnamed(data: pd.DataFrame, case_sensitive: bool = True, verbose: bool = False) -> int:
    """
    Counts the number of columns in a DataFrame whose names contain "Unnamed".

    Args:
        data (pd.DataFrame): Input DataFrame to check.
        case_sensitive (bool): If True, the check is case-sensitive (default). If False, it's case-insensitive.
        verbose (bool): If True, prints the names of matching columns.

    Returns:
        int: Count of columns with "Unnamed" in their name.

    Raises:
        ValueError: If input is not a pandas DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input 'data' must be a pandas DataFrame.")
    
    search_term = "Unnamed" if case_sensitive else "unnamed"
    
    matching_cols = [ 
        col for col in data.columns
        if search_term in (str(col).lower() if not case_sensitive else str(col))
    ] 

    
    if verbose:
        for col in matching_cols:
            print(col)
    
    return len(matching_cols)

    

def replace_nan_with_mode_and_rename(dataframe, ann_df, target_name="Target", need_rename=True, fallback_impute='mean'):
    """
    Imputes NaN values in a DataFrame using mode per target class, renames columns based on ann_df,
    and drops columns with unknown gene symbols.

    Args:
        dataframe (pd.DataFrame): Input DataFrame with columns to impute and a target column.
        ann_df (pd.DataFrame): Annotation DataFrame for renaming (e.g., with 'ID', 'SYMBOL').
        target_name (str): Name of the target column for grouping.
        need_rename (bool): If True, rename columns to gene symbols.
        fallback_impute (str): If mode is NaN, fallback to 'mean', 'median', or None (skip).

    Returns:
        pd.DataFrame: Processed DataFrame with imputed values and renamed columns (Target dropped).
    """
    # Ensure dataframe is numeric (excluding target)
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

    # Compute modes per target class once (for all columns)
    mode_df = dataframe.groupby(target_name).agg(lambda x: pd.Series.mode(x).iloc[0] if not x.mode().empty else np.nan).reset_index()

    # If fallback is needed, compute it (e.g., means or medians)
    if fallback_impute:
        fallback_df = dataframe.groupby(target_name).agg(fallback_impute).reset_index()

    # Prepare annotation dataframe (do this once, outside the loop)
    ann_df.columns = [c.upper().replace("_", " ") for c in ann_df.columns]
    id_col = 'ID' if 'ID' in ann_df.columns else 'NAME' if 'NAME' in ann_df.columns else None
    symbol_col = next((col for col in ['SYMBOL', 'GENE SYMBOL', 'GENE NAME'] if col in ann_df.columns), None)

    if id_col is None:
        print("Warning: Couldn't find 'ID' or 'NAME' column in ann_df.")
    if symbol_col is None:
        print(f"Warning: Couldn't find symbol column in ann_df. Columns: {ann_df.columns}")

    # Convert ID/NAME to string for matching
    if id_col:
        ann_df[id_col] = ann_df[id_col].astype(str)

    columns = [col for col in dataframe.columns if col != target_name]  # Exclude target
    rename_dict = {}
    cols_to_drop = []

    # Loop over columns for renaming and deciding drops
    for col in tqdm(columns, desc="Processing columns", disable=len(columns) < 1000):  # Progress bar; disable for small sets
        col_str = str(col)

        # Find gene symbol from ann_df
        if id_col and symbol_col:
            result = ann_df[ann_df[id_col] == col_str]
            if not result.empty:
                gene_name = result[symbol_col].values[0]
                if pd.isna(gene_name):
                    cols_to_drop.append(col_str)
                    continue  # Skip to next column
                rename_dict[col_str] = gene_name
            else:
                print(f"Warning: No match for {col_str} in ann_df. Keeping original name.")
                gene_name = col_str  # Fallback to original
        else:
            gene_name = col_str  # Fallback if no annotation columns

    # Drop columns with unknown genes
    dataframe = dataframe.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with unknown gene symbols.")

    # Impute NaNs using vectorized groupby operations (more efficient than per-column loops)
    def impute_group(group):
        col_name = group.name  # Current column
        for target_class in mode_df[target_name]:
            mode_value = mode_df.loc[mode_df[target_name] == target_class, col_name].values[0]
            
            if pd.isna(mode_value) and fallback_impute:
                fallback_value = fallback_df.loc[fallback_df[target_name] == target_class, col_name].values[0]
                mode_value = fallback_value if pd.notna(fallback_value) else np.nan
            
            if pd.notna(mode_value):
                mask = (dataframe[target_name] == target_class) & dataframe[col_name].isna()
                count_nans = mask.sum()
                if count_nans > 0:
                    print(f"Filling NA in {col_name} (class {target_class}): {count_nans} NaNs with {mode_value}")
                    dataframe.loc[mask, col_name] = mode_value
        
        # Check for remaining NaNs
        if dataframe[col_name].isna().any():
            print(f"Warning: Column {col_name} still has NaNs after imputation.")
        
        return group

    # Apply imputation to all relevant columns
    value_columns = [col for col in dataframe.columns if col != target_name]
    dataframe[value_columns] = dataframe.groupby(target_name)[value_columns].transform(impute_group)

    # Rename if needed
    if need_rename:
        dataframe = dataframe.rename(columns=rename_dict)

    # Drop target column
    dataframe = dataframe.drop(columns=[target_name])

    return dataframe

def replace_nan_with_mode_and_rename_old(dataframe, ann_df, target_name="Target", need_rename = True):
    
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


def rename_columns(dataframe: pd.DataFrame, ann_df: pd.DataFrame, target_name: str = "Target", drop_target: bool = True) -> pd.DataFrame:
    """
    Renames columns in a DataFrame using gene symbols from an annotation DataFrame.
    Drops columns with unknown (NaN) gene symbols and optionally drops the target column.

    Args:
        dataframe (pd.DataFrame): Input DataFrame with columns to rename.
        ann_df (pd.DataFrame): Annotation DataFrame (e.g., with 'ID', 'SYMBOL').
        target_name (str): Name of the target column (to exclude from renaming).
        drop_target (bool): If True, drop the target column at the end.

    Returns:
        pd.DataFrame: DataFrame with renamed columns and unknowns dropped.
    """
    # Prepare annotation dataframe (normalize columns once)
    ann_df.columns = [c.upper().replace("_", " ") for c in ann_df.columns]
    
    # Detect ID and symbol columns dynamically
    id_col = next((col for col in ['ID', 'NAME'] if col in ann_df.columns), None)
    symbol_col = next((col for col in ['SYMBOL', 'GENE SYMBOL', 'GENE NAME'] if col in ann_df.columns), None)
    
    if id_col is None:
        raise ValueError("Couldn't find 'ID' or 'NAME' column in ann_df.")
    if symbol_col is None:
        raise ValueError(f"Couldn't find symbol column in ann_df. Available columns: {ann_df.columns}")
    
    # Convert to dict for fast lookups: {ID: symbol}
    ann_df[id_col] = ann_df[id_col].astype(str)  # Ensure string matching
    ann_dict = dict(zip(ann_df[id_col], ann_df[symbol_col]))
    
    # Get columns to process (exclude target)
    columns = [col for col in dataframe.columns if col != target_name]
    
    rename_dict = {}
    cols_to_drop = []
    
    # Loop with progress (use tqdm if available, else fallback)
    progress_bar = tqdm(columns, desc="Renaming columns", disable=len(columns) < 1000)
    for col in progress_bar:
        col_str = str(col)
        
        gene_name = ann_dict.get(col_str, np.nan)  # Fast dict lookup
        
        if pd.isna(gene_name):
            cols_to_drop.append(col)
        else:
            rename_dict[col] = gene_name
        
        # Optional: Update progress description with current status
        progress_bar.set_postfix({"Processed": f"{len(rename_dict) + len(cols_to_drop)}/{len(columns)}"})
    
    # Apply drops and renames efficiently
    dataframe = dataframe.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with unknown gene symbols.")
    
    dataframe = dataframe.rename(columns=rename_dict)
    
    if drop_target and target_name in dataframe.columns:
        dataframe = dataframe.drop(columns=[target_name])
    
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
            dataframe = rename_columns(dataframe, ann_df)
            unnamed = count_unnamed(dataframe)
            if unnamed > 0:
                raise Exception("Input gene expression data might be corrupted. Found {} unnamed genes.".format(unnamed))
    return dataframe