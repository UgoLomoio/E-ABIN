import io
from torch.utils.data import TensorDataset, DataLoader
import torch 
from deepgonet import DeepGONet_pl
import pytorch_lightning as pl
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler 
   
import sys 
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'adin')))
import utils  # now you can use functions from utils.py

# X_train, y_train, X_val, y_val: Load your numpy arrays as usual (see your TF code)
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.__version__)
torch.set_float32_matmul_precision('high')  # or 'medium'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

cwd = os.getcwd()
sep = os.sep
pardir = os.path.dirname(cwd)
    
# connection_matrix: Prepare list of numpy arrays for LGO
n_hidden = [1574, 1386, 951, 515, 255, 90]  # Example hidden layer sizes

deepgonet_path = cwd + sep + "Deep-GONet"
if not os.path.exists(deepgonet_path):
    raise FileNotFoundError(f"Deep-GONet path {deepgonet_path} does not exist. Please check the path.")
        
print("Loading connection matrices...")
adj_matrix = pd.read_csv(os.path.abspath(os.path.join(deepgonet_path,"adj_matrix.csv")),index_col=0)
first_matrix_connection = pd.read_csv(os.path.abspath(os.path.join(deepgonet_path,"first_matrix_connection_GO.csv")),index_col=0)
csv_go = pd.read_csv(os.path.abspath(os.path.join(deepgonet_path,"go_level.csv")),index_col=0)

n_classes = 1
n_hidden = [1574, 1386, 951, 515, 255, 90]


print("Preparing connection matrix...")
connection_matrix = []
connection_matrix.append(np.array(first_matrix_connection.values,dtype=np.float32))
connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(7)].loc[lambda x: x==1].index,csv_go[str(6)].loc[lambda x: x==1].index].values,dtype=np.float32))
connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(6)].loc[lambda x: x==1].index,csv_go[str(5)].loc[lambda x: x==1].index].values,dtype=np.float32))
connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(5)].loc[lambda x: x==1].index,csv_go[str(4)].loc[lambda x: x==1].index].values,dtype=np.float32))
connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(4)].loc[lambda x: x==1].index,csv_go[str(3)].loc[lambda x: x==1].index].values,dtype=np.float32))
connection_matrix.append(np.array(adj_matrix.loc[csv_go[str(3)].loc[lambda x: x==1].index,csv_go[str(2)].loc[lambda x: x==1].index].values,dtype=np.float32))
connection_matrix.append(np.ones((n_hidden[-1], n_classes),dtype=np.float32))
connection_matrix = [torch.tensor(elem, dtype=torch.float32).to(device) for elem in connection_matrix]  # Move connection matrices to device


keep_prob = 0.4
use_bn = False
lr_method = 'adam'
type_trainings = ["l1", "l2", "lgo"]
alpha = 1e-2
epochs = 200
batch_size = 4   #2**9
is_training = True  # Set to False for evaluation
lr = 0.001
force_redo = "lgo"

# ----------- Data Loading Utilities ----------

def preprocess_data(X, y):
    """ Preprocess the input data X: remove NaN values, standardize, etc. """
    idx_nan = ~pd.isnull(X).any(axis=1) & ~pd.isnull(y)
    X = X[idx_nan]
    y = y[idx_nan]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def prepare_dataloader(X, y, batch_size=512, is_train=True):
    X_tensor = torch.tensor(X, dtype=torch.float32)  # Keep on CPU
    y_tensor = torch.tensor(y, dtype=torch.long)     # or torch.float32 for regression/BCE
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0)


if __name__ == "__main__":

    datasets_path = cwd + sep + "use_case" + sep + "data" 
    datasets_names = os.listdir(datasets_path)

    for type_training in type_trainings:
        print(f"Training type: {type_training}")
        for dataset_name in datasets_names:
            print(f"Dataset: {dataset_name}")

            model_path = os.path.join(deepgonet_path, 'model', f'deepgonet_model_{type_training}_{dataset_name}.pth')
            if os.path.exists(model_path) and type_training != force_redo:
                print(f"Model already exists for {type_training} on {dataset_name}. Skipping training.")
                continue

            datasets_path = cwd + sep + "use_case" + sep + "data" 
            dataset_path = datasets_path + sep + dataset_name

            files = os.listdir(dataset_path)
            gse_files = [f for f in files if "series_matrix" in f]
            gse_file = gse_files[0] if gse_files else None

            if not gse_file:
                print(f"No series_matrix file found in {dataset_path}. Skipping dataset.")
                continue
                
            path = os.path.join(dataset_path, gse_file)
            with open(path, 'r') as f:
                decoded = f.read().encode('utf-8')
            skiprows, gene_data = utils.read_gene_expression(io.StringIO(decoded.decode('utf-8')))
            genes = gene_data.columns.tolist()
            genes = genes[:-1]  # Exclude the last column which is usually "!series_matrix_table_end"
            #print(genes)

            """
            if dataset_name in ["colorectal_cancer", "parkinson"]:
                connection_matrix_filtered = []
                first_matrix_connection_copy = first_matrix_connection.copy()
                adj_matrix_copy = adj_matrix.copy()
                print(first_matrix_connection_copy.index)
                first_matrix_connection_filtered = first_matrix_connection_copy[first_matrix_connection_copy.index.isin(genes)]
                adj_matrix_filtered = adj_matrix_copy.loc[first_matrix_connection_filtered.index, first_matrix_connection_filtered.index]
                connection_matrix_filtered.append(np.array(first_matrix_connection_filtered.values,dtype=np.float32))
                connection_matrix_filtered.append(np.array(adj_matrix_filtered.loc[csv_go[str(7)].loc[lambda x: x==1].index,csv_go[str(6)].loc[lambda x: x==1].index].values,dtype=np.float32))
                connection_matrix_filtered.append(np.array(adj_matrix_filtered.loc[csv_go[str(6)].loc[lambda x: x==1].index,csv_go[str(5)].loc[lambda x: x==1].index].values,dtype=np.float32))
                connection_matrix_filtered.append(np.array(adj_matrix_filtered.loc[csv_go[str(5)].loc[lambda x: x==1].index,csv_go[str(4)].loc[lambda x: x==1].index].values,dtype=np.float32))
                connection_matrix_filtered.append(np.array(adj_matrix_filtered.loc[csv_go[str(4)].loc[lambda x: x==1].index,csv_go[str(3)].loc[lambda x: x==1].index].values,dtype=np.float32))
                connection_matrix_filtered.append(np.array(adj_matrix_filtered.loc[csv_go[str(3)].loc[lambda x: x==1].index,csv_go[str(2)].loc[lambda x: x==1].index].values,dtype=np.float32))
                connection_matrix_filtered.append(np.ones((n_hidden[-1], n_classes),dtype=np.float32))
                connection_matrix_filtered = [torch.tensor(elem, dtype=torch.float32).to(device) for elem in connection_matrix_filtered]  # Move connection matrices to device

            else:
                connection_matrix_filtered = connection_matrix
            """
            
            train = np.load(os.path.join(dataset_path, 'X_train.npz'))  
            X_train = train['arr_0']
            y_train = train['arr_1']

            # Preprocess the data
            X_train, y_train = preprocess_data(X_train, y_train)

            val = np.load(os.path.join(dataset_path, 'X_test.npz'))  
            X_val = val['arr_0']
            y_val = val['arr_1']

            # Preprocess the validation data
            X_val, y_val = preprocess_data(X_val, y_val)

            n_input = X_train.shape[1]  # Assuming X_train is a 2D array with shape (num_samples, num_features)
            #n_classes = len(np.unique(y_train))  # Number of unique classes in the training labels
            print(f"Input size: {n_input}, Number of classes: {n_classes}")

            print("Preparing DataLoaders...")
            train_loader = prepare_dataloader(X_train, y_train, batch_size, is_train=True)
            val_loader = prepare_dataloader(X_val, y_val, batch_size, is_train=False)

            print("Creating DeepGONet model...")
            if type_training == "lgo":
                model = DeepGONet_pl(n_input, n_classes, n_hidden, connection_matrix_filtered, keep_prob, use_bn, lr_method, lr, type_training, alpha, device) 
            else:
                model = DeepGONet_pl(n_input, n_classes, n_hidden, None, keep_prob, use_bn, lr_method, lr, type_training, alpha, device)
            model = model.to(device)  # Move model to device
            #model = torch.compile(model)

            print("Training DeepGONet model...")
            trainer = pl.Trainer(fast_dev_run=False, max_epochs=epochs, precision=32, devices=1, accelerator='gpu' if torch.cuda.is_available() else 'cpu',     
                                num_sanity_val_steps=0,  # disables expensive/lightning validation
                                enable_checkpointing=False,  # disables checkpoints for debugging
                                logger=False  # disables logging for debugging
                                )
            trainer.fit(model, train_loader, val_loader)
            print("Training completed.")
            
            # Save the model
            print("Saving the model...")
            model_save_path = os.path.join(deepgonet_path, 'model')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            torch.save(model.state_dict(), os.path.join(model_save_path, f'deepgonet_model_{type_training}_{dataset_name}.pth'))
