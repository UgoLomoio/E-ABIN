# E-ABIN: an Explainable module for Anomaly detection in BIological Networks 

## Python Version Compatibility

Compatible Python Versions: Python < 3.13
Tested on: Python 3.12

## Requirements
- Python 3.12
- pip (Python package installer)
- anaconda

For Python 3.13 compatibility:
- Remove isn_tractor requirement from "requirementstorchcuda" or requirementstorch"

## Installation Instructions
### For CUDA Users (NVIDIA GPUs - recommended)
To install the required dependencies for CUDA support, run the following commands:

```
conda create -n eabin python=3.12 pip setuptools
conda activate eabin
pip install -r requirementscuda.txt
pip install -r requirementstorchcuda.txt
```

### For Non-CUDA Users
If you do not have CUDA support, install the standard dependencies with:

```
pip install -r requirements.txt
pip install -r requirementstorch.txt
```

### For Windows Users
Make sure you have Microsoft Visual C++ 14.0 or greater installed. You can obtain it from "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/


## How to Run the Application
Once the dependencies are installed, you can run the application with the following command:

```
python app.py
```


## How to Use the Application

See the [user guide](UserGuide.pdf).



## Installation Requirements

This subsection provides a detailed description and motivation for the installation requirements of E-ABIN. E-ABIN is a Python-based GUI (developed and tested on Python 3.10) utilizing Dash, a framework developed by Plotly Technologies Inc. for creating low-code graphical interfaces and dashboards. The GUI visualization module employs Dash, while the plots within the dashboards are generated using the matplotlib, plotly and seaborn libraries.

The machine learning binary classification module is built on the scikit-learn library. It encompasses various functionalities such as random dataset splitting into training and test sets, stratified k-fold cross-validation, and the training and validation of commonly used machine learning models including Logistic Regression (LR), K Nearest Neighbours (KNN), Decision Trees (DT), Random Forest (RF), and Support Vector Machine (SVM). Model predictions are explained using the Shap library, which employs a game-theoretic approach to interpret model decisions. Due to high RAM demands, Shap values are disabled and unavailable for KNN and SVM models.

Deep learning analyses are conducted using a PyTorch-based module capable of performing both binary classification and anomaly detection. PyTorch is renowned for its robust capabilities in creating, training, and validating deep learning models. Binary classification tasks utilize an explainable torch-geometric-based Graph Convolutional Network (GCN), while anomaly detection leverages models from the PyGOD library, which is designed explicitly for Graph Outlier Detection using PyTorch. Torch Geometric enhances the efficiency of Graph Neural Network model creation, training, and validation.

GPU acceleration enables network creation and deep learning modules using torch-based implementations. Additionally, CUDA users can use GPU acceleration to expedite file reading and preprocessing through cudf-pandas.

The network building module can create two different types of networks: 
- Convergence-Divergence network created using the cosine similarity function available in the torch library;
- Individual Specific Networks available using the ISN-tractor library. ISN-tractor is a library that can create interactome based Individual Specific Networks that highlights patients' diversity.
    
Network visualization is based on networkx and dynamically updated using the dash-cytoscape library.



