### =====================================================
# ADIN: Anomaly Detection in Individual Networks      
### =====================================================

## Python Version Compatibility

Compatible Python Versions: Python 3.x

Not Supported: Python 3.12 (due to some external library incompatibilities)

Tested On: Python 3.10

## Requirements
- Python 3.x
- pip (Python package installer)

## Installation Instructions
### For CUDA Users
To install the required dependencies for CUDA support, run the following commands:

```
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
python main.py
```


