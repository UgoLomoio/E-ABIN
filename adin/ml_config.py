class ML_config:
    """A class used for ML models configs."""

    def __init__(self, solver = "lbfsg", penalty = "l2", max_iter = 100, kernel = "linear", c = 1, k = 5, distance = "euclidean", n_estimators = 100, max_depth = 1, max_depth_dt = 1, min_samples_split = 2, train_split = 0.7, cross_val_k = 2):
        """
        Parameters
        ----------
        - solver: Logistic Regression solver
        - penalty: Logistic Regression penalty metrics
        - max_iter: Logistic Regression maximum number of iterations
        - kernel: kernel for SVC model
        - C: regularization parameter for SVC 
        - k: number of neighbours for KNN
        - distance: metric distance for KNN 
        - n_estimators: number of decision trees estimators inside a Random Forest
        - max_depth: the maximum depth of each decision tree inside a Random Forest
        - max_depth_dt: the maximum depth of a decision tree
        - min_samples_split: the minimum number of samples needed to split a decision tree leaf
        - train_split: portion the dataset to train the model
        - cross_val_k: number of splits for cross_validation
        """

        if max_depth_dt == -1:
            max_depth_dt = None
        if max_depth == -1:
            max_depth = None 

        self.lr = {"solver": solver, "penalty": penalty, "max_iter": max_iter}
        self.svm = {"kernel": kernel, "C": c}
        self.dt = {"max_depth": max_depth_dt, "min_samples_split": min_samples_split}
        self.knn = {"k": k, "metric": distance}
        self.rf = {"n_estimators": n_estimators, "max_depth": max_depth} 
        
        if train_split <= 1.0:
            self.train_split = train_split 
        else:
            self.train_split = 0.7
        self.test_split = 1.0 - self.train_split 
        
        if cross_val_k < 1:
            self.cross_val_k = 2 
        elif not isinstance(cross_val_k, int):
            self.cross_val_k = int(cross_val_k)
        else:
            self.cross_val_k = cross_val_k