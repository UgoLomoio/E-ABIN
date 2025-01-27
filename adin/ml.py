import platform

if platform.system() == "linux":
    import cudf.pandas
    cudf.pandas.install()

import pandas as pd
import numpy as np
#from matplotlib import pyplot
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
#import seaborn as sns

from .utils import validate_model#, plot_cm

import shap     

import plotly
from plotly import graph_objects as go 
import plotly.express as px

from screeninfo import get_monitors
from dash_shap_components import ForcePlot, ForceArrayPlot

monitor = get_monitors()[0]
mwidth, mheight = monitor.width, monitor.height 


def plot_roc_curve_(y, y_pred):
    global mwidth
    global mheight

    """
    input:
    y: array-like containing the actual target values
    y_pred: array-like containing the predicted target values
    output: A Plotly figure (ROC curve)
    """
    y = np.array(y)       # cast array-like into numpy array
    y_pred = np.array(y_pred)
    
    # Scale the values between 0 and 1
    y = MinMaxScaler(feature_range=(0, 1)).fit_transform(y.reshape(-1, 1))
    y_pred = MinMaxScaler(feature_range=(0, 1)).fit_transform(y_pred.reshape(-1, 1))
    
    # Get unique labels
    labels = np.unique(y)
    
    # Calculate the False Positive Rate, True Positive Rate, and thresholds
    fpr, tpr, threshold = roc_curve(y, y_pred)
    
    # Compute the Area Under the Curve (AUC)
    roc_auc = round(auc(fpr, tpr), 2)
    
    # Create Plotly figure
    fig = go.Figure()

    # Add ROC curve trace
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC {roc_auc}', line=dict(color='blue')))

    # Add the diagonal line for a random model
    fig.add_trace(go.Scatter(x=labels, y=labels, mode='lines', name='Random', line=dict(color='red', dash='dash')))

    # Set figure layout with labels and title
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.75, y=0.1), 
        height=int(mheight/4),
        width=int(mwidth/3)
    )
    
    return fig

def train_test_split(data, test_size = 0.7, target_name = 'Target'):
    #data: pd.DataFrame
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data.drop(target_name, axis=1).values, data[target_name].values, test_size=test_size, stratify = data[target_name])
    return X_train, X_test, y_train, y_test


def baselineComparison(X, y, params, scoring='accuracy', class_weight=True):
    '''Input:
    X: array-like (features)
    y: array-like (target values)
    model_params: Model Config object containing params for cross validation and single models
    scoring: metric for cross-validation (default: 'accuracy')
    class_weight: boolean, True for classification problems (default True)
    '''
   
    # Define the models
    models = []

    if class_weight:
        cw = 'balanced'
    else:
        cw = None
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('NB', GaussianNB()))

    models.append(('LR', LogisticRegression(solver=params.lr['solver'], max_iter=params.lr["max_iter"], class_weight=cw)))
    models.append(('KNN', KNeighborsClassifier(params.knn["k"], metric=params.knn["metric"])))
    models.append(('DT', DecisionTreeClassifier(max_depth = params.dt["max_depth"], min_samples_split = params.dt["min_samples_split"], class_weight=cw)))
    models.append(('SVM', SVC(kernel = params.svm["kernel"], C = params.svm["C"], gamma='scale', class_weight=cw, probability=True)))
    models.append(('RF', RandomForestClassifier(max_depth=params.rf["max_depth"], n_estimators=params.rf["n_estimators"], class_weight=cw)))

    results = []
    namesModels = []
    best_model = None
    best_nameModel = ''
    best_scores = 0.0

    # ROC Curve visualization setup
    fig_roc = go.Figure()
    mean_fpr = np.linspace(0, 1, 100)

    """
    counts, uqs = np.unique(y, return_counts=True)
    
    if min(counts) > 20:
        n_splits = 10
    elif min(counts) < 20 and min(counts) > 10:
        n_splits = 5
    else:
        n_splits = 2
    """
    n_splits = params.cross_val_k

    #check if the number of data and balancing of the dataset is sufficient to enable cross validation using this number of splits 

    uqs, counts = np.unique(y, return_counts=True)
    uq_count = {uq: count for uq, count in zip(uqs, counts)}

    if min(uq_count) < n_splits:
        n_splits = 2
    else:
        if uq_count[0] >= 2*uq_count[1] or uq_count[1] >= 2*uq_count[0]:
            n_splits = 2

    print("Cross-validation splits: ", n_splits)
    # Cross-validation for each model
    for nameModel, model in models:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

        aucs = []
        tprs = []

        for fold, (train, test) in enumerate(kfold.split(X, y)):
            X_train_res, y_train_res = X[train], y[train]
            X_test_res, y_test_res = X[test], y[test]

            model.fit(X_train_res, y_train_res)
            y_pred_prob = model.predict_proba(X_test_res)[:, 1]

            fpr, tpr, _ = roc_curve(y_test_res, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        fig_roc.add_trace(go.Scatter(
            x=mean_fpr, y=mean_tpr,
            mode='lines',
            name=f"{nameModel} (AUC = {mean_auc:.2f} +/- {std_auc:.2f})",
            line=dict(width=2)
        ))

        if scoring == 'roc_auc':
            scores = aucs
        else:
            scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)

        if np.mean(scores) > np.mean(best_scores):
            best_scores = scores
            best_nameModel = nameModel

        results.append(scores)
        namesModels.append(nameModel)
        msg = f"{nameModel}: {np.mean(scores):.4f} ({np.std(scores):.4f})"
        print(msg)

    # Final ROC Curve Plot Configuration
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines', line=dict(dash='dash', color='red'),
        showlegend=False, name='Random'
    ))
    fig_roc.update_layout(
        title="Mean ROC Curve during cross validation",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=int(mheight/4),
        width=int(mwidth/3),
        template='plotly_white'
    )

    # Boxplot for model comparison
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(y=results[0], name=namesModels[0], boxmean='sd'))

    for i, result in enumerate(results):
        fig_box.add_trace(go.Box(y=result, name=namesModels[i], boxmean='sd'))

    fig_box.update_layout(
        title="Algorithm Comparison during cross validation",
        yaxis_title=scoring.capitalize(),
        height=int(mheight/4),
        width=int(mwidth/3),
        template='plotly_white'
    )

    print(f'Best model: {best_nameModel} with {scoring}: {np.mean(best_scores):.4f}')
    
    return models, fig_roc, fig_box

def models_roc_curves(models, X_test, y_test):
    # Create the figure object
    fig = go.Figure()

    best_score = 0.0
    best_model = None
    colors = ["green", "black", "cyan", "orange", "blue", "purple", "pink"]

    # Add diagonal reference line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                             line=dict(color='red', dash='dash'),
                             showlegend=False, name='Random'))

    for i, (name, model) in enumerate(models):
        color = colors[i]

        # Predict probabilities
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plotting the ROC curve
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 line=dict(color=color, width=2),
                                 name=f'{name} (AUC = {roc_auc:.5f})'))

        # Keep track of the best model based on AUC
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = model

    # Update layout for better appearance
    fig.update_layout(
        title="ROC Curve on the test set (80%)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        showlegend=True,
        height=int(mheight/4),
        width=int(mwidth/3),
        template='plotly_white'
    )
    
    return fig, best_model, best_score

def plot_feature_importance(importances, genes):
    
    # Create a DataFrame for gene importances
    feature_importances = pd.DataFrame({
        'Gene': genes,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Select the top 10 most important genes
    top_genes = feature_importances.head(10)

    # Create an interactive bar plot using Plotly Express
    fig = px.bar(top_genes, x='Importance', y='Gene', 
                 orientation='h',  # Horizontal bar plot
                 title='Top 10 Most Important Genes',
                 labels={'Importance': 'Importance Score', 'Gene': 'Gene'},
                 color='Importance',
                 color_continuous_scale='viridis')

    # Update layout for better presentation
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Gene",
        width=int(mwidth/3),
        height=int(mheight/2),
        template='plotly_white'
    )

    return fig

def explain_model(model, model_name, X, genes, X_train = None):
     
    # Fit your model 
    
    if model_name == "DT" or model_name == "RF":
        explainer = shap.Explainer(model)
    elif model_name == "LR":
        explainer = shap.Explainer(model, X_train)
    elif model_name == "SVM" or model_name == "KNN":
        return None
    else:
        raise Exception("Model {} not supported.".format(model_name))

    shap_values = explainer.shap_values(X)

    # Convert SHAP values to a DataFrame
    # Use shap_values.values[:, :, 1] for binary classification if you want the positive class
    if shap_values.ndim == 3:
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    shap_df = pd.DataFrame(values, columns=genes)

    # Calculate mean absolute SHAP values per feature
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)

    # Create DataFrame for plotting
    feature_importances = pd.DataFrame({
        'Gene': mean_abs_shap.index,
        'Mean Absolute SHAP Value': mean_abs_shap.values
    }).head(10)  # Select top 10 genes

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
            x=feature_importances["Mean Absolute SHAP Value"].values,
            y=feature_importances["Gene"].values,
            orientation='h')
    )
    fig.update_layout(title = 'Top 10 Most Important Genes Based on SHAP Values', xaxis_title="Mean Absolute SHAP Value", yaxis_title="Gene")

    return fig 

def shap_summary(model, model_name, X, genes, X_train = None):
    
    # Create Explainer object that can calculate shap values
    if model_name == "DT" or model_name == "RF":
        explainer = shap.TreeExplainer(model)
    elif model_name == "LR":
        explainer = shap.Explainer(model, X_train)
    elif model_name == "SVM" or model_name == "KNN":
        return None
    else:
        raise Exception("Model {} not supported.".format(model_name))
     
    # Calculate Shap values
    shap_values = explainer.shap_values(X)
    
    # Make plot
    fig = shap.summary_plot(shap_values[:, :, 1], X, max_display=10, sort = True, feature_names=genes, show = True)

def shap_summary_plotly(model, model_name, X, targets, genes, patients, X_train = None):

    """
    Create a SHAP summary plot using Plotly.

    Parameters:
    - model: Trained model used for SHAP calculation.
    - X: Input data used to compute SHAP values.
    - targets: DataFrame containing target values with a column 'Target'.
    - genes: List of feature names corresponding to the columns of X.
    - patients: List of patient IDs corresponding to the rows of X.
    - mheight: Height of the plot (default is 600).
    - mwidth: Width of the plot (default is 800).

    Returns:
    - fig: Plotly figure object.
    """

    global mheight
    global mwidth 

    # Create Explainer object that can calculate shap values
    if model_name == "DT" or model_name == "RF":
        explainer = shap.Explainer(model)
    elif model_name == "LR":
        explainer = shap.Explainer(model, X_train)
    elif model_name == "SVM" or model_name == "KNN":
        return None
    else:
        raise Exception("Model {} not supported.".format(model_name))
     

    # Calculate Shap values
    shap_values = explainer.shap_values(X)
    
    # Calculate the mean SHAP value difference between targets for each feature
    if shap_values.ndim == 3:
        mean_shap_values_target1 = shap_values[:, :, 0].mean(axis=0)
        mean_shap_values_target2 = shap_values[:, :, 1].mean(axis=0)
        shap_value_diff = np.abs(mean_shap_values_target1 - mean_shap_values_target2)
    else:
        shap_value_diff = np.abs(shap_values.mean(axis=0)) 
        
    # Get indices of the top 10 genes based on SHAP value difference
    selected_genes_idx = np.argsort(shap_value_diff)[-10:]
    selected_genes = genes[selected_genes_idx]
    
    # Flatten the array for plotting: features x patients x targets
    # Prepare a list to store data for each target

    if shap_values.ndim == 3:
        data = []

        for target_idx in range(shap_values.shape[2]):
        
            #print(target_idx)
            # Extract SHAP values for the current target
            shap_values_flat = shap_values[:, selected_genes_idx, target_idx].flatten()
    
            df_genes = [gene for i in range(shap_values.shape[0]) for gene in selected_genes]
            #print(selected_genes.shape, shap_values_flat.shape)
            # Create a DataFrame
            df = pd.DataFrame({
                'shap_value': shap_values_flat,
                'gene': df_genes,
                'target': target_idx
            })
    
            data.append(df)
    else:
        
        shap_values_flat = shap_values[:, selected_genes_idx].flatten()
        data = []
        df_genes = [gene for i in range(shap_values.shape[0]) for gene in selected_genes]
        df = pd.DataFrame({
                'shap_value': shap_values_flat,
                'gene': df_genes,
                'target': 0
        })
  
        data.append(df)

    # Concatenate all data into a single DataFrame
    full_df = pd.concat(data)
    #print(full_df)

    full_df.sort_values(by=["shap_value"], ascending = True, inplace=True)    

    # Plot using Plotly Express
    fig = px.strip(full_df, y='gene', x='shap_value', color='target',
                     labels={'shap_value': 'SHAP Value', 'gene': 'Gene'},
                     title='SHAP Summary Plot by Target', stripmode='overlay', color_discrete_sequence=['cyan', 'pink'])


    # Customize layout if needed
    fig.update_layout(
        xaxis_title='Gene',
        yaxis_title='SHAP Value',
        legend_title='Target',
        height=int(mheight/2), 
        width=int(mwidth/3)
    )

    fig.update_layout(xaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro'),
              yaxis=dict(showgrid=True, gridcolor='WhiteSmoke', zerolinecolor='Gainsboro'), boxgap=0)
    fig.update_layout(plot_bgcolor='white')
    fig.update_traces(jitter=1)
    
    return fig 


def shap_force(model, model_name, X_test, y_test, genes, index = 0, class_id = 1, X_train = None):
    
    # Create Explainer object that can calculate shap values
    if model_name == "DT" or model_name == "RF":
        explainer = shap.Explainer(model)
    elif model_name == "LR":
        explainer = shap.Explainer(model, X_train)
    elif model_name == "SVM" or model_name == "KNN":
        return None
    else:
        raise Exception("Model {} not supported.".format(model_name))
    
    y = y_test[index]
    choosen_instance = X_test[index, :]
    shap_values = explainer.shap_values(choosen_instance)[:, class_id]
    #shap.initjs()
    shap.force_plot(explainer.expected_value[class_id], shap_values, choosen_instance, feature_names=genes, matplotlib=True, show = True)


def shap_force_plotly(model, X_test, y_test, genes, index = 0):
           
        explainer = shap.Explainer(model)
        y = y_test[index]
        choosen_instance = X_test[index, :]
        shap_values = explainer.shap_values(choosen_instance)[:, 0]

        # Sort by SHAP value
        sorted_data = sorted(zip(genes, shap_values), key=lambda x: x[1], reverse=True)
        sorted_features, sorted_shap_values = zip(*sorted_data)

        # Create a cumulative sum of SHAP values
        expected_value = explainer.expected_value[y] #expected value for the class y 
        cumulative_contributions = np.cumsum(sorted_shap_values) + expected_value

        # Create the figure
        fig = go.Figure()

        # Plot the cumulative contributions
        fig.add_trace(go.Scatter(
            x=list(range(len(sorted_features))),
            y=cumulative_contributions,
            mode='lines+markers',
            name='Cumulative Contribution',
            line=dict(color='blue'),
            marker=dict(size=8)
        ))

        # Add horizontal line for expected value
        fig.add_trace(go.Scatter(
            x=[0, len(sorted_features)-1],
            y=[expected_value, expected_value],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Expected Value'
        ))

        # Annotate the final prediction
        final_prediction = cumulative_contributions[-1]
        fig.add_trace(go.Scatter(
            x=[len(sorted_features)-1],
            y=[final_prediction],
            mode='markers+text',
            text=['Final Prediction'],
            textposition='top right',
            marker=dict(color='green', size=10),
            name='Final Prediction'
        ))

        # Update layout
        fig.update_layout(
            title='SHAP Force Plot',
            xaxis_title='Feature Index',
            yaxis_title='Prediction Value',
            showlegend=True
        )
        return fig 



def get_plot(gid, style, model, model_name, X_test, y_test, genes, index = 0, top_n = 20, X_train = None, class_id = 1, title=None):
     
    if model_name == "DT" or model_name == "RF":
        explainer = shap.Explainer(model)
    elif model_name == "LR":
        explainer = shap.Explainer(model, X_train)
    elif model_name == "SVM" or model_name == "KNN":
        return None
    else:
        raise Exception("Model {} not supported.".format(model_name))
     
    y = y_test[index]
    choosen_instance = X_test[index, :]

    shap_values = explainer.shap_values(choosen_instance)
    if shap_values.ndim == 2:
        shap_values = shap_values[:, class_id]
        expected_value = explainer.expected_value[class_id]
    else:
        expected_value = explainer.expected_value

    features = {i: {'effect': shap_values[i], 'value': choosen_instance[i]} for i in range(shap_values.shape[0])}
    featureNames = {i: gene for i, gene in enumerate(genes)}

    # Combine features and SHAP values into a list and sort by the absolute SHAP value (impact)
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]['effect']), reverse=True)
    
    # Select the top N features with the highest impact
    top_features = dict(sorted_features[:top_n])
    
    # Update the feature names accordingly
    top_featureNames = {i: featureNames[i] for i in top_features.keys()}

    return ForcePlot(
            id=gid,
            style=style,
            className='col-md-12',
            title=title,
            baseValue=expected_value,
            outNames=["Output Value"],
            features=top_features,
            featureNames=top_featureNames,
            hideBaseValueLabel=False,
            hideBars=False,
            labelMargin=0,
            plot_cmap=['#DB0011', '#000FFF'],
            # style={'width': '50vw'},
    )

def create_results_df(models, X_test, y_test):
    
    df = pd.DataFrame([], columns = ["Model Name", "Accuracy", "F1", "Sensitivity", "Specificity", "AUC score", "Precision"])
    
    for i, (model_name, model) in enumerate(models.items()):

        if model_name not in ["LR", "SVM", "KNN", "RF", "DT"]:
            continue
        else:
            print(model_name)
            y_pred = model.predict(X_test)
            _, metrics, msg = validate_model(y_test, y_pred, model_name)
            acc = float(round(metrics['accuracy']*100, 2))
            f1 = float(round(metrics['f1'], 2))
            sensitivity = float(round(metrics['sensitivity']*100, 2))
            specificity = float(round(metrics['specificity']*100, 2))
            auc = float(round(metrics['auc_score'], 4))
            precision = float(round(metrics['precision']*100, 2))
            df.loc[i] = [model_name, acc, f1, sensitivity, specificity, auc, precision]

    df = df.sort_values(by = "AUC score", ascending = False)
    return df