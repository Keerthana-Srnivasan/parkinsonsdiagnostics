#author: keerthana srinivasan
#date: 6/16/2024
#description: surrogate model for parkinson's data. model can be trained recursively, and more data samples can be generated through latin hypercube sampling

#import libraries and dependencies
import pandas as pd #for data analysis and data organization
import numpy as np #numerical analysis and interpretation
import time #for the 20 second delay during training
import joblib #upload dataset
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score #training and validation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc #validation of accuracy
from sklearn.ensemble import RandomForestClassifier #random forest model
import matplotlib.pyplot as plt #for data interpretation and visualization

# function to load data from a csv file
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0)  # read csv file, setting the first column as the index
        return df  # return the dataframe
    except FileNotFoundError:
        print("File not found. Please check the file path.")  # print error if file is not found
        return None
    except Exception as e:
        print("An error occurred while loading the data:", str(e))  # print any other errors that may have occured
        return None

# function to split the data into training and testing sets
def collect_sample(df):
    if df is None or df.empty:
        print("DataFrame is empty or None.")  # check if dataframe is empty or None
        return None, None, None, None
    if 'status' not in df.columns:
        print("The 'status' column is missing in the DataFrame.")  # check if 'status' column is present
        return None, None, None, None
    X = df.drop(columns=["status"], axis=1)  # features (independent variables)
    y = df["status"]  # target (dependent variable)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)  # split data
    return train_X, train_y, test_X, test_y  # return split data

# function to normalize training and testing data
def normalize(train_X, test_X):
    train_stats = train_X.describe().transpose()  # get statistics for normalization
    def z_score(x, stats):
        return (x - stats['mean']) / stats['std']  # calculate z-score for normalization
    norm_train_X = z_score(train_X, train_stats)  # normalize training data
    norm_test_X = z_score(test_X, train_stats)  # normalize testing data
    return norm_train_X, norm_test_X  # return normalized data

# function to train randomforest model with hyperparameter tuning
def random_forest(norm_train_X, train_y):
    rf = RandomForestClassifier(random_state=1)  # initialize randomforestclassifier
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_features': ['sqrt', 'log2', None],  
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=10, cv=3, random_state=1, n_jobs=-1)  # randomized search for hyperparameters
    rf_random.fit(norm_train_X, train_y)  # fit model
    return rf_random.best_estimator_  # return best estimator

# function to evaluate model using cross-validation
def cross_validation_evaluation(df):
    if df is None:
        return None
    if df.empty:
        print("DataFrame is empty.")
        return None
    
    X = df.drop(columns=["status"], axis=1)  # features
    y = df["status"]  # target
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)  # split data
    norm_train_X, norm_test_X = normalize(train_X, test_X)  # normalize data
    print(norm_train_X.shape, y.shape)  # print shapes for debugging
    rf = RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=1
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # stratified k-fold cross-validation
    cv_scores = cross_val_score(rf, norm_train_X, train_y, cv=skf, scoring='accuracy')  # cross-validation scores
    return cv_scores  # return cross-validation scores

# function to evaluate the model
def evaluate_model(rf, norm_test_X, test_y):
    y_pred = rf.predict(norm_test_X)  # predictions
    print(confusion_matrix(test_y, y_pred))  # print confusion matrix
    print(classification_report(test_y, y_pred))  # print classification report
    disp = ConfusionMatrixDisplay(confusion_matrix(test_y, y_pred))  # display confusion matrix
    disp.plot()
    plt.show()  # show plot
    return rf, norm_test_X, test_y  # feturn model and test data

# function to train the model and plot ROC curve
def train(df):
    train_X, train_y, test_X, test_y = collect_sample(df)  # collect sample
    if train_X is None:
        return None, None
    norm_train_X, norm_test_X = normalize(train_X, test_X)  # normalize data
    rf = random_forest(norm_train_X, train_y)  # train model
    y_scores = rf.predict_proba(norm_test_X)[:, 1]  # get probabilities
    fpr, tpr, _ = roc_curve(test_y, y_scores)  # calculate ROC curve
    roc_auc = auc(fpr, tpr)  # calculate AUC
    plt.title('SM Model ROC Curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)  # plot ROC curve
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()  # display plot

    y_pred = rf.predict(norm_test_X)  # predictions
    accuracy = accuracy_score(test_y, y_pred)  # calculate accuracy
    evaluate_model(rf, norm_test_X, test_y)  # evaluate model
    return accuracy, roc_auc, rf  # return metrics and model

# main function to run the model training and evaluation
def model():
    df = load_data("audio_data.csv")  # load data
    if df is None:
        return
    cv_scores = cross_validation_evaluation(df)  # cross-validation evaluation
    if cv_scores is None:
        return
    mean_cv_score = cv_scores.mean()  # calculate mean cross-validation score
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean accuracy: {mean_cv_score}")

    train_X, train_y, test_X, test_y = collect_sample(df)  # collect sample
    if train_X is None:
        return
    norm_train_X, norm_test_X = normalize(train_X, test_X)  # normalize data
    rf = random_forest(norm_train_X, train_y)  # train model
    evaluate_model(rf, norm_test_X, test_y)  # evaluate model

    if mean_cv_score is not None and mean_cv_score >= 0.95:  # check if model is accurate
        print("Model Accurate.")
    else:
        while True:
            accuracy, roc_auc, rf = train(df)  # retrain model if accuracy is not sufficient
            if accuracy is None or roc_auc is None:
                continue
            print(f"Model accuracy: {accuracy}, ROC AUC: {roc_auc}")
            if accuracy >= 0.95:
                print("Model Accurate.")
                break
            else:
                print("Training...")
                time.sleep(20)  # sleep for 20 seconds before retraining
    try:
        joblib.dump(rf, 'C:\\Users\\Keerthana\\parkinsons_code\\finalized_model.sav')  # save model
    except Exception as e:
        print("An error occurred while saving the model:", str(e))  # print any errors while saving the model
        return

model()  # execute the model function
