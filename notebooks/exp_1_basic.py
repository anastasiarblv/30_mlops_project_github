import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner='anastasiarblv', repo_name='30_mlops_project_github', mlflow=True)
mlflow.set_experiment("exp_1: basic models")  # Name of the experiment in MLflow
MLflow_Tracking_remote = "https://dagshub.com/anastasiarblv/30_mlops_project_github.mlflow"
mlflow.set_tracking_uri(MLflow_Tracking_remote)

###################################### src/data/data_collection.py ###################################### 
data = pd.read_csv(r"C:\Users\honor\Desktop\water_potability.csv") # адрес файл с рабочего стола компа
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)
###################################### src/data/data_collection.py ###################################### 

###################################### src/data/data_prep.py ############################################
def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():  
            median_value = df[column].median()  
            df[column].fillna(median_value, inplace=True) 
    return df
train_processed_data_median = fill_missing_with_median(train_data)
test_processed_data_median = fill_missing_with_median(test_data)
###################################### src/data/data_prep.py ############################################

###################################### src/model/model_building.py ######################################
X_train = train_processed_data_median.drop(columns=["Potability"], axis=1)
y_train = train_processed_data_median["Potability"]

RANDOM_STATE = 42
classifiers = {'RandomForestClassifier': RandomForestClassifier(random_state = RANDOM_STATE),
               'LogisticRegression':LogisticRegression(random_state = RANDOM_STATE)}
RandomForestClassifier_space = {}
LogisticRegression_space = {}

params = {'RandomForestClassifier': RandomForestClassifier_space,
          'LogisticRegression': LogisticRegression_space}

models = list(classifiers.values())     
models_names =  list(classifiers.keys()) 
models_params = list(params.values())   

nabor = list(zip(models_names, models, models_params))
with mlflow.start_run(run_name="basic: Water_Potability_Models_Experiment"):
    for cur_model_name, cur_model, cur_model_param in nabor:
        with mlflow.start_run(run_name=cur_model_name, nested=True): 
            cur_model.fit(X_train, y_train)
            cur_model_filename = f"basic_{cur_model_name}.pkl"
            pickle.dump(cur_model, open(cur_model_filename, "wb"))
            cur_model = pickle.load(open(cur_model_filename, "rb"))
###################################### src/model/model_building.py ######################################

###################################### src/model/model_eval.py ##########################################    
            X_test = test_processed_data_median.drop(columns=["Potability"], axis=1)
            y_test = test_processed_data_median["Potability"]
            y_pred = cur_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1score = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1score)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for basic {cur_model_name}")
            plt.savefig(f"confusion_matrix_basic_{cur_model_name}.png")
            mlflow.log_artifact(f"confusion_matrix_basic_{cur_model_name}.png")

            mlflow.sklearn.log_model(cur_model, cur_model_name)
            mlflow.set_tag("author", "anastasiarblv")
            mlflow.set_tag("model", cur_model_name)
            mlflow.set_tag("model tuning", "No: basic model")
            mlflow.log_artifact(__file__)   
###################################### src/model/model_eval.py ########################################## 