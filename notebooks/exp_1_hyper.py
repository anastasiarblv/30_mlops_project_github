import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


dagshub.init(repo_owner='anastasiarblv', repo_name='30_mlops_project_github', mlflow=True)
mlflow.set_experiment("exp_1:  tunning models")  # Name of the experiment in MLflow
MLflow_Tracking_remote = "https://dagshub.com/anastasiarblv/30_mlops_project_github.mlflow"
mlflow.set_tracking_uri(MLflow_Tracking_remote)

###################################### src/data/data_collection.py ###################################### 
data = pd.read_csv(r"C:\Users\honor\Desktop\water_potability.csv") # адрес файл с рабочего стоала компа
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
RandomForestClassifier_space = {'n_estimators': [100, 200, 300, 500, 1000], 'max_depth': [None, 4, 5, 6, 10]}
LogisticRegression_space = {'penalty':['l1', 'l2'], 'C': np.logspace(0, 4, 10), 'max_iter': [50,75,100,200,300,400,500,700,800,1000]}

params = {'RandomForestClassifier': RandomForestClassifier_space,
          'LogisticRegression': LogisticRegression_space}

models = list(classifiers.values())     # [RandomForestClassifier(random_state=42)]
models_names  =  list(classifiers.keys()) # ['RandomForestClassifier']
models_params = list(params.values())   # [{'n_estimators': [100, 200, 300, 500, 1000], 'max_depth': [None, 4, 5, 6, 10]}]

nabor = list(zip(models_names, models, models_params))
#[('RandomForestClassifier',
#  RandomForestClassifier(random_state=42),
#  {'n_estimators': [100, 200, 300, 500, 1000], max_depth': [None, 4, 5, 6, 10]})]

with mlflow.start_run(run_name="tuning: Water_Potability_Models_Experiment"):
    for cur_model_name, cur_model, cur_model_param in nabor:
        random_search = RandomizedSearchCV(estimator=cur_model, param_distributions=cur_model_param, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42)
        with mlflow.start_run(run_name=cur_model_name, nested=True) as parent_run:
            random_search.fit(X_train, y_train)
            for i in range(len(random_search.cv_results_['params'])):
                with mlflow.start_run(run_name=f"Combination{i+1}", nested=True) as child_run:
                    mlflow.log_params(random_search.cv_results_['params'][i])  
                    mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])  
            best_cur_model = random_search.best_estimator_
            best_params_cur_model = random_search.best_params_
            mlflow.log_params(best_params_cur_model)
            best_cur_model.fit(X_train, y_train) # Train the model using the best parameters identified by RandomizedSearchCV
            best_cur_model_filename = f"tuning_{cur_model_name}.pkl" 
            pickle.dump(best_cur_model, open(best_cur_model_filename, "wb"))  # Save the trained model to a file for later use
            best_cur_model = pickle.load(open(best_cur_model_filename, "rb")) # Load the saved model from the file
###################################### src/model/model_building.py ######################################   

###################################### src/model/model_eval.py ##########################################           
            X_test = test_processed_data_median.drop(columns=["Potability"], axis=1)
            y_test = test_processed_data_median["Potability"]
            y_pred = best_cur_model.predict(X_test) # Make predictions on the test set using the loaded model
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
            plt.title(f"Confusion Matrix for tunning {cur_model_name}")
            plt.savefig(f"confusion_matrix_tuning_{cur_model_name}.png")
            mlflow.log_artifact(f"confusion_matrix_tuning_{cur_model_name}.png")

            mlflow.sklearn.log_model(best_cur_model, cur_model_name)
            mlflow.set_tag("author", "anastasiarblv")
            mlflow.set_tag("model", cur_model_name)
            mlflow.set_tag("model tuning", "Yes: RandomizedSearchCV")
            mlflow.log_artifact(__file__)    
###################################### src/model/model_eval.py ########################################## 