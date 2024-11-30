import mlflow
import dagshub

MLflow_Tracking_remote = "https://dagshub.com/anastasiarblv/30_mlops_project_github.mlflow"
mlflow.set_tracking_uri(MLflow_Tracking_remote)
dagshub.init(repo_owner='anastasiarblv', repo_name='30_mlops_project_github', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)