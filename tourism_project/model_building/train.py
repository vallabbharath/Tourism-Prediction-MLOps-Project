
# for data manipulation

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization

import joblib

# huggingface

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# mlflow

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

api = HfApi()

# ==============================

# Load train/test from HF dataset

# ==============================

Xtrain_path = "hf://datasets/vallabbharath/Tourism-Prediction-MLOps-Project-Data/Xtrain.csv"
Xtest_path  = "hf://datasets/vallabbharath/Tourism-Prediction-MLOps-Project-Data/Xtest.csv"
ytrain_path = "hf://datasets/vallabbharath/Tourism-Prediction-MLOps-Project-Data/ytrain.csv"
ytest_path  = "hf://datasets/vallabbharath/Tourism-Prediction-MLOps-Project-Data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest  = pd.read_csv(ytest_path).squeeze()

# ==============================

# Features

# ==============================

numeric_features = [
  'Age',
  'DurationOfPitch',
  'NumberOfFollowups',
  'PreferredPropertyStar',
  'NumberOfTrips',
  'Passport',
  'PitchSatisfactionScore',
  'OwnCar',
  'NumberOfChildrenVisiting',
  'MonthlyIncome'
]

categorical_features = [
  'TypeofContact',
  'Occupation',
  'Gender',
  'ProductPitched',
  'MaritalStatus',
  'Designation'
]

# ==============================

# Handle class imbalance

# ==============================

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# ==============================

# Preprocessing

# ==============================

preprocessor = make_column_transformer(
  (StandardScaler(), numeric_features),
  (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ==============================

# Model

# ==============================

xgb_model = xgb.XGBClassifier(
  scale_pos_weight=class_weight,
  random_state=42
)

# ==============================

# Hyperparameter grid

# ==============================

param_grid = {
  'xgbclassifier__n_estimators': [50, 75, 100],
  'xgbclassifier__max_depth': [2, 3],
  'xgbclassifier__colsample_bytree': [0.4, 0.6],
  'xgbclassifier__colsample_bylevel': [0.4, 0.6],
  'xgbclassifier__learning_rate': [0.01, 0.1],
  'xgbclassifier__reg_lambda': [0.4, 0.6],
}

# ==============================

# Pipeline

# ==============================

model_pipeline = make_pipeline(preprocessor, xgb_model)

# ==============================

# Training + MLflow tracking

# ==============================

with mlflow.start_run():
  grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
  grid_search.fit(Xtrain, ytrain)

  results = grid_search.cv_results_

  # log each parameter combination
  for i in range(len(results['params'])):
      param_set = results['params'][i]
      mean_score = results['mean_test_score'][i]
      std_score = results['std_test_score'][i]

      with mlflow.start_run(nested=True):
          mlflow.log_params(param_set)
          mlflow.log_metric("mean_test_score", mean_score)
          mlflow.log_metric("std_test_score", std_score)

  # best model
  mlflow.log_params(grid_search.best_params_)
  best_model = grid_search.best_estimator_

  # predictions
  classification_threshold = 0.45

  y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
  y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

  y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
  y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

  train_report = classification_report(ytrain, y_pred_train, output_dict=True)
  test_report  = classification_report(ytest, y_pred_test, output_dict=True)

  # log metrics
  mlflow.log_metrics({
      "train_accuracy": train_report['accuracy'],
      "train_precision": train_report['1']['precision'],
      "train_recall": train_report['1']['recall'],
      "train_f1-score": train_report['1']['f1-score'],
      "test_accuracy": test_report['accuracy'],
      "test_precision": test_report['1']['precision'],
      "test_recall": test_report['1']['recall'],
      "test_f1-score": test_report['1']['f1-score']
  })

  # save model
  model_path = "best_tourism_model_v1.joblib"
  joblib.dump(best_model, model_path)

  mlflow.log_artifact(model_path, artifact_path="model")
  print(f"Model saved as artifact at: {model_path}")

  # ==============================

  # Upload to HF Model Hub

  # ==============================

  repo_id = "vallabbharath/tourism-model"
  repo_type = "model"

  # Check if the model repo exists
  try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists. Using it.")
  except RepositoryNotFoundError:
    print(f"Model repo '{repo_id}' not found. Creating new model repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model repo '{repo_id}' created.")

  api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
  )

