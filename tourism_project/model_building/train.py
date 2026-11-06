
import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# 1. Define paths and configuration
#-----------------------------------

BASE_DIR = "tourism_project"

TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")

DEPLOYMENT_DIR = os.path.join(BASE_DIR, "deployment")
MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "best_model.joblib")

TARGET_COLUMN = "ProdTaken"

os.makedirs(DEPLOYMENT_DIR, exist_ok=True)


# 2. Load training and testing datasets
#--------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# Separate features and target
X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN]


# 3. Identify numerical and categorical columns
#----------------------------------------------

numerical_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# 4. Create preprocessing pipelines
#----------------------------------

# Numerical preprocessing:
# Fill missing values and scale numerical features
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical preprocessing:
# Fill missing values and apply one-hot encoding
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ]
)


# 5. Define model pipeline
#-------------------------

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ))
    ]
)


# 6. Define hyperparameter grid
# -------------------------------

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}


# 7. Configure MLflow experiment
# --------------------------------

mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run(run_name="RandomForest_GridSearch"):

    # Log general configuration
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("test_size", 0.20)
    mlflow.log_param("scoring", "f1")
    mlflow.log_param("random_state", 42)


    # 8. Perform hyperparameter tuning
    # -----------------------------------

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    print("\nStarting hyperparameter tuning...")

    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print("\nBest cross-validation F1 score:")
    print(grid_search.best_score_)

    # Log best hyperparameters and CV score
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric(
        "best_cv_f1_score",
        grid_search.best_score_
    )


    # 9. Evaluate model on test dataset
    # --------------------------------------

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    # ROC-AUC calculation
    y_probability = best_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probability)


    # 10. Log evaluation metrics to MLflow
    # ---------------------------------------

    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_roc_auc", roc_auc)

    print("\nModel Evaluation Results")
    print("------------------------")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC:", roc_auc)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    # 11. Save best model for deployment
    # ----------------------------------------

    joblib.dump(best_model, MODEL_PATH)

    print("\nBest model saved at:", MODEL_PATH)

    # Log the model as an MLflow artifact
    mlflow.sklearn.log_model(
        best_model,
        name="tourism_random_forest_model"
    )

    # Log saved model file as an artifact too
    mlflow.log_artifact(MODEL_PATH)

    print("\nMLflow experiment logging completed!")
