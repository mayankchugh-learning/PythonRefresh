
import joblib

from sklearn.datasets import fetch_openml

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

dataset = fetch_openml(data_id=42890, as_frame=True, parser="auto")

data_df = dataset.data

target = 'Machine failure'
numeric_features = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]
categorical_features = ['Type']

print("Creating data subsets")

X = data_df[numeric_features + categorical_features]
y = data_df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

model_logistic_regression = LogisticRegression(n_jobs=-1)

print("Estimating Best Model Pipeline")

model_pipeline = make_pipeline(
    preprocessor,
    model_logistic_regression
)

param_distribution = {
    "logisticregression__C": [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
}

rand_search_cv = RandomizedSearchCV(
    model_pipeline,
    param_distribution,
    n_iter=3,
    cv=3,
    random_state=42
)

rand_search_cv.fit(Xtrain, ytrain)

print("Logging Metrics")
print(f"Accuracy: {rand_search_cv.best_score_}")

print("Serializing Model")

saved_model_path = "model.joblib"

joblib.dump(rand_search_cv.best_estimator_, saved_model_path)
