
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data_df = pd.read_csv("Bank_Telemarketing.csv")

target = 'subscribed'
numerical_features = ['Age', 'Duration(Sec)', 'CC Contact Freq', 'Days Since PC','PC Contact Freq']
categorical_features = ['Job', 'Marital Status', 'Education', 'Defaulter', 'Home Loan',
       'Personal Loan', 'Communication Type', 'Last Contacted', 'Day of Week',
       'PC Outcome']

print("Creating data subsets")

X = data_df[numerical_features + categorical_features]
y = data_df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = make_column_transformer(
    (numerical_pipeline, numerical_features),
    (categorical_pipeline, categorical_features)
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
