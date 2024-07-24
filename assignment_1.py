import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, 5],
    'C': [1, 2, 3, np.nan, 5],
    'D': [1, 2, 3, 4, np.nan]
}
df = pd.DataFrame(data)

def knn_imputation(df):
    imputer = KNNImputer(n_neighbors=2)
    df_knn_imputed = imputer.fit_transform(df)
    return pd.DataFrame(df_knn_imputed, columns=df.columns)

def regression_imputation(df):
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    df_regression_imputed = imputer.fit_transform(df)
    return pd.DataFrame(df_regression_imputed, columns=df.columns)

def multiple_imputation(df):
    imputer = IterativeImputer()
    df_multiple_imputed = imputer.fit_transform(df)
    return pd.DataFrame(df_multiple_imputed, columns=df.columns)

def random_forest_imputation(df):
    df_rf_imputed = df.copy()
    for column in df.columns:
        if df[column].isnull().any():
            train_data = df[df[column].notnull()]
            test_data = df[df[column].isnull()]
            X_train = train_data.drop(columns=column)
            y_train = train_data[column]
            X_test = test_data.drop(columns=column)
            model = RandomForestRegressor(n_estimators=100, random_state=0)
            model.fit(X_train, y_train)
            df_rf_imputed.loc[df[column].isnull(), column] = model.predict(X_test)
    return df_rf_imputed

def bayesian_imputation(df):
    imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
    df_bayesian_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_bayesian_imputed

df_knn_imputed = knn_imputation(df)
df_regression_imputed = regression_imputation(df)
df_multiple_imputed = multiple_imputation(df)
df_rf_imputed = random_forest_imputation(df)
df_bayesian_imputed = bayesian_imputation(df)

print("Original Data:")
print(df)
print("\nKNN Imputed Data:")
print(df_knn_imputed)
print("\nRegression Imputed Data:")
print(df_regression_imputed)
print("\nMultiple Imputed Data:")
print(df_multiple_imputed)
print("\nRandom Forest Imputed Data:")
print(df_rf_imputed)
print("\nBayesian Imputed Data:")
print(df_bayesian_imputed)
