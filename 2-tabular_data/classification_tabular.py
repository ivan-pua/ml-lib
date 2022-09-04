'''
Author: Ivan Pua
Description: A modular code for preprocessing tabular data, 
training a simple ML model, and scoring it
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier


def tabular_data_classification():
    
    print("Reading data...")
    df = pd.read_csv("data.csv")
    df.drop("customerID", axis = 1, inplace=True)# Drop id column
    
    print("Preprocessing data...")
    
    y = df.pop('Churn') # Label
    
    # Separating Numeric and Categorical Features
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    all_features = list(df.columns)
    categorical_features = [c for c in all_features if c not in numeric_features]    
    
    # Extra preprocessing for this specific dataset
    for x in numeric_features:
        df[x] = pd.to_numeric(df[x],errors='coerce') 
           
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OrdinalEncoder(dtype = 'int64'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = AdaBoostClassifier(n_estimators=100, random_state=0)
    
    print("Training data...")

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                    ('classifier', model)
                   ])
    
    
    pipeline.fit(X_train, y_train)
    recall = pipeline.score(X_test, y_test)
    print(f"Accuracy (recall) is {round(recall, 4)}\n")
    
    
if __name__ == '__main__':
    
    # TODO: add arguments for inputs
    tabular_data_classification()
