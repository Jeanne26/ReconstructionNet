import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


#path : data/diabetes_prediction_dataset.csv
def load_diabetes(path):
    data = pd.read_csv(path)
    X = data.drop(columns=["diabetes"])
    y = data["diabetes"]
    return X, y


#j'au utilise le labelencoder au lieu du one hot pour eviter d'avoir trop de dimensions
def process_diabetes(X_train, X_val, X_test):
    categorical_cols = X_train.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col]   = le.transform(X_val[col])
        X_test[col]  = le.transform(X_test[col])

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler


#faire le split sur le X_train et y_train pour separer les classes 
def split_class(X,y):
    class_data = {}
    for cls in y.unique():
        class_data[cls] = X[y==cls]
    return class_data