import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error,mean_squared_log_error, median_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

data = pd.read_csv('Housing.csv')

# Preview the first few rows of the data
df=data

##LABEL ENCODER
data_encoded = data.copy()

binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
label_encoder = LabelEncoder()
for column in binary_columns:
    data_encoded[column] = label_encoder.fit_transform(data_encoded[column])


## OneHotEncoder
data_encoded = data_encoded.copy()

columns_to_encode = ['furnishingstatus']
one_hot_encoder = OneHotEncoder()
encoded_data = one_hot_encoder.fit_transform(data_encoded[columns_to_encode]).toarray()

encoded_columns = one_hot_encoder.get_feature_names_out(columns_to_encode)
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

data_encoded = data_encoded.drop(columns_to_encode, axis=1)
data_encoded = pd.concat([data_encoded, encoded_df], axis=1)
data=data_encoded

X = data.drop('price', axis=1)  # Assuming 'price' is the target column
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LinearRegression().fit(X, y)

import joblib
# 3. Save the model
joblib.dump(model, 'house_model.pkl')
print("Model saved successfully!")