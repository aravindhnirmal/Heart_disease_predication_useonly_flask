import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# load the heart disease dataset
df =  pd.read_csv('F:\sem-6\miniproone\dataset.csv')


# separate the target variable from the feature variables
X = df.drop(['target'], axis=1)
y = df['target']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create an instance of the StandardScaler class
scaler = StandardScaler()

# scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# train a logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# save the trained model as a .pkl file
with open('lrmodel.pkl', 'wb') as f:
    pickle.dump(model, f)
