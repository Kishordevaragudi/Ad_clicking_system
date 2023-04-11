# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r'D:\static\Dataset\advertising.csv')

dataset=dataset.drop(columns=['Ad Topic Line','City','Country','Timestamp'],axis=1)

X = dataset[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = dataset['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()

#Fitting model with trainig data
regressor.fit(X_train, y_train)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[30, 45, 60000,300,1]]))