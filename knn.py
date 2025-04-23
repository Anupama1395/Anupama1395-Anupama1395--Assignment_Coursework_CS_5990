#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: knn.py
# SPECIFICATION: KNN model with hyperparameter tuning to predict weather using discretized temperature values
# FOR: CS 5990- Assignment #4
# TIME SPENT: 3.5 hours
#-----------------------------------------------------------*/

#importing some Python libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
#reading the test data
train_df = pd.read_csv('weather_training.csv')
test_df = pd.read_csv('weather_test.csv')

# dropping the non-numeric 'Formatted Date' column and separating features/labels
X_training = train_df.drop(columns=['Formatted Date', 'Temperature (C)']).values.astype('f')
y_training = train_df['Temperature (C)'].values.astype('f')
X_test = test_df.drop(columns=['Formatted Date', 'Temperature (C)']).values.astype('f')
y_test = test_df['Temperature (C)'].values.astype('f')

# initialize tracking of best accuracy
highest_accuracy = 0

#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            clf = KNeighborsRegressor(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)
            
            #make the KNN prediction for each test sample and start computing its accuracy
            correct = 0
            for x_testSample, y_testSample in zip(X_test, y_test):
                predicted_value = clf.predict([x_testSample])[0]
                percentage_diff = 100 * abs(predicted_value - y_testSample) / abs(y_testSample)
                if percentage_diff <= 15:
                    correct += 1

            accuracy = correct / len(y_test)
            
            # check if this is the highest accuracy so far
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {accuracy:.2f}, Parameters: k={k}, p={p}, w= '{w}'")

            




