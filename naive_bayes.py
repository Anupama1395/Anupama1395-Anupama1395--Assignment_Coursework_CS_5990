#-------------------------------------------------------------------------
# AUTHOR: Anupama singh
# FILENAME: naive_bayes.py
# SPECIFICATION: Train and evaluate Naïve Bayes classifier with grid search
# FOR: CS 5990- Assignment #4
# TIME SPENT: 3.5 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
train_df = pd.read_csv("weather_training.csv")
X_training = train_df.drop(columns=["Formatted Date", "Temperature (C)"])
y_training_raw = train_df["Temperature (C)"]


#update the training class values according to the discretization (11 values only)
discretizer = KBinsDiscretizer(n_bins=11, encode='ordinal', strategy='quantile')
y_training = discretizer.fit_transform(y_training_raw.values.reshape(-1, 1)).ravel()
bin_edges = discretizer.bin_edges_[0]


#reading the test data
test_df = pd.read_csv("weather_test.csv")
X_test = test_df.drop(columns=["Formatted Date", "Temperature (C)"])
y_test = test_df["Temperature (C)"]

# Function to check if predicted bin falls within ±15% of actual temperature
def is_prediction_correct(pred_bin, true_temp):
    pred_bin = int(round(pred_bin))
    pred_bin = max(0, min(pred_bin, len(bin_edges) - 2))  # prevent out-of-range access
    predicted_temp = (bin_edges[pred_bin] + bin_edges[pred_bin + 1]) / 2
    lower_bound = true_temp * 0.85
    upper_bound = true_temp * 1.15
    return lower_bound <= predicted_temp <= upper_bound

#loop over the hyperparameter value (s)
best_accuracy = 0
best_s = None

for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    y_pred_bins = clf.predict(X_test)
    correct_predictions = [
        is_prediction_correct(pred_bin, true_temp)
        for pred_bin, true_temp in zip(y_pred_bins, y_test)
    ]
    accuracy = np.mean(correct_predictions)

    # check if the calculated accuracy is higher than the previously one calculated
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_s = s
        print(f"Highest Naive Bayes accuracy so far: {best_accuracy:.2f}")
        print(f"Parameter: s = {s}")