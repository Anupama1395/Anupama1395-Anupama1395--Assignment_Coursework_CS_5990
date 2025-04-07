# -------------------------------------------------------------------------
# AUTHOR: Anupama Singh
# FILENAME: decision_tree.py
# SPECIFICATION: Train and evaluate decision trees using different training datasets
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']
# Load the test data once
test_df = pd.read_csv('cheat_test.csv', sep=',', header=0)
test_data = np.array(test_df.values)[:, 1:]

def preprocess_row(row):
    refund = 1 if row[0] == 'Yes' else 0
    marital_status = row[1]
    single = 1 if marital_status == 'Single' else 0
    divorced = 1 if marital_status == 'Divorced' else 0
    married = 1 if marital_status == 'Married' else 0
    taxable_income_str = row[2].replace('k', '').strip()
    taxable_income = float(taxable_income_str)
    return [refund, single, divorced, married, taxable_income]

def new_func(row):
    taxable_income = float(row[2])
    return taxable_income

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    # Preprocess training data
    for row in data_training:
        X.append(preprocess_row(row))
        Y.append(1 if row[3] == 'Yes' else 2)

    X = np.array(X)
    Y = np.array(Y)

    total_accuracy = 0

    for i in range(10):
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # Uncomment to show plots each time
        # tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
        #                class_names=['Yes', 'No'], filled=True, rounded=True)
        # plt.show()

        correct = 0
        total = 0

        for row in test_data:
            test_features = preprocess_row(row)
            true_label = 1 if row[3] == 'Yes' else 2
            prediction = clf.predict([test_features])[0]
            if prediction == true_label:
                correct += 1
            total += 1

        accuracy = correct / total
        total_accuracy += accuracy

    avg_accuracy = total_accuracy / 10
    print(f"Final accuracy when training on {ds}: {round(avg_accuracy, 2)}")
