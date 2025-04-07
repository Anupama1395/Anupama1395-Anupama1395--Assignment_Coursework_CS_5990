# -------------------------------------------------------------------------# AUTHOR: 
# Anupama Singh
# FILENAME: roc_curve.py
# SPECIFICATION: Compute and plot ROC curve for decision tree classifier
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: 30 mins
# -------------------------------------------------------------------------

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('cheat_data.csv', sep=',', header=0)

# Prepare features and labels
X = []
y = []

def preprocess_row(row):
    refund = 1 if row['Refund'] == 'Yes' else 0
    single = 1 if row['Marital Status'] == 'Single' else 0
    divorced = 1 if row['Marital Status'] == 'Divorced' else 0
    married = 1 if row['Marital Status'] == 'Married' else 0
    taxable_income = float(row['Taxable Income'].replace('k', '').strip())
    return [refund, single, divorced, married, taxable_income]

# Process each row
for index, row in df.iterrows():
    X.append(preprocess_row(row))
    y.append(1 if row['Cheat'] == 'Yes' else 0)

X = np.array(X)
y = np.array(y)

# Split dataset
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3)

# No-skill prediction
ns_probs = [0 for _ in range(len(testy))]

# Train decision tree
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf.fit(trainX, trainy)

# Predict probabilities
dt_probs = clf.predict_proba(testX)[:, 1]

# Calculate AUC
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

print('No Skill: ROC AUC = %.3f' % ns_auc)
print('Decision Tree: ROC AUC = %.3f' % dt_auc)

# Plot ROC
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.title('ROC Curve')
pyplot.show()
