# -------------------------------------------------------------------------
# AUTHOR: Anupama Singh
# FILENAME: pca_analysis.py
# SPECIFICATION: Perform PCA and determine feature importance by removing one feature at a time
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 3 hours
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data

df = pd.read_csv('heart_disease_dataset.csv')

# Print column names to verify
print("Columns in dataset:", df.columns.tolist())

#Create a training matrix without the target variable (Heart Diseas)
target_column = 'Cholesterol'

df_features = df.drop(columns=[target_column])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

#Get the number of features

num_features = df_features.shape[1]
# Store results
results = {}

# Run PCA for 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    reduced_data = np.delete(scaled_data, i, axis=1)

    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)

    #Store PC1 variance and the feature removed
    results[df_features.columns[i]] = pca.explained_variance_ratio_[0]
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    

# Find the maximum PC1 variance
best_feature_to_remove = max(results, key=results.get)
highest_variance = results[best_feature_to_remove]



#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {highest_variance} when removing {best_feature_to_remove}")





