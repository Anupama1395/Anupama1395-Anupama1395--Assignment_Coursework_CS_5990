# -------------------------------------------------------------------------
# AUTHOR: Anupama Singh
# FILENAME: pca_analysis.py
# SPECIFICATION: Perform PCA and determine feature importance by removing one feature at a time
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 3 hours
# -----------------------------------------------------------*/

# Import necessary Python libraries
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('heart_disease_dataset.csv')

# Print column names to verify
print("Columns in dataset:", df.columns.tolist())

# all columns for PCA
df_features = df.copy()

# Standardize the dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# Set number of iterations (ensuring it doesn't exceed available features)
num_iterations = min(10, len(df_features.columns))

# Store results
results = {}

# Keep track of removed features to ensure uniqueness
removed_features = set()

# Run PCA exactly 10 times, removing a different feature in each iteration
for iteration in range(num_iterations):
    # Get list of available features that haven't been removed yet
    available_features = list(set(df_features.columns) - removed_features)
    
    # Ensure there is a feature to remove
    if not available_features:
        print("No more features left to remove.")
        break  # Exit loop if all features are removed
    
    # Randomly select a feature to remove
    feature_to_remove = random.choice(available_features)
    removed_features.add(feature_to_remove)  # Track removed feature

    # Create dataset without the selected feature
    reduced_data = df_features.drop(columns=[feature_to_remove])

    # Standardize the reduced dataset
    scaled_reduced_data = scaler.fit_transform(reduced_data)

    # Apply PCA
    pca = PCA()
    pca.fit(scaled_reduced_data)  #Use the standardized version

    # Store the explained variance of the first principal component (PC1)
    pc1_variance = pca.explained_variance_ratio_[0]
    results[feature_to_remove] = pc1_variance

    # Print each iteration result
    print(f"Iteration {iteration+1}: Removed '{feature_to_remove}', PC1 Variance: {pc1_variance:.6f}")

# Identify the feature removal that resulted in the highest PC1 variance
best_feature_to_remove = max(results, key=results.get)
highest_variance = results[best_feature_to_remove]

# Print the final result
print("\nFinal Result:")
print(f"The highest PC1 variance ({highest_variance:.6f}) was found when removing '{best_feature_to_remove}'.")
