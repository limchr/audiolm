import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

y_label = "best_val_loss"

# Load the data from the CSV file
data = pd.read_csv("results/parameter_search.csv")

# Display the first few rows of the dataframe to understand its structure
print("First few rows of the data:")
print(data.head())

# filter out invalid runs
data = data[data.best_val_loss > -1]

# Extract the input variables and the outcome variable
X = data.drop(columns=[y_label])  # Assuming "Outcome" is the name of the outcome variable
y = data[y_label]

# Basic statistics of the input variables
print("\nSummary statistics of input variables:")
print(X.describe())

print("\nBest trials so far:")
best_trials = data.sort_values(y_label).head(10)
print(best_trials)

print("\nStats of best trials:")
print(best_trials.describe())

# Correlation matrix to identify relationships between variables
correlation_matrix = X.corr()
print("\nCorrelation matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Correlation Matrix')
plt.xticks(np.arange(len(X.columns)), X.columns, rotation=90)
plt.yticks(np.arange(len(X.columns)), X.columns)
plt.show()

# Scatter plots of input variables against outcome variable
for column in X.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(X[column], y, alpha=0.5)
    plt.title(f"{column} vs "+y_label)
    plt.xlabel(column)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()
