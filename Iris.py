import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the .data file into a pandas DataFrame as a CSV
file_path = '/Users/nas/Documents/Learning/Data Science/Iris/iris/iris.data'  # Replace with the actual file path if needed

# Read the CSV file
df = pd.read_csv(file_path, header = None)
num_entries = len (df)

df.iloc[:, -1] = df.iloc[:, -1].str.replace('Iris-', '', regex=False)


# Assuming the last column is the target variable (class labels)
X = df.iloc[:, :-1]  # Features (all columns except the last one)
y = df.iloc[:, -1]   # Target (last column)

# Split the dataset into training and testing sets (80:20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 5  # You can choose an appropriate value for k
knn = KNeighborsClassifier(n_neighbors=k)

# Train the KNN model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model's efficacy
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print accuracy with 4 decimal places
print(f"Accuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)


# List of k values to evaluate
k_values = list(range(1, 80))  # For example, testing k from 1 to 20
accuracies = []

# Evaluate KNN performance for each k value
for k in k_values:
    # Initialize the KNN classifier with the current k
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train the KNN model
    knn.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Define tick size
tick_size = 5

# Plotting accuracy vs k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors (k)')
plt.grid(True)

# Set x-axis labels at intervals of tick_size
plt.xticks(range(min(k_values), max(k_values) + 1, tick_size))

plt.show()