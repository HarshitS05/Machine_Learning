import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef

# --------------------------------------------------
# Step 1: Load the dataset
# --------------------------------------------------
data = pd.read_csv("yeast.csv")

# First column is protein name → not useful for prediction
X = data.iloc[:, 1:-1]   # features
y = data.iloc[:, -1]    # target (localization site)

# --------------------------------------------------
# Step 2: Define k-fold cross validation
# --------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --------------------------------------------------
# Step 3: Lists to store evaluation metrics
# --------------------------------------------------
recall_scores = []
f1_scores = []
mcc_scores = []

# --------------------------------------------------
# Step 4: Train and evaluate model using k-fold CV
# --------------------------------------------------
for train_index, test_index in kf.split(X):

    # Split data into training and testing sets
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create Decision Tree model
    model = DecisionTreeClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    recall_scores.append(recall_score(y_test, y_pred, average="macro"))
    f1_scores.append(f1_score(y_test, y_pred, average="macro"))
    mcc_scores.append(matthews_corrcoef(y_test, y_pred))

# --------------------------------------------------
# Step 5: Print average results
# --------------------------------------------------
print("Average Recall:", np.mean(recall_scores))
print("Average F1 Score:", np.mean(f1_scores))
print("Average MCC:", np.mean(mcc_scores))
