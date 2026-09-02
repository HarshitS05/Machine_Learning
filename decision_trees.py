import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef

data = pd.read_csv("yeast.csv")


X = data.iloc[:, 1:-1]   
y = data.iloc[:, -1]    

kf = KFold(n_splits=5, shuffle=True, random_state=42)

recall_scores = []
f1_scores = []
mcc_scores = []

for train_index, test_index in kf.split(X):

    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    recall_scores.append(recall_score(y_test, y_pred, average="macro"))
    f1_scores.append(f1_score(y_test, y_pred, average="macro"))
    mcc_scores.append(matthews_corrcoef(y_test, y_pred))

print("Average Recall:", np.mean(recall_scores))
print("Average F1 Score:", np.mean(f1_scores))
print("Average MCC:", np.mean(mcc_scores))
