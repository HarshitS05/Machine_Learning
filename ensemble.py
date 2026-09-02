import pandas as pd
import numpy as np


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, f1_score, matthews_corrcoef

data = pd.read_csv("yeast.csv")

X = data.iloc[:, 1:-1]  
y = data.iloc[:, -1]   

kf = KFold(n_splits=5, shuffle=True, random_state=42)


models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest (Bagging)": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "AdaBoost (Boosting)": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        random_state=42
    )
}


for name, model in models.items():

    recalls, f1s, mccs = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        recalls.append(recall_score(y_test, y_pred, average="macro"))
        f1s.append(f1_score(y_test, y_pred, average="macro"))
        mccs.append(matthews_corrcoef(y_test, y_pred))

    print("\nModel:", name)
    print("Average Recall:", np.mean(recalls))
    print("Average F1 Score:", np.mean(f1s))
    print("Average MCC:", np.mean(mccs))
