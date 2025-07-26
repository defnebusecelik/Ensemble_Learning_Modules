from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Base model
base_model = DecisionTreeClassifier(random_state=42)

# Bagging: train 10 trees on bootstrap samples and vote
bagging_model = BaggingClassifier(
    estimator=base_model,
    n_estimators=10,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=42
)

bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
