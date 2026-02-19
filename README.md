# sklearn-lite

A lightweight, educational implementation of a scikit-learn-like machine learning library in pure Python and NumPy.

## Features

- **Classification Models**: Logistic Regression, Decision Trees, Random Forests, Naive Bayes, Support Vector Classifiers
- **Regression Models**: Linear Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression
- **Clustering Algorithms**: K-Means, DBSCAN
- **Preprocessing**: StandardScaler, MinMaxScaler, OneHotEncoder
- **Model Selection**: K-Fold Cross-Validation, Grid Search, Cross-Validation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, MSE, MAE, R² Score
- **Consistent API**: Inspired by scikit-learn's fit-predict interface

## Installation

```bash
git clone https://github.com/sergeauronss01/sklearn-lite.git
cd sklearn-lite
pip install -r requirements.txt
```

## Quick Start

### Basic Classification

```python
from simpleml.linear_model import LogisticRegression
from simpleml.preprocessing import StandardScaler
from simpleml.model_selection import train_test_split
from simpleml.metrics import accuracy_score
import numpy as np

# Generate sample data
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
clf = LogisticRegression(n_iterations=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### Decision Tree Classifier

```python
from simpleml.tree import DecisionTreeClassifier

X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y = np.array([0, 1, 1, 0])

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)
predictions = clf.predict(X)
```

### Random Forest

```python
from simpleml.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### K-Means Clustering

```python
from simpleml.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X_new)
```

### Cross-Validation

```python
from simpleml.model_selection import cross_validate

scores = cross_validate(clf, X, y, cv=5)
print(f"Mean test score: {scores['mean_test_score']}")
```

### Grid Search

```python
from simpleml.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_iterations': [100, 500],
}

gs = GridSearchCV(clf, param_grid, cv=5)
gs.fit(X_train, y_train)
print(f"Best parameters: {gs.best_params_}")
```

## Module Structure

```
simpleml/
├── __init__.py              # Package initialization
├── base.py                  # Base classes (BaseEstimator, ClassifierMixin, etc.)
├── linear_model.py          # Linear regression and classification
├── tree.py                  # Decision tree models
├── ensemble.py              # Ensemble methods
├── cluster.py               # Clustering algorithms
├── naive_bayes.py           # Naive Bayes classifiers
├── svm.py                   # Support Vector Machines
├── preprocessing.py         # Data preprocessing
├── model_selection.py       # Model selection and evaluation
├── metrics.py               # Evaluation metrics
└── utils.py                 # Utility functions
```

## Available Models

### Supervised Learning

#### Classification
- `LogisticRegression`: Binary logistic regression with gradient descent
- `DecisionTreeClassifier`: Decision tree with Gini/Entropy splits
- `RandomForestClassifier`: Ensemble of decision trees
- `GaussianNaiveBayes`: Gaussian Naive Bayes classifier
- `MultinomialNaiveBayes`: Multinomial Naive Bayes for discrete features
- `LinearSVC`: Linear Support Vector Classifier

#### Regression
- `LinearRegression`: Ordinary least squares regression
- `DecisionTreeRegressor`: Decision tree regression
- `RandomForestRegressor`: Ensemble regression
- `SVR`: Support Vector Regression

### Unsupervised Learning

#### Clustering
- `KMeans`: K-Means clustering algorithm
- `DBSCAN`: Density-based clustering

### Preprocessing

- `StandardScaler`: Standardization (zero mean, unit variance)
- `MinMaxScaler`: Min-Max scaling to a fixed range
- `OneHotEncoder`: One-hot encoding for categorical features

### Model Selection

- `KFold`: K-Fold cross-validator
- `cross_validate()`: Cross-validation scoring
- `GridSearchCV`: Exhaustive grid search

## Metrics

All metrics are available in `simpleml.metrics`:

- `accuracy_score`: Classification accuracy
- `precision_score`: Precision for binary classification
- `recall_score`: Recall for binary classification
- `f1_score`: F1-Score for binary classification
- `confusion_matrix`: Confusion matrix
- `mean_squared_error`: Mean squared error
- `mean_absolute_error`: Mean absolute error
- `r2_score`: R² coefficient of determination

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run tests with coverage:

```bash
pytest tests/ --cov=simpleml
```

## Documentation

Each module is thoroughly documented with docstrings following NumPy style. Use `help()` in Python:

```python
from simpleml.linear_model import LogisticRegression
help(LogisticRegression)
```

## Design Philosophy

This library is designed to:

1. **Educate**: Clear, understandable implementations without external ML dependencies
2. **Mirror scikit-learn**: Familiar API for users of scikit-learn
3. **Be extensible**: Easy to add new models and algorithms
4. **Be tested**: Comprehensive test coverage

## Limitations

- No support for neural networks or deep learning
- No GPU acceleration
- Limited to basic algorithms (no advanced techniques like stacking, boosting beyond basic implementation)
- Not optimized for large-scale datasets

## Performance

This library is for educational purposes. For production use and performance-critical applications, use scikit-learn or similar production libraries.

## Contributing

To contribute:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License

## Author

sergeauronss01

## References

- Scikit-learn documentation: https://scikit-learn.org
- NumPy documentation: https://numpy.org
- Machine Learning fundamentals

---

**Note**: This is an educational project created to understand machine learning algorithms and best practices for library design. For production machine learning work, please use [scikit-learn](https://scikit-learn.org/).
