# simpleml

A lightweight, educational implementation of a scikit-learn-like machine learning library in pure Python and NumPy.

## Features

- **Classification Models**: Logistic Regression, Decision Trees, Random Forests, Naive Bayes, Support Vector Classifiers
- **Regression Models**: Linear Regression, Decision Tree Regression, Random Forest Regression, Support Vector Regression
- **Clustering Algorithms**: K-Means, DBSCAN (with working `predict`)
- **Preprocessing**: StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
- **Model Selection**: K-Fold Cross-Validation, Grid Search, Cross-Validation
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC, MSE, MAE, MAPE, R² Score
- **Consistent API**: Inspired by scikit-learn's fit-predict interface

## Requirements

- Python >= 3.9
- NumPy >= 2.0

> **Note:** This library requires **NumPy 2.0+**. Earlier versions used deprecated APIs
> (`np.trapz`, `np.array(copy=False)`) that were removed or changed in NumPy 2.0.

## Installation

```bash
pip install simpleml-auronss
```

But if you want to have the folder locally you can do it that way:

```bash
git clone https://github.com/sergeauronss01/simpleml.git
cd simpleml
pip install -e .
```

The `-e` flag installs the package in editable mode — changes to the source are
reflected immediately without reinstalling.

Alternatively, if you do not want to install the package, add this to
`pyproject.toml` so pytest can find the source:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

## Quick Start

> **Import paths:** All models live under `simpleml.models`, utilities under
> `simpleml.utils`, and metrics under `simpleml.metrics`. The examples below
> use the correct paths.

### Basic Classification

```python
from simpleml.models.linear import LogisticRegression
from simpleml.utils.preprocessing import StandardScaler, train_test_split
from simpleml.metrics.classification import accuracy_score
import numpy as np

# Generate sample data
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

# Preprocess
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
clf = LogisticRegression(n_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

### Linear Regression

```python
from simpleml.models.linear import LinearRegression
from simpleml.metrics.regression import r2_score

reg = LinearRegression(learning_rate=0.01, n_iter=1000, alpha=0.0001)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred)}")
```

### Decision Tree Classifier

```python
from simpleml.models.tree import DecisionTreeClassifier

X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
y = np.array([0, 1, 1, 0])

clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
clf.fit(X, y)
predictions = clf.predict(X)
```

### Random Forest

```python
from simpleml.models.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=42)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### Support Vector Machines

```python
from simpleml.models.svm import LinearSVC, SVR

# Classification — predict() returns original class labels (e.g. {0, 1}),
# not the internal {-1, +1} encoding.
svc = LinearSVC(C=1.0, max_iter=1000, random_state=42)
svc.fit(X_train, y_train)
labels = svc.predict(X_test)   # returns {0, 1}, not {-1, 1}

# Regression
svr = SVR(C=1.0, epsilon=0.1, max_iter=1000)
svr.fit(X_train, y_train)
values = svr.predict(X_test)
```

### Naive Bayes

```python
from simpleml.models.naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes

# Gaussian — for continuous features
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
predictions = gnb.predict(X_test)

# Multinomial — for discrete count features (e.g. word counts)
# Works best when classes differ in *which* features are active,
# not just in their overall magnitude.
mnb = MultinomialNaiveBayes(alpha=1.0)   # alpha controls Laplace smoothing
mnb.fit(X_counts_train, y_train)
predictions = mnb.predict(X_counts_test)
```

### K-Means Clustering

```python
from simpleml.models.clustering import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.predict(X_new)
print(f"Inertia: {kmeans.inertia_}")
```

### DBSCAN Clustering

```python
from simpleml.models.clustering import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X)
print(db.labels_)          # -1 means noise/outlier

# predict() assigns new points to the nearest core point's cluster
# or returns -1 if no core point is within eps distance.
new_labels = db.predict(X_new)
```

### Preprocessing

```python
from simpleml.utils.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    OneHotEncoder,
    train_test_split,
)

# Standardise (zero mean, unit variance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Scale to [0, 1]
mms = MinMaxScaler()
X_minmax = mms.fit_transform(X_train)

# Generate polynomial features
# e.g. [a, b] → [1, a, b, a², ab, b²] for degree=2
pf = PolynomialFeatures(degree=2, include_bias=True)
X_poly = pf.fit_transform(X_train)

# One-hot encode categorical columns
enc = OneHotEncoder()
X_encoded = enc.fit_transform(X_categorical)

# Split — note the parameter is `random_seed`, not `random_state`
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_seed=42
)
```

### Cross-Validation

```python
from simpleml.utils.model_selection import cross_validate, KFold

scores = cross_validate(clf, X, y, cv=5)
print(f"Mean test score:  {scores['mean_test_score']:.4f}")
print(f"Std  test score:  {scores['std_test_score']:.4f}")
print(f"Mean train score: {scores['mean_train_score']:.4f}")

# Custom splitter
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(clf, X, y, cv=kf)
```

### Grid Search

```python
from simpleml.utils.model_selection import GridSearchCV

# Note: use `n_iter`, not `n_iterations`
param_grid = {
    'learning_rate': [0.01, 0.1],
    'n_iter': [100, 500],
}

gs = GridSearchCV(clf, param_grid, cv=5)
gs.fit(X_train, y_train)
print(f"Best parameters: {gs.best_params_}")
print(f"Best CV score:   {gs.best_score_:.4f}")
predictions = gs.predict(X_test)
```

### Metrics

```python
from simpleml.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    CrossEntropyLoss,
    softmax,
)
from simpleml.metrics.regression import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

# Classification
print(accuracy_score(y_true, y_pred))
print(precision_score(y_true, y_pred))
print(recall_score(y_true, y_pred))
print(f1_score(y_true, y_pred))
print(roc_auc_score(y_true, y_probs))
confusion_matrix(y_true, y_pred)
classification_report(y_true, y_pred)

# Regression
print(mean_squared_error(y_true, y_pred))
print(mean_absolute_error(y_true, y_pred))
print(mean_absolute_percentage_error(y_true, y_pred))
print(r2_score(y_true, y_pred))
```

## Module Structure

```
Scratch-ML-Library/
├── .gitignore
├── LICENSE
├── README.md
├── conftest.py              ← shared pytest fixtures and sys.path setup
├── pyproject.toml
├── requirements.txt
├── src/
│   └── simpleml/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── base.py      ← BaseEstimator, RegressorMixin, ClassifierMixin,
│       │                       TransformerMixin, ClusterMixin
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── classification.py   ← accuracy, precision, recall, F1, ROC-AUC,
│       │   │                          CrossEntropyLoss, softmax, confusion_matrix
│       │   └── regression.py       ← MSE, MAE, MAPE, R²
│       ├── models/
│       │   ├── __init__.py
│       │   ├── clustering.py       ← KMeans, DBSCAN
│       │   ├── ensemble.py         ← RandomForestClassifier, RandomForestRegressor
│       │   ├── linear.py           ← LinearRegression, LogisticRegression
│       │   ├── naive_bayes.py      ← GaussianNaiveBayes, MultinomialNaiveBayes
│       │   ├── svm.py              ← LinearSVC, SVR
│       │   └── tree.py             ← DecisionTreeClassifier, DecisionTreeRegressor
│       ├── nn/
│       │   └── __init__.py         ← reserved for future neural network support
│       └── utils/
│           ├── __init__.py
│           ├── model_selection.py  ← KFold, cross_validate, GridSearchCV
│           ├── preprocessing.py    ← StandardScaler, MinMaxScaler,
│           │                          PolynomialFeatures, OneHotEncoder,
│           │                          train_test_split
│           └── validation.py       ← check_array, check_X_y
└── tests/
    ├── conftest.py                  ← (generated by pytest from root conftest.py)
    ├── test_core_base.py
    ├── test_metrics_classification.py
    ├── test_metrics_regression.py
    ├── test_models_clustering.py
    ├── test_models_ensemble.py
    ├── test_models_linear.py
    ├── test_models_naive_bayes.py
    ├── test_models_svm.py
    ├── test_models_tree.py
    ├── test_utils_model_selection.py
    ├── test_utils_preprocessing.py
    └── test_utils_validation.py
```

## API Reference

### Parameter names

Some parameter names differ from scikit-learn. The table below lists the ones
most likely to cause confusion:

| Class | simpleml parameter | scikit-learn equivalent |
|---|---|---|
| `LinearRegression` | `n_iter` | `max_iter` |
| `LogisticRegression` | `n_iter` | `max_iter` |
| `train_test_split` | `random_seed` | `random_state` |

### LinearSVC output labels

`LinearSVC.predict()` returns **the original class labels** you trained on
(e.g. `{0, 1}`). Internally the SVM uses `{-1, +1}`, but this is mapped back
before returning. You do **not** need to manually remap predictions.

### DBSCAN predict behaviour

`DBSCAN.predict()` assigns new points to the nearest core point's cluster if
that core point is within `eps` distance. Points with no core point nearby
are labelled `-1` (noise). Core points are identified and stored during `fit`.

### R² on constant targets

`r2_score` returns `0.0` when all true values are identical (the denominator
`SS_tot` is zero). This is by convention — do not expect `1.0` for a perfect
predictor on constant data.

### DecisionTreeRegressor leaf predictions

With `min_samples_split=2` (the default), the tree stops splitting when a
node would produce children with fewer than 2 samples. Leaf nodes predict the
**mean** of all samples they contain, not individual sample values.

### MultinomialNaiveBayes separability

`MultinomialNaiveBayes` classifies based on **relative feature frequency
proportions** within each sample. For good accuracy, classes should differ in
*which* features are active, not just in overall magnitude. Ranges like
`[0–2]` vs `[20–22]` may be insufficient; use ranges like `[high, ~0]` vs
`[~0, high]` across different feature subsets.

## Testing

Ensure the package is installed or `src/` is on the path first (see
Installation above), then run:

```bash
# All tests, verbose
pytest tests/ -v

# With coverage report
pytest tests/ --cov=simpleml

# Single file
pytest tests/test_models_linear.py -v

# Single test
pytest tests/test_models_linear.py::TestLinearRegressionFit::test_fit_returns_self -v
```

The test suite contains **339+ tests** across 12 files covering:

- Core base classes and mixins
- All model `fit` / `predict` / `score` flows
- Edge cases: unfitted models, shape mismatches, single-class folds
- Numerical correctness: convergence, regularisation, impurity metrics
- Preprocessing: scaling, encoding, polynomial expansion, splitting
- Model selection: KFold splits, cross-validation, grid search

## Limitations

- No support for neural networks or deep learning (the `nn/` module is reserved for future use)
- No GPU acceleration
- Limited to basic algorithms — no gradient boosting, stacking, or other advanced ensemble techniques
- Not optimised for large-scale datasets
- `LinearSVC` supports **binary classification only** — raises `ValueError` for more than two classes

## Performance

This library is for educational purposes. For production use and
performance-critical applications, use [scikit-learn](https://scikit-learn.org/).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality in the `tests/` directory
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request

## License

MIT License

## Author

Serge Auronss Gbaguidi

## References

- [Scikit-learn documentation](https://scikit-learn.org)
- [NumPy documentation](https://numpy.org)
- [NumPy 2.0 migration guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)

---

**Note:** This is an educational project created to understand machine learning
algorithms and best practices for library design. For production machine
learning work, please use [scikit-learn](https://scikit-learn.org/).
