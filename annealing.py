# Guide followed from: https://github.com/dwavesystems/dwave-scikit-learn-plugin
# Setting API key: https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html

# Importing necessary libraries
from dwave.plugins.sklearn import SelectFromQuadraticModel # API key must be defined in the config file. See above link for guide.

from sklearn.datasets import load_breast_cancer

# Loading data and setting x and y
data = load_breast_cancer()
x = data.data
y = data.target

# Dimensionality reduction using quantum annealing (from 30 features to 4)
x_new = SelectFromQuadraticModel(num_features=4).fit_transform(x, y)

# Setting features variable.
features = x_new.shape[1]

print(f'New features: {features}')
