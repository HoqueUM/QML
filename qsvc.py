# Link to guide: https://qiskit.org/ecosystem/machine-learning/tutorials/03_quantum_kernel.html
# Link to QSVC: https://qiskit.org/ecosystem/machine-learning/stubs/qiskit_machine_learning.algorithms.QSVC.html

# Importing the necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from qiskit_machine_learning.algorithms import QSVC

# Loading data and setting x and y
data = load_iris()
x = data.data
y = data.target

# Splitting into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)

# Training the model
model = QSVC()
model.fit(x_train, y_train)

# Finding score of the model
predictions1 = model.predict(x_test)
score1 = accuracy_score(predictions1, y_test)

# Printing the score
print(f'Quantum Score: {score1}')





