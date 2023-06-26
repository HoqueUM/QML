# Link to article: https://qiskit.org/ecosystem/machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html
# Link to VQC: https://qiskit.org/ecosystem/machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html

# Loading sklearn dependencies
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from qiskit.utils import algorithm_globals

# Feature Map 
from qiskit.circuit.library import ZZFeatureMap

# Ansatz 
from qiskit.circuit.library import EfficientSU2

# Optimizers
from qiskit.algorithms.optimizers import COBYLA

# Sampler and VQC
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC

# For plotting callback graph
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Loading the data and setting x and y
data = load_iris()
x = data.data
y = data.target

# Rescaling x (required for VQC)
x = MinMaxScaler().fit_transform(x)

# Splitting x and y into training and tetsing splits
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.25, random_state=42)

# Getting the shape of x for feature map dimension
x_num = x.shape[1]

# Setting Parameters for VQC

# Feature map dimensions must be the amount of features in the dataset, reps is the amount of times it repeats the process
x_map = ZZFeatureMap(feature_dimension=x_num, reps=1)

# Ansatz are just premade circuits for calculations, the number of qubits must be equal to number of features, reps are number of times it repeats
ansatz = EfficientSU2(num_qubits=x_num, reps=3)

# Optimizer sets the number of iterations the VQC executes, anything more than 100 is kind of useless as it doesn't improve much past 100
optimizer = COBYLA(maxiter=100)

# Sets the random seed for sampling from the dataset
sampler = Sampler()
sampler.set_options(seed=42)

# Creating a callback graph to visualize
objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)


def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)

# Running VQC and fitting it to data
vqc = VQC(sampler=sampler, feature_map=x_map, ansatz=ansatz, optimizer=optimizer, callback=callback_graph)

objective_func_vals = []

vqc.fit(x_train, y_train)

plt.show()

# Finding the score of the quantum model
predictions1 = vqc.predict(x_test)
accuracy = accuracy_score(predictions1, y_test)
print(accuracy)