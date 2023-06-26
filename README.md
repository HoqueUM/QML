# QML
A Collection of Templates of Quantum Machine Learning for the University of Michigan Biomedical and Clinical Information Lab.

# Quantum Simulators

### Qiskit
[Github](https://github.com/Qiskit)
>
[Website](https://qiskit.org/)
> An open-source SDK for working with quantum computers at the level of pulses, circuits, and algorithms.
> 

Qiskit's VQC an QSVC inheret methods from scikit-learn classes, making it ideal for ease of use.

### Installation
```
pip install qiskit[machine-learning]
```

### Pennylane
[Github](https://github.com/PennyLaneAI)
>
[Website](https://pennylane.ai/)
> PennyLane is a cross-platform Python library for differentiable programming of quantum computers. Train a quantum computer the same way as a neural network.
> 

Pennylane is very customizable, making it perfect for machine learning, but requires more time to master.

### Installation
```
pip install pennylane
```

### DWave
[Github](https://github.com/dwavesystems)
>
[Website](https://www.dwavesys.com/)
> D-Wave's Ocean software and other open-source projects
> 

Requires payment for API, only get 20 minutes of processing time for free.
Only use DWave for dimensionality reduction.

### Installation
```
pip install dwave-scikit-learn-plugin
```

# Files
### Annealing
Dimensionality reduction using quantum annealing.

### QKernel
A scikit-learn SVC using a quantum kernel.

### QSVC
A simple quantum SVC.

### VQC
A simple Variation Quantum Classifier



# Limitations
Number of qubits must be equal to number of features in the dataset. On current computers this is not practical as it is very computationally expensive.
