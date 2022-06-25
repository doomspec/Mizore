# Mizore: a universal quantum program analyzer

Mizore aims to provide a universal framework for analyzing quantum program
with **implementation details to be considered**. 
Mizore is design to show the user how the quantum program will actually work
on real quantum devices. Especially, Mizore wishes to provide convenient tool for analyzing the 
**execution time** of quantum programs on quantum devices. To achieve this goal, Mizore combined 
several ideas, which are stated below.

- Variance analysis
- Program transpiler
- Execution time estimator

# Variance analysis

In quantum programs, parameters such as expectation value measured from quantum states are processed,
passed and even used for constructing new quantum circuits. However, because of the probabilistic 
nature of quantum mechanics, these parameters are also naturally random variables whose variance 
decreases with the times of repetition (shot number) that one measure the circuits. 
Estimating the shot number needed in each quantum circuit in a quantum program is the key to estimate
the required execution time of it.

Mizore provides built-in functionality for 
differentiable variance analysis of the parameters in quantum programs so that the shot numbers needed 
to achieve required accuracy for each quantum circuit can be optimized.

# Noise-cancelling transpilers

A program transpiler is a map that maps the original program to a new one. 
For example, as near-term quantum devices are noisy, an implicit transpiler which adds noise to 
quantum circuits is applied to the quantum program. To compensate the noise transpiler, additional
noise-cancelling transpiler must be applied before executing the program. These transpilers usually 
scale the number of required qubits or the shot number of the quantum circuit. 
Mizore provides ready-to-use common noise-cancelling transpilers to help the user make their cost
into the consideration for program analysis. 
In Mizore, the variance of the parameters in the new program is automatically handled.

The implemented transpiler includes:

Error mitigation:
- Error extrapolation
- Passive error correction (TODO)
- Virtual distillation (TODO)
- 

Error correction:
- (TODO)

# Measurement methods

Measuring the expectation value of observables is a core procedure of quantum computing.
Different measurement method will give different variance of the result.
There are a series of methods which we provide in Mizore.

- Naively evaluating the Pauli words.
- Clifford shadow
- Commuting groups

# Quantum computational graph (QCG)

# Execution time estimator

## Hardware configuration from random benchmarking (RB) (TODO)
Random benchmarking is a method to estimate the noise level/type of in quantum devices.
Mizore provides random benchmarking tools to test whether the noise-adding transpilers one uses matches
the actual noise model on devices.

# What Mizore is not