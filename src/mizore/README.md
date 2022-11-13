# Mizore

## Setup a development environment

1. Create a conda env by `conda create -n mizore python=3.10`.
2. Enter the env by `conda activate mizore`.
3. Use `conda-develop /your/path/to/Mizore/src` to add `mizore` as a package in development.
4. Enjoy your development

## Project structure

`block_circuit`: High level quantum circuit which supports inverse, parameterization and other handy usages. We expect
the users build their algorithm in this level.

`task`: tasks for quantum computer to run. We expect the tasks to be transformed by a series of processors, such as the
circuit optimizer and error mitigation methods.

`transpiler`:  transpiler that processes the tasks. The inputs of a transpiler is a tuple with elements

- tasks
- resources

Tasks are quantum tasks (SPAM) and resources are things such as accuracy, time and gate number.

A transpiler can

- transform tasks,
- calculate resources
- or **trade tasks with resources**.

