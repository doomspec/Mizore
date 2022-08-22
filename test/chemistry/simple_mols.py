

import pickle, os

from mizore.backend_circuit.one_qubit_gates import X
from mizore.meta_circuit.block.gates import Gates
from mizore.meta_circuit.meta_circuit import MetaCircuit
from mizore.operators import QubitOperator
folder_path = os.path.dirname(os.path.abspath(__file__))


def simple_4_qubit_lih_1():
    with open(folder_path+"/hamil_cache/lih.pkl", "rb") as f:
        terms = pickle.load(f)
    hamil = QubitOperator()
    hamil.terms = terms
    return hamil

def simple_4_qubit_lih():
    with open(folder_path+"/hamil_cache/lih.pkl", "rb") as f:
        terms = pickle.load(f)
    hamil = QubitOperator()
    hamil.terms = terms
    #hamil = hamil * QubitOperator("X0")
    return hamil - (-7.979466)

def simple_8_qubit_h4():
    with open(folder_path+"/hamil_cache/h4.pkl", "rb") as f:
        terms = pickle.load(f)
    hamil = QubitOperator()
    hamil.terms = terms
    #hamil = hamil * QubitOperator("X0")
    init_block = Gates(X(0), X(2))
    init_circuit = MetaCircuit(8, blocks=[init_block])
    return hamil - (-1.9961503), init_circuit