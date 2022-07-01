

import pickle, os
from mizore.operators import QubitOperator
folder_path = os.path.dirname(os.path.abspath(__file__))



def simple_4_qubit_lih():
    with open(folder_path+"/hamil_cache/lih.pkl", "rb") as f:
        terms = pickle.load(f)
    hamil = QubitOperator()
    hamil.terms = terms
    #hamil = hamil * QubitOperator("X0")
    return hamil - (-7.979466)