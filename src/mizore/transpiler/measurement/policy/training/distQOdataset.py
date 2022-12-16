from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from mizore.operators import QubitOperator
from mizore.testing.hamil import get_test_hamil



class DistQubitOperator(Dataset):

    def __init__(self, mol_name):
        self.mol_name = mol_name
        self.pauli_map = {"X": 0, "Y": 1, "Z": 2}
        self.hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
        self.n_qubits = self.hamil.n_qubit
        self.pwords, self.coeff = self.get_operator_tensor(self.hamil, self.n_qubits)
        

    def get_operator_tensor(self, op: QubitOperator, n_qubit):
        coeffs = []
        pwords = []
        for pword, coeff in op.terms.items():
            pwords.append(self.get_pword_tensor(pword, n_qubit))
            coeffs.append(coeff)
        return torch.stack(pwords), torch.tensor(coeffs)

    def get_pword_tensor(self, pword, n_qubit):
        pauli_tensor = [[0.0, 0.0, 0.0] for _ in range(n_qubit)]
        for i_qubit, pauli in pword:
            pauli_tensor[i_qubit][self.pauli_map[pauli]] = 1.0
        return torch.tensor(pauli_tensor)
    
    def __len__(self):
        return self.coeff.size()[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx[0]
        
        return {'pword': self.pwords[idx], 'coeff': self.coeff[idx]}
        

