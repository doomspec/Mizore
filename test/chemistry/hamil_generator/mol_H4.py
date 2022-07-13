import tequila as tq
import pickle

geomstring="H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5"
molecule = tq.quantumchemistry.Molecule(geometry=geomstring, basis_set='sto-3g', transformation="bravyi-kitaev")

# get the qubit hamiltonian
H = molecule.make_hamiltonian()

with open("../hamil_cache/h4.pkl", "wb") as f:
    pickle.dump(H.to_openfermion().terms, f)

# create an k-UpCCGSD backend_circuit of order k
U = molecule.make_upccgsd_ansatz(order=1)

print(U)

# define the expectation value
E = tq.ExpectationValue(H=H, U=U)

# compute reference energies
fci = molecule.compute_energy("fci")

# optimize
print("FCI : {:+2.8}f".format(fci))
# -1.9961503