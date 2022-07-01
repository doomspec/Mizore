import tequila as tq
import pickle
# define a molecule within an active space
active = {"a1": [1], "b1":[0]}
geomstring="Li 0.0 0.0 0.0\nH 0.0 0.0 1.6"
molecule = tq.quantumchemistry.Molecule(geometry=geomstring, basis_set='6-31g', active_orbitals=active, transformation="bravyi-kitaev")

# get the qubit hamiltonian
H = molecule.make_hamiltonian()

with open("../hamil_cache/lih.pkl", "wb") as f:
    pickle.dump(H.to_openfermion().terms, f)

# create an k-UpCCGSD backend_circuit of order k
U = molecule.make_upccgsd_ansatz(order=1)

print(U)

# define the expectationvalue
E = tq.ExpectationValue(H=H, U=U)

# compute reference energies
fci = molecule.compute_energy("fci")

# optimize
#result = tq.minimize(objective=E, method="BFGS", initial_values=0.0)

#print("VQE : {:+2.8}f".format(result.energy))
print("FCI : {:+2.8}f".format(fci))