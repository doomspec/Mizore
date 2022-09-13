from chemistry.simple_mols import simple_4_qubit_lih
from mizore.transpiler.measurement.grouping.qwc import get_qwc_cliques_by_LDF, get_shot_num_map_from_cliques

if __name__ == '__main__':
    hamil, _ = simple_4_qubit_lih().remove_constant()
    cliques = get_qwc_cliques_by_LDF(hamil)
    clique_pauliwords, clique_shot_ratio = get_shot_num_map_from_cliques(cliques, hamil)
    for i in range(len(clique_pauliwords)):
        print(clique_shot_ratio[i], clique_pauliwords[i])