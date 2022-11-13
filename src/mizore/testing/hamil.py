from mizore.operators import QubitOperator
import os

this_folder_path = os.path.dirname(os.path.abspath(__file__))


def get_test_hamil(category, name):
    """
    Please see the files in mizore/testing/hamil/
    Args:
        category: "mol"
    """
    folder_path = this_folder_path + f"/hamil/{category}/"
    op = QubitOperator.read_op_file(name, folder_path)
    for pword, coeff in op:
        op.terms[pword] = float(coeff.real)
        assert abs(coeff.imag) < 1e-7
    return op


if __name__ == '__main__':
    op = get_test_hamil("mol", "C2H2_24_BK")
    print(len(op.terms))
