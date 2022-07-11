from jax import numpy as jnp
from scipy.linalg import eigh


def generalized_eigv_by_wang(A, B, eigvals_only=True, eps=1e-10):
    """
    Solve the generalized eigenvalue problem. Ax=λBx, where λ is the eigenvalues

    The implementation follows https://arxiv.org/pdf/1903.11240.pdf Page 7
    Or http://fourier.eng.hmc.edu/e161/lectures/algebra/node7.html (This link is not available)
    The author of this algorithm is Wang Ruye.

    :return: eigvals, eigvec
    """
    # Line 1
    lambda_B, phi_B = jnp.linalg.eigh(B)
    # Line 2
    lambda_inv_sqrt_B = 1.0 / jnp.sqrt(lambda_B + eps)
    phi_B_cup = phi_B @ jnp.diag(lambda_inv_sqrt_B)
    # Line 3
    A_cup = jnp.conjugate(jnp.transpose(phi_B_cup)) @ A @ phi_B_cup
    # Line 4
    eigvals, phi_A = jnp.linalg.eigh(A_cup)
    # Line 5
    if eigvals_only:
        return eigvals
    # Line 6
    eigvec = phi_B_cup @ phi_A
    return eigvals, eigvec


if __name__ == '__main__':
    A = jnp.array([[1.0, 0.1], [0.1, 1.0]])
    B = jnp.array([[1.0, 0.1j], [-0.1j, 1.0]])
    print(generalized_eigv_by_wang(A, B))
    print(eigh(A, B, eigvals_only=True))
