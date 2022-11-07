from functools import partial

from mizore.operators import QubitOperator
from mizore.transpiler.measurement.policy.utils_for_tensor import get_pword_tensor, pauli_map

import jax.numpy as jnp
import jax.random as rand
from jax import vmap, jit
import numpy as np
import jax


class UniversalPolicy:
    def __init__(self, heads_tensor, probs, heads_children, hamil, n_qubit, hamil_term_tensor=None):
        self._n_qubit = n_qubit
        self._heads_tensor = jnp.array(heads_tensor)
        self._heads_tensor_filled = None
        self._probs = probs
        self._heads_children = heads_children
        self._hamil: QubitOperator = hamil
        self._hamil_term_tensor = hamil_term_tensor

    @property
    def n_qubit(self):
        return self._n_qubit

    @property
    def heads_children(self):
        return self._heads_children

    def validate_probs(self):
        if abs(sum(self._probs) - 1.0) > 1e-6:
            raise Exception(f"{sum(self._probs)} is far from 1.0!")
        heads_qubit_prob = [jnp.sum(ht, axis=1) for ht in self._heads_tensor]
        for h in heads_qubit_prob:
            for v in h:
                if v - 1.0 > 1e-6:
                    raise Exception(f"{v} is too large!")

    def generate_hamil_term_tensor(self):
        self._hamil_term_tensor = {pword: get_pword_tensor(pword, self._n_qubit) for pword in self._hamil.terms}

    def get_children_overlap_by_heads(self):
        if self._hamil_term_tensor is None:
            self.generate_hamil_term_tensor()
        overlaps = []
        for i in range(len(self._heads_tensor)):
            head_tensor = self._heads_tensor[i]
            overlap = []
            for child in self._heads_children[i]:
                o = 1.0
                for i_qubit, pauli in child:
                    o *= head_tensor[i_qubit][pauli_map[pauli]]
                overlap.append(o)
            overlaps.append(overlap)
        return overlaps

    def get_children_overlap(self):
        overlap_by_heads = self.get_children_overlap_by_heads()
        overlap_dict = {}
        for i in range(len(self._heads_tensor)):
            head_prob = self._probs[i]
            children = self._heads_children[i]
            overlap_list = overlap_by_heads[i]
            for j in range(len(children)):
                child = children[j]
                overlap = overlap_list[j]
                if child in overlap_dict:
                    overlap_dict[child] += head_prob * overlap
                else:
                    overlap_dict[child] = head_prob * overlap
        return overlap_dict

    def get_variance_by_average_cov(self):
        overlap_dict = self.get_children_overlap()
        var = 0.0
        for pword, coeff in self._hamil.terms.items():
            var += coeff ** 2 * (1 / overlap_dict[pword] - 1 / (2 ** len(pword) + 1))
        return var

    def sample_pwords(self, n_shot, seed):
        n_head = len(self._heads_tensor)
        head_indices = np.random.choice(len(self._heads_tensor), n_shot, p=self._probs)
        shot_nums = [0] * n_head
        for head in head_indices:
            shot_nums[head] += 1
        if self._heads_tensor_filled is None:
            self._heads_tensor_filled = fill_head_tensor(self._heads_tensor)
        head_samples = []
        for i_head in range(n_head):
            sampled_pwords = sample_from_head(self._heads_tensor_filled[i_head], shot_nums[i_head], seed + i_head * 11)
            head_samples.append(sampled_pwords)
        return head_samples


def fill_head_tensor(head_tensor):
    head_tensor_prob = jnp.sum(head_tensor, axis=2)
    vacant = 1.0 - head_tensor_prob
    offset = jnp.repeat(jnp.expand_dims(vacant / 3, 2), 3, axis=2)
    return head_tensor + offset


pauli_op_marks = jnp.array([3 * 5, 2 * 5, 2 * 3])


def sample_single_pauli_operator(key, prob):
    return rand.choice(key, a=pauli_op_marks, p=prob)


sample_pword = vmap(sample_single_pauli_operator, in_axes=(0, 0), out_axes=0)

sample_many_pword = jit(vmap(sample_pword, in_axes=(0, None), out_axes=0))


def sample_from_head(head_tensor_filled, n_shot, seed):
    keys = rand.split(rand.PRNGKey(seed), len(head_tensor_filled) * n_shot).reshape(
        (n_shot, len(head_tensor_filled), 2))
    pwords = sample_many_pword(keys, head_tensor_filled)
    return pwords


def get_heads_tensor_from_pwords(pwords, n_qubit):
    return [get_pword_tensor(head, n_qubit) for head in pwords]
