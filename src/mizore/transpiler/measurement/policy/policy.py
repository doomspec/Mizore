from functools import partial

from mizore.operators import QubitOperator
from mizore.transpiler.measurement.policy.utils_for_tensor import get_pword_tensor, pauli_map

import jax.numpy as xnp
import jax.random as rand
from jax import vmap, jit
import jax.numpy as jnp
import jax

import numpy as np


class UniversalPolicy:
    def __init__(self, heads_tensor, probs, heads_children, hamil, n_qubit, hamil_term_tensor=None, derand=""):
        self._n_qubit = n_qubit
        self._heads_tensor = np.array(heads_tensor)
        self._heads_tensor_filled = None
        self._probs = np.array(probs)
        self._heads_children = heads_children
        self._hamil: QubitOperator = hamil
        self._hamil_term_tensor = hamil_term_tensor
        self.derand = derand

    @property
    def n_qubit(self):
        return self._n_qubit

    @property
    def heads_children(self):
        return self._heads_children

    def validate_probs(self):
        if abs(sum(self._probs) - 1.0) > 1e-6:
            raise Exception(f"{sum(self._probs)} is far from 1.0!")
        heads_qubit_prob = [np.sum(ht, axis=1) for ht in self._heads_tensor]
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

    def generate_pwords(self, n_shot, seed):
        if self.derand == "":
            return self.sample_pwords(n_shot, seed)
        elif self.derand == "head":
            return self.sample_pwords_with_allocate_heads(n_shot, seed)
        else:
            raise Exception(f"Not supported type of derand: {self.derand}")

    def sample_pwords(self, n_shot, seed):
        n_head = len(self._heads_tensor)
        head_indices = np.random.choice(len(self._heads_tensor), n_shot, p=self._probs)
        shot_nums_by_head = [0] * n_head
        for head in head_indices:
            shot_nums_by_head[head] += 1
        return self.sample_pwords_with_shots_on_heads(shot_nums_by_head, seed)

    def sample_pwords_with_allocate_heads(self, n_shot, seed):
        allocated_shots = 0
        allocation = self._probs * n_shot
        floored_allocation = np.floor(allocation)
        allocation_diff = allocation - floored_allocation
        random_offset = np.random.binomial(1, allocation_diff, len(allocation_diff))
        allocation = floored_allocation + random_offset
        allocation = allocation.astype(int)
        allocated_shots = np.sum(allocation)
        # print(n_shot, allocated_shots, allocated_shots - n_shot)
        return self.sample_pwords_with_shots_on_heads(allocation, seed)

    def sample_pwords_with_shots_on_heads(self, shot_nums_by_head, seed):
        n_head = len(self._heads_tensor)
        if self._heads_tensor_filled is None:
            self._heads_tensor_filled = fill_head_tensor(self._heads_tensor)
        head_samples = []
        for i_head in range(n_head):
            sampled_pwords = sample_from_head_by_numpy(self._heads_tensor_filled[i_head], shot_nums_by_head[i_head],
                                                       seed + i_head * 11)
            head_samples.append(sampled_pwords)
        return head_samples


def fill_head_tensor(head_tensor):
    head_tensor_prob = np.sum(head_tensor, axis=2)
    vacant = 1.0 - head_tensor_prob
    offset = np.repeat(np.expand_dims(vacant / 3, 2), 3, axis=2)
    return head_tensor + offset


pauli_op_marks = np.array([3 * 5, 2 * 5, 2 * 3])

"""
def sample_single_pauli_operator(key, prob):
    return rand.choice(key, a=pauli_op_marks, p=prob)


sample_pword = vmap(sample_single_pauli_operator, in_axes=(0, 0), out_axes=0)

sample_many_pword = jit(vmap(sample_pword, in_axes=(0, None), out_axes=0))


def sample_from_head(head_tensor_filled, n_shot, seed):
    keys = rand.split(rand.PRNGKey(seed), len(head_tensor_filled) * n_shot).reshape(
        (n_shot, len(head_tensor_filled), 2))
    pwords = sample_many_pword(keys, head_tensor_filled)
    return pwords
"""


def sample_from_head_by_numpy(head_tensor_filled, n_shot, seed):
    samples_by_qubit = []
    for i_qubit in range(len(head_tensor_filled)):
        sample = np.random.choice(pauli_op_marks, n_shot, replace=True, p=head_tensor_filled[i_qubit])
        samples_by_qubit.append(sample)
    samples_by_qubit = np.array(samples_by_qubit)
    pwords = samples_by_qubit.transpose()
    return pwords


def get_heads_tensor_from_pwords(pwords, n_qubit):
    return [get_pword_tensor(head, n_qubit) for head in pwords]
