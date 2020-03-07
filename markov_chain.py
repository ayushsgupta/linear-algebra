from copy import deepcopy
from matrix import Matrix, SquareMatrix
from vector import Vector

class MarkovChain:

    def __init__(self, transition, prob):
        assert isinstance(transition, SquareMatrix)
        assert isinstance(prob, Vector)
        assert transition.dim == prob.dim
        self._transition_matrix = deepcopy(transition)
        self._probability_vector = deepcopy(prob)
        self._original_probability = deepcopy(prob)

    @property
    def transition_matrix(self):
        return deepcopy(self._transition_matrix)

    @property
    def probability_vector(self):
        return deepcopy(self._probability_vector)

    @property
    def initial_probabilities(self):
        return deepcopy(self._original_probability)

    def cycle(self, n=1):
        assert n > 0
        self._probability_vector = self._probability_vector.apply_transformation(self._transition_power(n))

    def _transition_power(self, n):
        if n == 1:
            return self.transition_matrix
        return SquareMatrix(Matrix.Compose([self.transition_matrix for _ in range(n)]).matrix)

    def state_at(self, which):
        assert which > 0 and which <= self.probability_vector.dim
        return self.probability_vector.get(which)

    def restore(self):
        self._probability_vector = self.initial_probabilities