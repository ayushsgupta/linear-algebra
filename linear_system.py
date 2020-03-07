from matrix import Matrix, SquareMatrix, AugmentedMatrix
from vector import Vector


class LinearSystem:

    def __init__(self, coefficients, knowns):
        assert isinstance(coefficients, Matrix), 'Coefficients must be a Matrix'
        self._coefficients = coefficients
        assert isinstance(knowns, Vector), 'Knowns must be a Vector'
        assert coefficients.rows == knowns.dim, 'Improper linear system'
        self._knowns = knowns
        self._augmented, self._is_augmented = self._coefficients, False

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def knowns(self):
        return self._knowns

    @property
    def augmented_matrix(self):
        assert self._is_augmented, 'Matrix of coefficients not yet augmented'
        return self._augmented

    def augment(self):
        if not self._is_augmented:
            a = self._augmented.matrix
            for i in range(len(a)):
                a[i].append(self._knowns.vector[i])
            self._is_augmented = True
            self._augmented = AugmentedMatrix(a)
