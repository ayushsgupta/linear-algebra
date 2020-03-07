from operator import add, sub
from copy import deepcopy
from math import sqrt, acos, atan2, pi
from matrix import Matrix

levi_civita = lambda i, j, k: 1 if [i, j, k] in [[1, 2, 3], [2, 3, 1], [3, 1, 2]] else -1 if [i, j, k] in [[1, 3, 2], [2, 1, 3], [3, 2, 1]] else 0

class Vector:

    def __init__(self, vector):
        assert type(vector) in [list, str], 'Vector must be given as a list or string'
        if isinstance(vector, str):
            vector = [int(k) if float(k) % 1 == 0 else float(k) for k in vector.split(',')]
        assert all([type(elem) in [int, float] for elem in vector]), 'Vector components can only be numbers'
        self._vector = deepcopy(vector)

    def __repr__(self):
        return 'Vector({0})'.format(str(self.vector))

    @property
    def vector(self):
        return deepcopy(self._vector)

    @property
    def dim(self):
        return len(self.vector)

    @property
    def mag_squared(self):
        return sum([e * e for e in self._vector])

    @property
    def mag(self):
        k = sqrt(self.mag_squared)
        return int(k) if k % 1 == 0 else k

    def unitized(self):
        return Vector([e / self.mag for e in self._vector])

    def unitize(self):
        self = self.unitized()

    @property
    def heading(self):
        assert self.dim == 2, 'Heading only applied to vectors in R2'
        return atan2(self.y, self.x)

    @property
    def heading_degrees(self):
        return self.heading * 180 / pi

    def get(self, i):
        return self._vector[i - 1]

    def get_0(self, i):
        return self._vector[i]

    @property
    def x(self):
        assert self.dim >= 1
        return self.get(1)

    @property
    def y(self):
        assert self.dim >= 2
        return self.get(2)

    @property
    def z(self):
        assert self.dim >= 3
        return self.get(3)

    def setval(self, val, i):
        self._vector[i - 1] = val

    def setval_0(self, val, i):
        self._vector[i] = val

    def _operate_two(self, other, operation):
        assert self.dim == other.dim, 'Vectors must have the same basis'
        a, b = self.vector, other.vector 
        ret = deepcopy(a)
        for i in range(len(ret)):
            ret[i] = operation(a[i], b[i])
        return Vector(ret)

    def __add__(self, other):
        return self._operate_two(other, add)

    def __sub__(self, other):
        return self._operate_two(other, sub)

    def scalarmul(self, scalar):
        return Vector([scalar * e for e in self.vector])

    def dot(self, other):
        assert isinstance(other, Vector), 'Must take dot product with another Vector'
        assert self.dim == other.dim, 'Vectors must have the same basis'
        a, b = self.vector, other.vector
        return sum([a[i] * b[i] for i in range(self.dim)])

    def orthogonal(self, other):
        return self.dot(other) == 0

    def angle_to(self, other):
        assert self.dim in [2, 3] and other.dim == self.dim, 'Invalid angle determination'
        return acos(self.dot(other) / (self.mag * other.mag))

    def angle_to_degrees(self, other):
        return self.angle_to(other) * 180 / pi

    def cross(self, other):
        assert self.dim == other.dim == 3, 'Cross product valid for vectors in R3, R9 not yet implemented'
        prod = Vector([0, 0, 0])
        for i in range(1, 4):
            component = []
            for j in range(1, 4):
                for k in range(1, 4):
                    component.append(levi_civita(i, j, k) * self.get(j) * other.get(k))
            prod.setval(sum(component), i)
        return prod

    def apply_transformation(self, matrix):
        assert isinstance(matrix, Matrix), 'Must transform Vector using Matrix'
        assert self.dim == matrix.columns, 'Invalid matrix multiplication with vector'
        t = Vector([0 for _ in range(matrix.rows)])
        for i in range(t.dim):
            val = sum([self.get_0(j) * matrix.get_0(i, j) for j in range(self.dim)])
            t.setval_0(val, i)
        return t




