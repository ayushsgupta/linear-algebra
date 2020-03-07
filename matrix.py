from operator import add, sub
from copy import deepcopy

kronecker_delta = lambda i, j: 1 if i == j else 0
# levi_civita = lambda i, j, k: 1 if [i, j, k] in [[1, 2, 3], [2, 3, 1], [3, 1, 2]] else -1 if [i, j, k] in [[1, 3, 2], [2, 1, 3], [3, 2, 1]] else 0

class Matrix:

    def __init__(self, matrix):
        if isinstance(matrix, str):
            matrix = [[int(k) if float(k) % 1 == 0 else float(k) for k in row.split(',')] for row in matrix.split(';')]
        lens = [len(row) for row in matrix]
        assert all(elem == lens[0] for elem in lens), 'Matrix must be rectangular'
        assert all([[type(elem) in [int, float] for elem in row] for row in matrix]), 'All elements must be of type float or int'
        self._matrix = deepcopy(matrix)
        self._rows = len(self.matrix)
        self._columns = len(self.matrix[0])
        self._square = self.columns == self.rows

    def __repr__(self):
        return 'Matrix({0})'.format(str(self.matrix))

    def __str__(self):
        return '\n'.join(['\t'.join([str(cell) for cell in row]) for row in self.matrix])
        # return *(' '.join(row) for row in self.matrix), sep='\n'
        # return('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in self.matrix]))

    @property
    def matrix(self):
        return deepcopy(self._matrix)
    
    @property
    def rows(self):
        return self._rows

    @property
    def columns(self):
        return self._columns

    @property
    def n(self):
        return self.rows
    
    @property
    def m(self):
        return self.columns

    @property
    def dimensions(self):
        return self.rows, self.columns

    @property
    def square(self):
        return self._square

    def Zero(n, m):
        return Matrix([[0 for _ in range(m)] for _ in range(n)])

    def get(self, i1, i2):
        assert i1 <= self.rows and i2 <= self.columns, 'Invalid indices'
        return self._matrix[i1 - 1][i2 - 1]

    def get_0(self, i1, i2):
        assert i1 < self.rows and i2 < self.columns, 'Invalid indices'
        return self._matrix[i1][i2]

    def setval(self, val, i1, i2):
        assert i1 <= self.rows and i2 <= self.columns, 'Invalid indices'
        self._matrix[i1 - 1][i2 - 1] = val

    def setval_0(self, val, i1, i2):
        assert i1 < self.rows and i2 < self.columns, 'Invalid indices'
        self._matrix[i1][i2] = val

    def _operate_two(self, other, operation):
        assert self.dimensions == other.dimensions, 'Matrices must have same dimensions'
        a, b = self.matrix, other.matrix 
        ret = deepcopy(a)
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i][j] = operation(a[i][j], b[i][j])
        return Matrix(ret)

    def __add__(self, other):
        return self._operate_two(other, add)

    def __sub__(self, other):
        return self._operate_two(other, sub)

    def scalarmul(self, scalar):
        return Matrix([[scalar * elem for elem in row] for row in deepcopy(self.matrix)])

    def scalardiv(self, scalar):
        return Matrix([[elem / scalar for elem in row] for row in deepcopy(self.matrix)])

    def __mul__(self, other):
        assert self.columns == other.rows, 'Invalid matrix multiplication'
        ret = [[0 for _ in range(other.columns)] for _ in range(self.rows)]
        for i in range(len(ret)):
            for k in range(len(ret[0])):
                ret[i][k] = sum([self.get_0(i, j) * other.get_0(j, k) for j in range(self.columns)])
        return Matrix(ret)

    def __eq__(self, other):
        return self.matrix == other.matrix

    @property
    def trace(self):
        assert self.square, 'Trace undefined for a non-square matrix'
        return sum([sum([kronecker_delta(i, j) * self.get_0(i, j) for j in range(self.columns)]) for i in range(self.rows)])

    @property
    def det(self):
        assert self.square, 'Cannot take determinant of a non-square matrix'
        if self.rows == 1:
            return self.get(1, 1)
        elif self.rows == 2:
            return self.get(1, 1) * self.get(2, 2) - self.get(1, 2) * self.get(2, 1)
        else:
            cofactor = self.cofactor()
            return sum([self.get_0(0, j) * cofactor.get_0(0, j) for j in range(self.columns)])

    def minor(self):
        c = deepcopy(self.matrix)
        d = deepcopy(c)
        for i in range(len(c)):
            for j in range(len(c[0])):
                e = deepcopy(d)
                e.pop(i)
                for k in range(len(e)):
                    e[k].pop(j)
                c[i][j] = Matrix(e).det
        return Matrix(c)

    def cofactor(self):
        c = self.minor().matrix
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] *= pow(-1, i + j)
        return Matrix(c)            

    def transpose(self):
        ret = [[0 for _ in range(self.rows)] for _ in range(self.columns)]
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i][j] = self.get_0(j, i)
        return Matrix(ret)


class SquareMatrix(Matrix):

    def __init__(self, matrix):
        super().__init__(matrix)
        assert self.square, 'Not a square matrix'

    def Identity(dim):
        return SquareMatrix([[kronecker_delta(i, j) for j in range(dim)] for i in range(dim)])

    def Zero(dim):
        return SquareMatrix([[0 for _ in range(dim)] for _ in range(dim)])

    @property 
    def dim(self):
        return self.columns