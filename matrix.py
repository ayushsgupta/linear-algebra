from operator import add, sub
from copy import deepcopy
from vector import Vector

kronecker_delta = lambda i, j: 1 if i == j else 0

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

    def copy(self):
        return self.__class__(self.matrix)

    def intify(self):
        c = self.matrix
        for i in range(len(self._matrix)):
            for j in range(len(self._matrix[0])):
                k = c[i][j]
                c[i][j] = int(k) if k % 1 == 0 else k
        return self.__class__(c)

    def Zero(n, m):
        return self.__class__([[0 for _ in range(m)] for _ in range(n)])

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
        return self.__class__(ret)

    def __add__(self, other):
        return self._operate_two(other, add)

    def __sub__(self, other):
        return self._operate_two(other, sub)

    def scalarmul(self, scalar):
        return self.__class__([[scalar * elem for elem in row] for row in deepcopy(self.matrix)])

    def scalardiv(self, scalar):
        return self.__class__([[elem / scalar for elem in row] for row in deepcopy(self.matrix)])

    def __mul__(self, other):
        assert self.columns == other.rows, 'Invalid matrix multiplication'
        ret = [[0 for _ in range(other.columns)] for _ in range(self.rows)]
        for i in range(len(ret)):
            for k in range(len(ret[0])):
                ret[i][k] = sum([self.get_0(i, j) * other.get_0(j, k) for j in range(self.columns)])
        return self.__class__(ret)

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
                c[i][j] = self.__class__(e).det
        return self.__class__(c)

    def cofactor(self):
        c = self.minor().matrix
        for i in range(len(c)):
            for j in range(len(c[0])):
                c[i][j] *= pow(-1, i + j)
        return self.__class__(c)            

    def transpose(self):
        ret = [[0 for _ in range(self.rows)] for _ in range(self.columns)]
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i][j] = self.get_0(j, i)
        return self.__class__(ret)

    def Compose(matrix_list, matrix_type=Matrix):
        assert isinstance(matrix_list, list) and len(matrix_list) >= 2 and \
               all([isinstance(elem, Matrix) for elem in matrix_list]), 'Invalid list of matrices'
        composed = matrix_list.pop(0) * matrix_list.pop(0)
        while matrix_list:
            composed *= matrix_list.pop(0)
        return matrix_type(composed.matrix)

    def apply_transformation(self, vector):
        assert isinstance(vector, Vector), 'Must transform a Vector'
        assert self.columns == vector.dim, 'Invalid matrix multiplication with vector'
        t = Vector([0 for _ in range(self.rows)])
        for i in range(t.dim):
            val = sum([self.get_0(i, j) * vector.get_0(j) for j in range(vector.dim)])
            t.setval_0(val, i)
        return t

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

    @property
    def symmetric(self):
        return self == self.transpose()

    @property
    def antisymmetric(self):
        return self == self.transpose().scalarmul(-1)

    def symmetric_decomposition(self):
        m1 = self.copy()
        m2 = m1.transpose()
        return (m1 + m2).scalardiv(2).intify(), (m1 - m2).scalardiv(2).intify()

class AugmentedMatrix(Matrix):
    
    def __init__(self, matrix):
        super().__init__(matrix)
        assert self.columns == 1 + self.rows, 'Not an augmented matrix'

    def ref(self):
        m = self.matrix
        for row_index in range(len(m)):
            factor = m[row_index][row_index]
            m[row_index] = [k / factor for k in m[row_index]]
            for second_index in range(row_index, len(m)):
                if row_index != second_index:
                    factor2 = m[second_index][row_index] / m[row_index][row_index]
                    m[second_index] = [m[second_index][k] - factor2 * m[row_index][k] for k in range(len(m[row_index]))]
        return self.__class__(m).intify()

    def rref(self):
        m = self.matrix
        for row_index in range(len(m)):
            factor = m[row_index][row_index]
            m[row_index] = [k / factor for k in m[row_index]]
            for second_index in range(len(m)):
                if row_index != second_index:
                    factor2 = m[second_index][row_index] / m[row_index][row_index]
                    m[second_index] = [m[second_index][k] - factor2 * m[row_index][k] for k in range(len(m[row_index]))]
        return self.__class__(m).intify()



    