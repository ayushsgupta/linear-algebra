from matrix import Matrix, SquareMatrix, AugmentedMatrix
from vector import Vector
from linear_system import LinearSystem

a = Matrix([[1, 1],
            [1, 1]])
b = Matrix([[1, 0],
            [0, 1]])
c = Matrix([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
d = Matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])
e = Matrix([[7, 2, 9],
            [8, 1, 4],
            [4, 0, -2]])
f = Matrix([[1, 2, 3, 4]])
g = Matrix([[1],
            [2],
            [3],
            [4]])
h = Matrix([[1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [4, 5, 6, 7]])


b = Vector('1, 2, 3')
C = Matrix('2, 1, 3; 5, 2, 4; 1, 0, 3')
s = LinearSystem(C, b)
s.augment()
A = s.augmented_matrix