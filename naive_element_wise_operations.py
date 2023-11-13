import numpy as np


def naive_relu(x:np.array):
    assert len(x.shape) == 2

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] = max(x[i][j], 0)
    return x


def naive_add(x:np.array, y:np.array):
    assert len(x.shape) == 2
    assert x.shape == y.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] += y[i][j]
    return x


def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] += y[j]

    return x


def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1

    z = 0
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i][j] * y[j]
    return z


def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i][j] = naive_vector_dot(row_x, column_y)
    
    return z

test_relu = np.array([[-1, 2, -3],[4, -5, 6]])
print("naive relu:\n", naive_relu(test_relu), "\n")

test_x = np.array([[1, 2, 3],[4, 5, 6]])
test_y = np.array([[3, 2, 1],[6, 5, 4]])
# numpy array add: test_x + test_y
print("naive add:\n", naive_add(test_x, test_y), "\n")

test_matrix = np.random.randint(0, 10, (5, 8))
test_vector = np.array([3 for _ in range(8)])
# add a 8D vector to a n*8 matrix by broadcasting the vector first
print("matrix:\n", test_matrix, "\n")
print("vector:\n", test_vector, "\n")
print("matrix + vector:\n", naive_add_matrix_and_vector(test_matrix, test_vector), "\n")

test_vector_x = np.array([1,2,3,4,5])
test_vector_y = np.array([5,4,3,2,1])
print("vector x:\n", test_vector_x, "\n")
print("vector y:\n", test_vector_y, "\n")
print("vector product of x and y:\n", naive_vector_dot(test_vector_x, test_vector_y), "\n")

test_matrix_x = np.random.randint(0, 10, (5, 8))
test_vector_y = np.array([3 for _ in range(8)])
print("matrix x:\n", test_matrix_x, "\n")
print("vector y:\n", test_vector_y, "\n")
print("matrix vector product of x and y:\n", naive_matrix_vector_dot(test_matrix_x, test_vector_y), "\n")

test_matrix_x = np.random.randint(0, 9, (4, 8))
test_matrix_y = np.random.randint(0, 9, (8, 4))
print("matrix x:\n", test_matrix_x, "\n")
print("matrix y:\n", test_matrix_y, "\n")
print("matrix dot product of x and y:\n", naive_matrix_dot(test_matrix_x, test_matrix_y), "\n")