import numpy as np

def matrix_to_array(adj_m):
    assert adj_m.shape[0]==adj_m.shape[1]
    array_as_list = []
    for i in range(adj_m.shape[0]):
        for j in range(adj_m.shape[0]):
            if (i==j):
                continue
            else:
                array_as_list.append(adj_m[i,j])

    return np.array(array_as_list)

def array_to_adj_matrix(array):

    # n^2 -n -len = 0 -> (n-0.5)^2 = len+0.25
    n =  int(round(np.sqrt(array.shape[0]+0.25)+0.5))
    adj_m = np.zeros((n,n))

    i = 0
    j = 0
    for value in array:
        if (j==n):
            i += 1
            j = 0

        if i==j:
            j += 1

        adj_m[i,j] = value

        j += 1

    return adj_m
