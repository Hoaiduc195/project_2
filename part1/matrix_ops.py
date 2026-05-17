import math

def transpose(A):
    if not A: return []
    if not isinstance(A[0], list):
        return [[x] for x in A]
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def matmul(A, B):
    A_2d = A if isinstance(A[0], list) else [A]
    B_2d = B if isinstance(B[0], list) else [[x] for x in B]
    
    rows_A, cols_A = len(A_2d), len(A_2d[0])
    rows_B, cols_B = len(B_2d), len(B_2d[0])
    
    if cols_A != rows_B:
        raise ValueError(f"Incompatible dimensions for multiplication: {cols_A} and {rows_B}")
        
    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            C[i][j] = sum(A_2d[i][k] * B_2d[k][j] for k in range(cols_A))
    return C

def invert(A):
    # Gauss-Jordan elimination
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    
    for i in range(n):
        pivot_row = i
        for j in range(i+1, n):
            if abs(M[j][i]) > abs(M[pivot_row][i]):
                pivot_row = j
                
        M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
            
        for j in range(i, 2*n):
            M[i][j] /= pivot
            
        for k in range(n):
            if k == i: continue
            factor = M[k][i]
            for j in range(i, 2*n):
                M[k][j] -= factor * M[i][j]
                
    return [row[n:] for row in M]

def mat_add(A, B):
    if not isinstance(A[0], list):
        return [A[i] + B[i] for i in range(len(A))]
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_sub(A, B):
    if not isinstance(A[0], list):
        return [A[i] - B[i] for i in range(len(A))]
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def scalar_mul(scalar, A):
    if not isinstance(A[0], list):
        return [scalar * x for x in A]
    return [[scalar * x for x in row] for row in A]

def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def add_intercept(X):
    new_X = []
    for row in X:
        if isinstance(row, list):
            if len(row) > 0 and row[0] == 1.0:
                new_X.append(row[:])
            else:
                new_X.append([1.0] + row)
        else:
            if row == 1.0:
                new_X.append([row])
            else:
                new_X.append([1.0, row])
    return new_X
