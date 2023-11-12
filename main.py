import numpy as np
import nashpy as nash

def solveLP(A):
    global turn
    game = nash.Game(A)
    arr = game.linear_program()
    (x, _) = A.shape
    sum = 0
    for i in range(x):
        for j in range(x):
            sum += arr[0][i]*arr[1][j]*A[i][j]
    turn = arr
    return sum

def sign(x):
    if x > 0:
        return 1
    if x == 0:
        return 0
    return -1

#for k = -1 general f(v, y, p)
def f(v, y, p, k):
    global prev

    if (tuple(v), tuple(y), tuple(p), k) in prev:
        return prev[(tuple(v), tuple(y), tuple(p), k)]

    if len(p) == 1:
        if v[0] > y[0]:
            return p[0]
        if v[0] == y[0]:
            return 0
        return -p[0]

    if k == -1:
        sum = 0
        for i in range(len(p)):
            sum += f(v, y, p, i)
        return sum / len(p)
    
    if len(p) == 2:
        if k == 0:
            a = sign(v[0] - y[0]) * p[0] + sign(v[1] - y[1]) * p[1]
            b = sign(v[1] - y[0]) * p[0] + sign(v[0] - y[1]) * p[1]
            c = sign(v[0] - y[1]) * p[0] + sign(v[1] - y[0]) * p[1]
            d = sign(v[1] - y[1]) * p[0] + sign(v[0] - y[0]) * p[1]
            A = np.array([[a, b], [c, d]])
            prev[(tuple(v), tuple(y), tuple(p), k)] = solveLP(A)
            return solveLP(A)
        a = sign(v[0] - y[0]) * p[1] + sign(v[1] - y[1]) * p[0]
        b = sign(v[1] - y[0]) * p[1] + sign(v[0] - y[1]) * p[0]
        c = sign(v[0] - y[1]) * p[1] + sign(v[1] - y[0]) * p[0]
        d = sign(v[1] - y[1]) * p[1] + sign(v[0] - y[0]) * p[0]
        A = np.array([[a, b], [c, d]])
        prev[(tuple(v), tuple(y), tuple(p), k)] = solveLP(A)
        return solveLP(A)

    A = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            v1 = v.copy()
            y1 = y.copy()
            p1 = p.copy()
            del v1[i]
            del y1[j]
            del p1[k]
            A[i][j] = p[k]*sign(v[i] - y[j]) + f(v1, y1, p1, -1)
    prev[(tuple(v), tuple(y), tuple(p), k)] = solveLP(A)
    return solveLP(A)

turn = 0

prev = {}

def main():
    print(f([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 3))
    print(turn)

if __name__ == "__main__":
    main()