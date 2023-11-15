import numpy as np
import nashpy as nash
from functools import lru_cache
import time

def solveLP(A):
    global turn
    game = nash.Game(A)
    arr = game.linear_program()
    payoff = np.dot(np.dot(arr[0], A), arr[1])
    turn = arr
    return payoff

# for k = -1 general f(v, y, p)
@lru_cache(maxsize=None)
def f(v, y, p, k):

    if k == -1:
        sum = 0
        n = len(p)
        for i in range(n):
            sum += f(v, y, p, i)
        return sum / n
    
    if len(p) == 1:
        return np.sign(v[0] - y[0]) * p[0]

    m = len(v)
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            v1, y1, p1 = list(v), list(y), list(p)
            del v1[i]
            del y1[j]
            del p1[k]
            A[i][j] = p[k]*np.sign(v[i] - y[j]) \
                + f(tuple(v1), tuple(y1), tuple(p1), -1)
    return solveLP(A)

turn = None

def main():
    start = time.time()
    print(f((1, 2, 3, 4, 5), (1, 2, 3, 4, 5), (1, 2, 3, 4, 5), 0))
    print(turn)
    end = time.time()
    print(end - start)


if __name__ == "__main__":
    main()