import math
from distutils.ccompiler import gen_lib_options

import numpy as np

L = 8
n = (L**2)//2
board = np.arange(n, step= 1)

rect_board = np.reshape(board, (L//2,L))

def generate_plus_x_trans_matrix():
    H = np.zeros(( n,  n), dtype=int)
    for b in  range(n):
        #H[new_bit][old_bit]
        print(b)
        new_bit = ((b + 1) %L)  + ((b//L) * (L)) 
        H[new_bit][b] = 1
    return H 

Hx = generate_plus_x_trans_matrix()
print("testing")
for i in range(4):
    print( np.reshape(board, (L//2,L)))
    board = np.matmul(Hx, board)
    print()
    print()
    
print( np.reshape(board, (L//2,L)))

print()
print()
print(Hx)
