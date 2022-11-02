# grug brain 
import math
import random
from turtle import st

import numpy as np


class FibCode():
    """
    self.board = 
    [L*(L//2 - 1)....          ((L**2)//2) - 1]
     .
     .
     .   
     2L
     L 
     0 1 2 3 ....                                L - 1]
    """
    def __init__(self, L=8,p=0.001, decip=1000, start_arr = None):
        assert math.log2(L) % 1 == 0, "L must be some 2**n where n is an int >= 1"
        self.L = L # len
        self.no_bits = (L**2)//2 # no bits
        self.p = p #probability of error on a given bit
        self.decip = decip
        # self.board = np.packbits(np.zeros((L**2)//2), axis=1)
        self.board =np.zeros((self.L**2)//2, dtype=int)
        self.fund_symm = self._generate_fundamental_symmetry(start_arr)
        print(self.fund_symm)

    def set_code_word(bottom_row_start_sequence):
        pass 
    
    def generate_errors(self):
        cutoff = self.p*self.decip
        for i in len(self.board):
            if random.randrange(0, self.decip) <= cutoff:
                self.board[i] ^= 1
                
    
    def bit_to_rc(self, bit):
        row_len = self.L 

        rindx = bit // row_len
        cindx = bit%row_len 
        return (rindx, cindx)
        
    # def rc_to_bit(self, rc):
        
    def _generate_plus_x_trans_matrix(self):
        H = np.zeros((self.no_bits,  self.no_bits), dtype=int)
        for b in  range(self.no_bits):
            #H[new_bit][old_bit]
            print(b)
            new_bit = ((b + 1) %self.L)  + ((b//self.L) * (self.L)) 
            H[new_bit][b] = 1
        return H 

    def _generate_plus_y_trans_matrix(self):
        " performs a -1 y (or a + 1 y if you're indexing rows top to bottom like numpy :...( "
        H = np.zeros(( self.no_bits,  self.no_bits), dtype=int)
        for b in  range(self.no_bits):
            #H[new_bit][old_bit]
            new_bit = (b + self.L) %  self.no_bits
            H[new_bit][b] = 1
        return H 

    def _generate_fundamental_symmetry(self, start_arr = None):
        # fundamental symmetries start from the top instead of the bottom because numpy
        rect_board = np.reshape(self.board, (self.L//2, self.L))
        if not start_arr: 
            midpoint = self.L//2
            start_arr = np.zeros(self.L, dtype=int)
            start_arr[midpoint] = 1
        rect_board[0] = start_arr
        for row in range(1, self.L//2):
            for bit in range(self.L):
                new_val =  rect_board[row - 1][(bit - 1)%self.L] ^  rect_board[row - 1][(bit )%self.L] ^  rect_board[row - 1][(bit + 1)%self.L]
                rect_board[row][bit] = new_val
        return rect_board
        # build fundamental symmetry, into array (ones on which ones)
        
        
        # which stabs go on the fundamental symmetry? 
        
    def _generate_parity_check_matrix(self):
        # (L**2)//2 long and # stabs down (on fundamental symmetry)
        pass 
        
    

def decode_fib_code(fibcode):
    pass 
    
    
    


    
    
if __name__ == "__main__":
    fb = fib_code()
    print(fb.board) 
    print(type(fb.board))
    print(type(fb.board[0]))
        
    
    