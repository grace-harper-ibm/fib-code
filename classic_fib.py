# grug brain 
import math
import random

import numpy as np


class fib_code():
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
    def __init__(self, L=8,p=0.001, decip=1000):
        assert math.log2(L) % 1 == 0, "L must be some 2**n where n is an int >= 1"
        self.L = L # len
        self.no_bits = (L**2)//2 # no bits
        self.p = p #probability of error on a given bit
        self.decip = decip
        # self.board = np.packbits(np.zeros((L**2)//2), axis=1)
        self.board =np.zeros((self.L**2)//2, dtype=int)
        self.fund_symm = self._generate_fundamental_symmetry()

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
        H = np.zeros((self.n,  self.n), dtype=int)
        for b in  range(self.n):
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

    def _generate_fundamental_symmetry(self):
        midpoint = self.L//2
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
        
    
    