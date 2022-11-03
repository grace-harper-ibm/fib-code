# grug brain 
import copy
import math
import random

import numpy as np
import pymatching as pm
from scipy.sparse import csc_matrix


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
        self.init_symm = self._generate_init_symmetry(start_arr)
        self.Hx = self._generate_plus_x_trans_matrix()
        self.Hy = self._generate_plus_y_trans_matrix()

        pass 
    def set_code_word(bottom_row_start_sequence):
        pass 
    
    def generate_errors(self):
        cutoff = self.p*self.decip
        for i in len(self.board):
            if random.randrange(0, self.decip) <= cutoff:
                self.board[i] ^= 1
                
    
    def bit_to_rc(self, bit):
        """
         This functions maps from bit index (0 to   (L**2)//2) - 1) to row, column mapping
        # DOESNT yet handle bits that are too big 


        in bit notation we think of all the bits being in a line: 
        So we have bit 0, bit 1, ... all the way until the last bit ((L**2)//2) - 1 
        [0, 1, 2,  ...................  ((L**2)//2) - 1 ]
        
        However, we can picture these as being on the 
        L//2 by L  board. 
        
        In numpy, if we reshape the  ((L**2)//2) - 1  to a L//2 by L array, 
        then bit 0 will get mapped to the 
        0th row, 0th column.  
        This is called row, column notation 
        Bit 1 will get mapped to the 0th row, 1st column.
        ...
        L//2 gets mapped to the 1st row, 0th column
        etc. 
        
        I show here: 
        
        [  [0 1 2  3 ...............................    (L//2) -1]
            L//2
                .
                .            
                .
                [(L - 1) * (L//2)  ....................    ((L**2)//2) - 1 ]
            ]
        
        """
        row_len = self.L 

        rindx = bit // row_len
        cindx = bit%row_len 
        return (rindx, cindx)
        
    def rc_to_bit(self, row, col):
        """
        # DOESNT yet handle row/columns that are too big 
        This functions maps from (row, column) indexing to bit  (0 to   (L**2)//2) - 1) indexing

        in bit notation we think of all the bits being in a line: 
        So we have bit 0, bit 1, ... all the way until the last bit ((L**2)//2) - 1 
        [0, 1, 2,  ...................  ((L**2)//2) - 1 ]
        
        However, we can picture these as being on the 
        L//2 by L  board. 
        
        In numpy, if we reshape the  ((L**2)//2) - 1  to a L//2 by L array, 
        then bit 0 will get mapped to the 
        0th row, 0th column.  
        This is called row, column notation 
        Bit 1 will get mapped to the 0th row, 1st column.
        ...
        L//2 gets mapped to the 1st row, 0th column
        etc. 
        
        I show here: 
        
        [  [0 1 2  3 ...............................    (L//2) -1]
            L//2
                .
                .            
                .
                [(L - 1) * (L//2)  ....................    ((L**2)//2) - 1 ]
            ]
        
        """
        bit = (row * self.L) + col
        return bit 
        
    def _generate_plus_x_trans_matrix(self):
        H = np.zeros((self.no_bits,  self.no_bits), dtype=int)
        for b in  range(self.no_bits):
            #H[new_bit][old_bit]
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

    def _generate_init_symmetry(self, start_arr = None):
        # fundamental symmetries start from the top instead of the bottom because numpy
        rect_board = np.reshape(self.board, (self.L//2, self.L))
        if not start_arr: 
            start_arr = np.zeros(self.L, dtype=int)
            start_arr[0] = 1
        rect_board[0] = start_arr
        for row in range(1, self.L//2):
            for bit in range(self.L):
                new_val =  rect_board[row - 1][(bit - 1)%self.L] ^  rect_board[row - 1][(bit )%self.L] ^  rect_board[row - 1][(bit + 1)%self.L]
                rect_board[row][bit] = new_val
        return rect_board
        
    def shift_by_x(self, board, power=1):
        Hx = np.linalg.matrix_power(self.Hx, power)
        return np.matmul(Hx, board)
    
    def shift_by_y(self, board, power=1):
        Hy = np.linalg.matrix_power(board, power)
        return np.matmul(Hy, board)
        
        
        
    def generate_parity_from_faces(self, stab_faces):
        # AHHHH 
        # create parity check matrix for each 
        # np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
        parity_mat = np.array([0] * self.L)
        
        for row in self.L//2:
            for col in self.L:
                if stab_faces[row][col] == 1:
                    a = self.rc_to_bit(row, col)
                    b = self.rc_to_bit((row)% self.L, (col - 1)%(self.L//2))
                    c = self.rc_to_bit((row)% self.L, (col + 1 )%(self.L//2))
                    d = self.rc_to_bit((row - 1)% self.L, (col )%(self.L//2))
                    new_stab = [0] * self.L 
                    new_stab[a] = 1
                    new_stab[b] = 1
                    new_stab[c] = 1
                    new_stab[d] = 1
                    np.append(parity_mat, [new_stab])
        return parity_mat
    
    def decode_fib_code(self):
        hori = [0] * (self.L**2)/2
        verti = [0] * (self.L**2)/2
        
        hori_stab_faces = self._generate_init_symmetry()
        verti_stab = self._generate_init_symmetry()
        
        hori_stab_faces.shape = (self.L**2)//2 # how bad is this 
        verti_stab.shape = (self.L**2)//2 # how bad is this 
        
        # center them on 0 bit
        hori_stab_faces = self.shift_by_y(hori_stab_faces)
        verti_stab = self.shift_by_x(self.shift_by_y(hori_stab_faces), self.L//2)
        



    
        hori_matching = pm.Matching() # TODO add weights 
        verti_matching = pm.Matching() # TODO add weights 
        

        
        hori_prediction = hori_matching.decode(self.board) 
        verti_prediction = verti_matching.decode(self.board) 
        
        hboard = self.board ^ hori_prediction #apply correction 
        vboard = self.board ^ verti_prediction
        dboard = self.board ^ (hori_prediction * verti_prediction) # only flip bits they agree should be flipped 
        
        
        
        
        hori_stab_faces.shape = (self.L//2, self.L) # how bad is this 
        verti_stab.shape = (self.L//2, self.L)  # how bad is this        
        # so here we go again... 
        for height in range(self.L/2):
            cur_parity_check = self.shift_by_y(cur_parity_check)
            for length in range(self.L):
                cur_parity_check = self.shift_by_x(cur_parity_check)
                # consider csc_matrix? are the gains even worth it? 
                cur_parity_check.shape = (self.L//2, self.L) # TODO  how slow is this reshaping?? 
                
                matching = pm.Matching(cur_parity_check) # TODO add weights 
                prediction = matching.decode(self.board)
                
                # TODO only run each graphing decoder once and save the output?  -- i guess can't bc dynamically changes the board
                # for a bit b:
                # hori is fundamental symmetry 
                
                
                
                
                
                 
        for bindx in range(self.no_bits ):
            pass 
        # for bit in codebits: 
        #      hori_sym = create_symmetry_with_hori_cross_along_bit(bit)
        #      vert_sym = create_symmetry_with_vert_cross_along_bit(bit)  
        #      
        #      h_corrections = run_matching(hori_sym)
        #      v_corrections = run_matching(vert_sym)
        # 
        #      h_syndrome = test_apply_correction(h_corrections)
        #      v_syndrome = test_apply_correction(v_corrections)
        #      d_syndrome = test_apply_correction(h_corrections + v_corrections)
        # 
        #      # NOW apply the only the correction that has the syndrome with the fewest 1s 
        #       min_syn = get_best_syndrome(h_syndrome, v_syndrome, d_syndrome)
        #       
        #       if h_syndrome == min_syn: 
        #           apply_correction_BUT_only_on_the_bit_in_question(h_corrections, bit)
        #       elif v_su... # you get the idea 
    


    
    
if __name__ == "__main__":
    fb = FibCode()
     
    
    