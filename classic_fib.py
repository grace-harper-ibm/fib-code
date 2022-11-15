# grug brain 
# https://www.youtube.com/watch?v=h1cevveAaPs
# https://www.youtube.com/watch?v=rZjpsT7VgNU
import copy
import datetime
import logging
import math
import random

import networkx as nx
import rustworkx as rx
from this import d

tt = datetime.datetime.now()
import numpy as np
import pymatching as pm
import rustworkx as rx
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
    def __init__(self, L=8,p=0.001, decip=1000, code_bottom_row_start_sequence=None, pause=1000, error_board_override=None):
        assert math.log2(L) % 1 == 0, "L must be some 2**n where n is an int >= 1"
        logging.basicConfig(filename= "logs/" + f'L={L}_' + str(tt)+'fibcode_probs.log', encoding='utf-8', level=logging.INFO)
        self.logger = logging
        self.L = L # len
        self.no_cols = L
        self.no_rows = L//2 
        self.no_bits = (L**2)//2 # no bits
        self.p = p #probability of error on a given bit
        self.decip = decip
        self.pause = pause
        # fund_sym 
        self.original_code_board = np.zeros(self.no_bits, dtype=int)
        if error_board_override is not None:
            if p > 0:
                raise Exception("To use error_board_override, p must equal 0")
            self.original_errors_board = error_board_override
        else:
            self.original_errors_board =  self.generate_errors()
        self.board = copy.deepcopy(self.original_errors_board)
        self.fundamental_symmetry = self._generate_init_symmetry()
        self.fundamental_symmetry.shape = (self.L//2, self.L)
        self.fundamental_stabilizer_parity_check_matrix, self.fundamental_parity_rows_to_faces = self.generate_check_matrix_from_faces(self.fundamental_symmetry)
        self.fundamental_symmetry.shape = self.no_bits
        self.fundamental_single_error_syndromes = self.generate_all_possible_error_syndromes(self.fundamental_stabilizer_parity_check_matrix)
        self.Hx = self._generate_plus_x_trans_matrix()
        self.Hy = self._generate_plus_y_trans_matrix()
        
        self.all_stab_faces = np.ones((self.L//2, L), dtype=int)
        self.all_stabs_check_mat, self.all_stabs_parity_rows_to_faces = self.generate_check_matrix_from_faces(self.all_stab_faces)

        
        self.logger.info(f" original code baord is  {self.original_code_board}")
    
        self.logger.info(f" original_errors_board is  {self.original_errors_board}")
        self.logger.info(f" error board is code {self.board}")
        self.logger.info(f" initial symmetry is: {self.fundamental_symmetry}")
        self.logger.info(f"fundamental_stabilizer_parity_check_matrix is : {self.fundamental_stabilizer_parity_check_matrix}")
        #self.logger.info(f"fundamental_single_error_syndromes is : {self.fundamental_single_error_syndromes}")
        self.logger.info(f" Hx {self.Hx}")
        self.logger.info(f" Hy is code {self.Hy}")


    def set_code_word(self, bottom_row_start_sequence):
        raise NotImplementedError("wish you were here...")
    
    def generate_errors(self):
        board = copy.deepcopy(self.original_code_board)
        cutoff = self.p*self.decip
        for i in range(self.no_bits):
            if random.randrange(1, self.decip + 1) <= cutoff:
                board[i] ^= 1
        return board 
                
    
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
            new_bit = (b + self.L) %  self.no_bits
            H[new_bit][b] = 1
        return H 

    def _generate_init_symmetry(self, start_arr = None):
        # fundamental symmetries start from the top instead of the bottom because numpy
        rect_board = np.zeros((self.L//2, self.L), dtype=int)
        if start_arr is None: 
            start_arr = np.zeros(self.L, dtype=int)
            start_arr[(self.L//2) - 1] = 1
        rect_board[0] = start_arr
        for row in range(1, self.L//2):
            for bit in range(self.L):
                new_val =  rect_board[row - 1][(bit - 1)%self.L] ^  rect_board[row - 1][(bit )%self.L] ^  rect_board[row - 1][(bit + 1)%self.L]
                rect_board[row][bit] = new_val
        return rect_board
    
    def _generate_init_code_word(self, start_arr = None):
        # fundamental symmetries start from the top instead of the bottom because numpy
        rect_board = np.zeros((self.L//2, self.L), dtype=int)
        if start_arr is None: 
            start_arr = np.zeros(self.L, dtype=int)
            start_arr[(self.L//2) - 1] = 1
        rect_board[(self.L//2) - 1] = start_arr
        for row in range( self.L//2, 1, -1):
            for bit in range(self.L):
                new_val =  rect_board[row + 1][(bit - 1)%self.L] ^  rect_board[row + 1][(bit )%self.L] ^  rect_board[row + 1][(bit + 1)%self.L]
                rect_board[row][bit] = new_val
        return rect_board 
    
        
    def shift_by_x(self, bitarr, power=1): 
        # shifts by x + 1 aka right 
        power = power % self.L
        Hx = np.linalg.matrix_power(self.Hx, power)
        sol =  np.matmul(Hx, bitarr)
        sol = sol.astype(int)
        return sol
    
    def shift_by_y(self, bitarr, power=1): 
        power = power % (self.L//2)
        Hy = np.linalg.matrix_power(self.Hy, power)
        sol =  np.matmul(Hy, bitarr)
        sol = sol.astype(int)
        return sol 

        
    def _calc_syndrome(self,check_matr, board=None):
        if board is None: 
            board = self.board
        #  % 2 # TODO numpy almost certainly has a way of efficiently dealing w binary matrices -- figure that out 
        sol =  np.matmul(check_matr, board) % 2 
        sol = sol.astype(int)
        return sol
        
    def generate_check_matrix_from_faces(self, stab_faces):
        """
        STABS:           xxx
                                  
        """
        # TODO slow because of all the shape changing 
        """REQUIRES L//2 x L """
        # AHHHH 
        # create parity check matrix for each 
        # np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
        
        faces_to_stabs_rows = {}  # {x:y} where row x of the parity check matrix has face centered at bit b
        parity_mat = np.empty((0, self.no_bits), int)
        
        stab_row = 0
        for row in range(self.no_rows):
            for col in range(self.no_cols):
                if stab_faces[row][col] == 1:
                    a = self.rc_to_bit(row, col)
                    b = self.rc_to_bit((row)% self.no_rows, (col - 1)%self.no_cols)
                    c = self.rc_to_bit((row)% self.no_rows, (col + 1 )%self.no_cols)
                    d = self.rc_to_bit((row - 1)% self.no_rows, col %self.no_cols) # changed to point the other direction
                    new_stab = np.array([0] * self.no_bits)
                    new_stab[a] = 1
                    new_stab[b] = 1
                    new_stab[c] = 1
                    new_stab[d] = 1
                    parity_mat = np.append(parity_mat, [new_stab], axis = 0)
                    faces_to_stabs_rows[stab_row] = b
                    
        return parity_mat, faces_to_stabs_rows
   
    def generate_all_possible_error_syndromes(self, parity_check_matrix, no_bits = None):
        def map_and_update(face, staberr_id_count):
            if face not in board2staberr:
                staberr_face = staberr_id_count
                board2staberr[face] = staberr_face
                staberr2board[staberr_face] = face 
                staberr_id_count += 1 
            return board2staberr[face], staberr_id_count 
        
        # "there's gotta be a smarter way to do this "
        if no_bits is None:
            no_bits = self.no_bits
        error_pairs = set() # (stab_face1, stab_face2, errorbit)
        single_error = np.zeros(no_bits, dtype=int)
        
        staberr2board = {}
        board2staberr = {}
        staberr_id_count = 0 # if we start with a parity check for a symmetry not the fundamental symmetry, the labellings with be a bit wonky
        
    
        for b in range(no_bits):
            if no_bits > 10 and b % (no_bits//10) == 0:
                self.logger.info(f"on bit: {b} and error set looks like: {error_pairs}")
        
            ## set up new single error 
            # clear prev_bit 
            prev_bit = (b - 1) % no_bits
            single_error[prev_bit] = 0 
            # set new error 
            single_error[b] = 1
            
            ## what do it light? 
            lighted = self._calc_syndrome(parity_check_matrix, single_error)
            stabs = (lighted== 1).nonzero()[0]
            
            
            if  len(stabs) %2 !=0:
                emsg = f"Minor panic. Error on  bit {b} causes a BAD syndrome: {stabs} for lighted: {lighted}"
                self.logger.error(emsg) # TODO just do this via inspection on 1s per column in stab parity check matrix 
                raise Exception(emsg)
            
            if len(stabs) > 0:
                for indx in range(0,len(stabs), 2):                     
                    bindx_stab0 = stabs[indx]
                    bindx_stab1 = stabs[indx + 1]
                    
                    staberr_s0, staberr_id_count = map_and_update(bindx_stab0, staberr_id_count)
                    staberr_s1, staberr_id_count = map_and_update(bindx_stab1, staberr_id_count)
                    staberr_ebit, staberr_id_count = map_and_update(b, staberr_id_count)
                        
                    
                    error_pairs.add((staberr_s0, staberr_s1, staberr_ebit))
        
        return error_pairs, board2staberr, staberr2board



    def ret2net(self, graph: rx.PyGraph): # stolen from Wootton 
        """Convert rustworkx graph to equivalent networkx graph."""
        nx_graph = nx.MultiGraph() 
        for j, stabid in enumerate(graph.nodes()):
            nx_graph.add_node(j)
            nx.set_node_attributes(nx_graph, {j: stabid}, str(stabid))
        for j, (n0, n1) in enumerate(graph.edge_list()):
            nx_graph.add_edge(n0, n1, fault_id=j)
        return nx_graph

    def generate_error_syndrome_graph(self, parity_check_matrix, board_size):
        all_possible_errors = self.generate_all_possible_error_syndromes(parity_check_matrix, board_size)
        matching_graph =  self.error_pairs2graph(all_possible_errors)
        return matching_graph
    
    def error_pairs2graph(self, error_graphs, no_stabs=None):
        if no_stabs is None:
            no_stabs = len(self.fundamental_stabilizer_parity_check_matrix) # rows of parity check matrix 
        staberr2node = {}
        node2staberr = {} 
        graph = rx.PyGraph()
        
        # for in range(no_stabs):
        #     graph.add(i)
        # for i in range(no_stabs): # +1 is boundary node
        #     S.add_node(i)
        # # copied James Wootton's code, a true hero. 
        # for n0, n1, e in error_graphs:
        #     j = S.nodes().index(n0) 
        #     k = S.nodes().index(n1) 
        #     S.add_edge(j, k, str(e))
        # return S 
            
            
    
    def decode_fib_code(self):
        # generate graphs and mappings
        
        # center everything on b 
        
        # for b in bits 
        
        hori_stab_faces = copy.deepcopy(self.fundamental_symmetry)
        verti_stab_faces = copy.deepcopy(self.fundamental_symmetry)
        
        hori_stab_faces.shape = (self.L**2)//2 # how bad is this 
        verti_stab_faces.shape = (self.L**2)//2 # how bad is this 
        
        hori_parity = self.generate_check_matrix_from_faces(hori_stab_faces)
        verti_parity  = self.generate_check_matrix_from_faces(verti_stab_faces)
        
        # generate graph and board2graph mappings 
        hori_graph_error_pairs, hori_board2staberr, hori_staberr2board = self.generate_all_possible_error_syndromes(hori_parity) # TODO test this
        verti_graph_error_pairs, verti_board2staberr, verti_staberr2board = self.generate_all_possible_error_syndromes(verti_parity) # TODO test this
        
        hori_matching_graph =  self.error_pairs2graph(hori_graph_error_pairs)
       
        
        
        
        
        
        # Now, center things on zero
        hori_stab_faces = self.shift_by_x(self.shift_by_y(hori_stab_faces),  power=(-self.L//2) + 1)
        verti_stab_faces = self.shift_by_x(self.shift_by_y(hori_stab_faces))
        
        for row in range(len(hori_parity)):
            hori_parity[row] =self.shift_by_x(self.shift_by_y(hori_parity[row] ),  power=(-self.L//2) + 1)    
        
        for row in range(len(verti_parity)):
            verti_parity[row] =self.shift_by_x(self.shift_by_y( verti_parity[row] ))  #0th entry here should correspond to midpoint of bottom triangle and should be on 0th bit 
        
        # TODO center special bit
        # TODO center staberror2board
        # TODO center board2staberror
        
        
        
        
        
        
        
    
        
        round_count = 0

        for _ in range(self.L//2): # will wrap around to all bits 
            hori_stab_faces = self.shift_by_y(hori_stab_faces)
            verti_stab_faces = self.shift_by_y(verti_stab_faces)
            for _ in range(self.L):
                round_count += 1
                if self.no_bits > 10:
                    if round_count % (self.no_bits//10) == 0: # log every additional 10% of board coverage
                        self.logger.info(f" currently on round: {round_count}")
                        self.logger.info(f"current board is {self.board}")
                        
                # TODO consider csc_matrix? are the gains even worth it? 
                hori_stab_faces = self.shift_by_x(hori_stab_faces)
                verti_stab_faces = self.shift_by_x(verti_stab_faces)
                
                
                hori_stab_faces_rect = np.reshape(hori_stab_faces, (self.L//2, self.L))
                verti_stab_faces_rect = np.reshape(verti_stab_faces, (self.L//2, self.L))

                hori_check_matrix, hori_parity_rows_to_faces  = self.generate_check_matrix_from_faces(hori_stab_faces_rect)  # TODO make a special +y/x for check mats
                verti_check_matrix, verti_parity_rows_to_faces = self.generate_check_matrix_from_faces(verti_stab_faces_rect)

                hori_matching_graph = self.ret2net(self.generate_error_syndrome_graph(hori_check_matrix, self.no_bits))
                verti_matching_graph =  self.ret2net(self.generate_error_syndrome_graph(verti_check_matrix, self.no_bits))
                
    
                hori_matching = pm.Matching(hori_matching_graph) # TODO add weights 
                verti_matching = pm.Matching(verti_matching_graph) # TODO add weights 
        
                hori_syndrome_mat = self._calc_syndrome(hori_check_matrix) 
                verti_syndrome_mat = self._calc_syndrome(verti_check_matrix)
                
                # # make sure hori_syndrome matches TODO make sure nodes are what I think they are. 
                # hori_syndrome_post_graph = [0]*len(hori_syndrome_mat) 
                # for i in range(len(hori_syndrome_mat)):
                #     if hori_syndrome_mat[i] == 1:
                #         hori_syndrome_post_graph[hori_matching_graph.nodes().index(i) ] = 1
                
                # verti_syndrome_post_graph = [0] * len(verti_syndrome_mat)
                # for i in range(len(verti_syndrome_mat)):
                #     if verti_syndrome_mat[i] == 1:
                #         verti_syndrome_post_graph[verti_matching_graph.nodes().index(i)] = 1 
                
 
                hori_prediction = hori_matching.decode(hori_syndrome_mat) 
                verti_prediction = verti_matching.decode(verti_syndrome_mat) 
                
                self.logger.info(f"HORI: check matrix\n {hori_check_matrix}\n  syndrome:\n {hori_syndrome_mat}")
                self.logger.info(f"VERTI: check matrix\n {verti_check_matrix}\n  syndrome:\n {verti_syndrome_mat}")
        
                # take stabs to faces:
                hori_correction = np.zeros(self.no_bits, dtype=int)
                verti_correction = np.zeros(self.no_bits, dtype=int)
                
                for h in hori_prediction:
                    hori_correction[hori_parity_rows_to_faces[h]] = 1 
                
                for v in verti_prediction:
                    verti_correction[verti_parity_rows_to_faces[v]] = 1
        
        
                hboard = self.board ^ hori_correction #apply correction 
                vboard = self.board ^ verti_correction
                dboard = self.board ^ (hori_correction * verti_correction) # only flip bits they agree should be flipped 
       
                # test new syndroms 
        
                hcorsynd  = [(self._calc_syndrome(self.all_stabs_check_mat, hboard)== 1).sum(), hboard, "hori"]
                vcorsynd  = [(self._calc_syndrome(self.all_stabs_check_mat, vboard)== 1).sum(), vboard, "verti"]
                dcorsynd  = [(self._calc_syndrome(self.all_stabs_check_mat, dboard)== 1).sum(), dboard, "dcor"]
        
                opts = [hcorsynd, vcorsynd, dcorsynd]
        
                winner = min(opts, key=lambda x: x[0])
                self.board = winner[1]         # update board to best one 
                self.logger.info(f"initial correction correction information: f{winner}")
                
                if (winner[0] == 0):
                    return "yay!!"
            
   
    