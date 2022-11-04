# https://www.youtube.com/watch?v=RHyE_erqAe0
import math
import unittest

import numpy as np
from classic_fib import FibCode


class FibCodeTest(unittest.TestCase):
    """sanity checks"""
    
    def test_basic(self):
        f = FibCode(8)
        

    def test_bit_representation_trans(self):
        f = FibCode(32) # L = 32 
        trans_tests = [ 
                        [(0,0), 0],
                        [(4, 1),  129],
                        [(15, 3), 483],
                        [(0, 31), 31],
                        [(1,31 ), 63],
                        [(15, 31), 511],
                        [(6, 19), 211]
                        ]
        for test in trans_tests:
            rc = test[0]
            sol = test[1]
            assert f.rc_to_bit(rc[0], rc[1]) == sol, f"{rc} did not convert to correct bit rep: {sol}"

        for test in trans_tests:
            bit = test[1]
            sol = test[0]
            assert f.bit_to_rc(bit) == sol, f"{bit} did not convert to correct rc rep: {sol}" 
    
    def test_x_shift(self):
        L = 32 
        f = FibCode(L)
        input = np.array(list(range((L**2)//2)))
        
        # shift_by_x
        # shift by 1 
        shiftby1 =np.array( [[(L * (row + 1)) - 1 ] + list(range((L * row), (L * row) + L - 1)) for row in range(L//2)])
        shiftby1.shape = ((L**2)//2)


        xtests = [
            [L, np.array(list(range((L**2)//2)))],
             [1, shiftby1]
        ]
        for xt in xtests:
            s = xt[0]
            sol =  xt[1]
            
            ans = f.shift_by_x(input, s) 
            assert (ans == sol).all(), f"for {s} the result: {ans} is not {sol}"            
            assert ans.shape == sol.shape, f"for ans shape: {s.shape}  is not {sol.shape} on {s}"
        
      
    def test_y_shift(self):
        L = 32 
        f = FibCode(L)
        input = np.array(list(range((L**2)//2)))
        
        shiftbyminus1 = np.array([np.array(list(range((L*row), ((L*row) + L)))) for row in range(1, L//2)])
        shiftbyminus1 = np.append(shiftbyminus1, [list(range(L))]) # put top row at bottom 
        
        shiftby1 = np.array(list(range(L * (L//2 - 1), L * (L//2 ))))
        shiftby1 = np.append(shiftby1, np.array([np.array(list(range((L*row), ((L*row) + L)))) for row in range(0, (L//2) - 1)]))

        ytests = [
            [-1, shiftbyminus1],
            [1, shiftby1],
            [L//2, np.array(list(range((L**2)//2)))]
        ]
        
        for yt in ytests:
            s = yt[0]
            sol = yt[1]
            ans = f.shift_by_y(input, s)
            print(s)
            assert (ans == sol).all(), f"for {s} the ans: {ans} was not {sol} "
            assert ans.shape == sol.shape, f" for {s} the ans shape: {ans.shape} did not match the sol shape: {sol.shape}"
        
      
      
      
if __name__ == "__main__":
    unittest.main()
