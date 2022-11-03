# https://www.youtube.com/watch?v=RHyE_erqAe0
import math
import unittest

import numpy as np
from classic_fib import FibCode


class FibCodeTest(unittest.TestCase):
    """sanity checks"""
    
    def test_basic(self):
        f = FibCode(8)
        f = FibCode(32)

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
            
            
            

if __name__ == "__main__":
    unittest.main()
