import numpy as np

from classic_fib import FibCode

if __name__ == "__main__":
   
    # Issues: Decoder should pass if there is only a codeword 
    L = 32
    print(L)
    f = FibCode(L, p=0.05)
    print(f.decode_fib_code())