import numpy as np

from classic_fib import FibCode

if __name__ == "__main__":
   
    # Issues: Decoder should pass if there is only a codeword 
    L = 8
    f = FibCode(L, p=0)
    print(f.decode_fib_code())