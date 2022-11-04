from classic_fib import FibCode

L = 4
f = FibCode(L)
print("initted original board")
print(f.original_code_board)
f.decode_fib_code()
print("original board")
print(f.original_code_board)
print()
print("corrected board:")
print(f.board)
