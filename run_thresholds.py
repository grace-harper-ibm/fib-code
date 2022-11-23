# test iid
import datetime as dt
import logging

import numpy as np

from fib_code.classic_fib_decoder import ClassicFibDecoder
from fib_code.code_generator import generate_init_code_word
from fib_code.error_generator import generate_swath_error
from fib_code.utils.logging_utils import new_logger_for_classic_fib_code_decoder

""" 
L=8: [ #(error_prob, fail_rate, num_shots) ]
L=16:[]

[
    [8, (),()],
    [16,()],
    ...
]

"""

res_file = "results.txt"


def res_log(info):
    with open(res_file, "a") as f:
        f.write("\n")
        f.write(str(info))


def cnow():
    return dt.datetime.now()


all_L = [4, 8, 32, 64]
all_p = np.linspace(0.01, 0.2, 10).round(decimals=2)

res_log(f"beginning program at {cnow()}")

results = []
num_shots = 1000
for L in all_L:
    start_L = cnow()
    res_log(f"\n\n\n\n\n============================================================")
    res_log(f"STARTING L={L} as {cnow()}\n")
    l_row = []
    for p in all_p:
        res_log(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        Lp_logger = new_logger_for_classic_fib_code_decoder(
            "logs", f"L={L}p={p}_logger_", logging.INFO
        )  # too many logs rip x_x
        start_p = cnow()
        res_log(f"p={p} for starting L={L}  at {cnow()}")
        if L == 32 and p < 0.06:
            continue
        success_no = 0
        for round in range(num_shots):
            codeword = generate_init_code_word(
                L
            )  # generate an initial codeword. The default one bit at bottom center and cellular automata rules upward
            error_board, error_mask = generate_swath_error(
                codeword, L, probability_of_error=p
            )  # setting width to L and vertical=True makes iid noise
            f = ClassicFibDecoder(
                error_board, Lp_logger
            )  # give this class the errored codeword
            f.decode_fib_code()
            f.board.shape = (L // 2, L)
            if (f.board == codeword).all():
                success_no += 1
        stop_p = cnow()
        p_info = (p, success_no / num_shots)
        l_row.append(p_info)
        print(p_info)

        res_log(f"p={p} for L={L} took {stop_p - start_p} and produced: {p_info}")
        res_log(f"p={p} for L={L}  DONE\n")

    stop_L = cnow()
    results.append(l_row)
    res_log(f"L={L} took {stop_L - start_L} with result:\n {l_row}")
    res_log(f"L={L} is DONE\n\n\n")
