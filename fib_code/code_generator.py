import numpy as np


def generate_init_code_word(L, start_arr=None):
    # generates from bottom row up
    rect_board = np.zeros((L // 2, L), dtype=int)
    if start_arr is None:
        start_arr = np.zeros(L, dtype=int)
        start_arr[((L - 1) // 2)] = 1
    rect_board[(L // 2) - 1] = start_arr
    for row in range((L // 2) - 2, -1, -1):
        for bit in range(L):
            new_val = (
                rect_board[row + 1][(bit - 1) % L]
                ^ rect_board[row + 1][(bit) % L]
                ^ rect_board[row + 1][(bit + 1) % L]
            )
            rect_board[row][bit] = new_val
    return rect_board


""" .  
    @classmethod
    def generate_init_code_word(cls, L, start_arr=None):
        if start_arr is not None and sum(start_arr) != 1:
            raise ArgumentError(
                f"Can only have a single 1 in start_arr. All else should be 0 but you have: {start_arr}"
            )
        # generates from bottom row up
        rect_board = np.zeros((L // 2, L), dtype=int)
        if start_arr is None:
            start_arr = np.zeros(L, dtype=int)
            start_arr[((L - 1) // 2)] = 1
        rect_board[(L // 2) - 1] = start_arr
        for row in range((L // 2) - 2, -1, -1):
            for bit in range(L):
                new_val = (
                    rect_board[row + 1][(bit - 1) % L]
                    ^ rect_board[row + 1][(bit) % L]
                    ^ rect_board[row + 1][(bit + 1) % L]
                )
                rect_board[row][bit] = new_val
        return rect_board
        """
