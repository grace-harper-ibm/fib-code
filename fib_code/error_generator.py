import numpy as np


def generate_swath_error(
    codeword_array: np.ndarray,
    width: int,
    offset: int = 0,
    probability_of_error=1,
    is_vertical=True,
):
    if probability_of_error <= 0:
        raise Exception("Can't create errors with zero probability")
    # TODO add is_wavy
    """Expects ALL codewords to be numpy array of shape: (L//2, L)"""
    error_mask = np.zeros(codeword_array.shape, dtype=int)

    num_rows = len(codeword_array)
    num_col = len(codeword_array[0])
    if is_vertical:
        if probability_of_error == 1:
            error_mask[:, offset : width + offset] = 1
        else:
            for i in range(num_rows):
                for j in range(width):
                    error_mask[i][(j + offset) % num_col] = np.random.choice(
                        [0, 1], p=[1 - probability_of_error, probability_of_error]
                    )

    else:
        height = width
        if probability_of_error == 1:
            error_mask[offset : offset + height, :] = 1
        else:
            for i in range(width):
                for j in range(num_col):
                    error_mask[(i + offset) % num_rows][j] = np.random.choice(
                        [0, 1], p=[1 - probability_of_error, probability_of_error]
                    )

    return error_mask ^ codeword_array, error_mask


""" 

    def generate_errors(self, original_board, p):
        board = copy.deepcopy(original_board)
        cutoff = p * self.decip
        for i in range(self.no_bits):
            if random.randrange(1, self.decip + 1) <= cutoff:
                board[i] ^= 1
        return board

"""
