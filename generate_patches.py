from pathlib import Path
import sys

section_dir = Path.cwd()
sys.path.append(str(section_dir))
print(section_dir.name)

OTHELLO_ROOT = (section_dir / "othello_world").resolve()
OTHELLO_MECHINT_ROOT = (OTHELLO_ROOT / "mechanistic_interpretability").resolve()

# if not OTHELLO_ROOT.exists():
#     !git clone https://github.com/likenneth/othello_world

sys.path.append(str(OTHELLO_MECHINT_ROOT))

from mech_interp_othello_utils import (
    OthelloBoardState
)

# import deepcopy
from copy import deepcopy
import torch as t

from training_utils import get_state_stack_num_flipped
from utils import square_to_tuple

import random
# set random seed
random.seed(42)

# This function test all possibilities of two moves to see if they can play move_clean on the second move
def get_corrupt_move(board_corrupt, move_clean):
    # board_state_initial = board_corrupt.state.copy()
    valid_moves_corrupt_initial = board_corrupt.get_valid_moves().copy()
    for move1 in valid_moves_corrupt_initial:
        board_corrupt_copy = deepcopy(board_corrupt)
        # board_corrupt_copy = board_corrupt.copy()
        board_corrupt_copy.umpire(move1)
        valid_moves_corrupt = board_corrupt_copy.get_valid_moves().copy()
        # board_corrupt.state = board_state_initial
        if move_clean in valid_moves_corrupt:
            return move1
    return move_clean

def calculate_next_board(board_clean, board_corrupt):
    valid_moves_clean = board_clean.get_valid_moves()
    move_clean = random.choice(valid_moves_clean)
    move_corrupt = get_corrupt_move(board_corrupt, move_clean)
    board_clean.umpire(move_clean)
    board_corrupt.umpire(move_corrupt)
    if move_clean == move_corrupt:
        did_corruption_occur = False
        return did_corruption_occur
    valid_moves_clean = board_clean.get_valid_moves()
    move_clean_new = random.choice(valid_moves_clean)
    move_corrupt_new = move_clean
    board_clean.umpire(move_clean_new)
    board_corrupt.umpire(move_corrupt_new)
    did_corruption_occur = True
    return did_corruption_occur

def play_same_move(board_clean, board_corrupt):
    valid_moves_clean = board_clean.get_valid_moves()
    valid_moves_corrupt = board_corrupt.get_valid_moves()
    valid_moves_both = list(set(valid_moves_clean) & set(valid_moves_corrupt))
    if len(valid_moves_both) == 0:
        was_move_possible = False
        return was_move_possible
    move = random.choice(valid_moves_both)
    board_clean.umpire(move)
    board_corrupt.umpire(move)
    was_move_possible = True
    return was_move_possible
    
def get_patch():
    move = 0
    board_clean = OthelloBoardState()
    board_corrupt = OthelloBoardState()
    did_corruption_occur = False
    while not did_corruption_occur:
        did_corruption_occur = calculate_next_board(board_clean, board_corrupt)
        if did_corruption_occur:
            move += 2
        else:
            move += 1
    corrupted_square = board_corrupt.history[-1]
    corrupted_move = move-1
    was_move_possible = True
    while was_move_possible:
        was_move_possible = play_same_move(board_clean, board_corrupt)
        move += 1
    assert len(board_clean.history) == len(board_corrupt.history)
    length = len(board_clean.history)
    return board_clean.history, board_corrupt.history, corrupted_square, corrupted_move, length

def get_flipped_list(board_history, square):
    if type(board_history) == list:
        board_history = t.Tensor(board_history).to(dtype=t.int64)
    assert len(board_history.shape) == 1
    end_move = len(board_history)
    square_tuple = square_to_tuple(square)
    focus_states_num_flipped = get_state_stack_num_flipped(board_history.unsqueeze(dim=0))
    flipped_list = list(focus_states_num_flipped[0, :, square_tuple[0], square_tuple[1]])
    first_flip = True if flipped_list[0] == 1 else False
    flipped_list = [first_flip] + [flipped_list[i-1] < flipped_list[i] for i in range(1, end_move)]
    flipped_list = [i for i in range(0, end_move) if flipped_list[i]]
    return flipped_list

def generate_patch(min_length: int = 55, min_flipped_clean: int = 1, max_flipped_corrupt: int = 100, min_tries: int = 1000, same_flips: bool = True, max_first_flip: int = 20):
    for _ in range(min_tries):
        board_history_clean, board_history_corrupt, corrupted_square, corrupted_move, length = get_patch()
        flipped_list_clean = get_flipped_list(board_history_clean, corrupted_square)
        flipped_list_corrupt = get_flipped_list(board_history_corrupt, corrupted_square)
        if length < min_length:
            continue
        if len(flipped_list_clean) < min_flipped_clean:
            continue
        if len(flipped_list_corrupt) > max_flipped_corrupt:
            continue
        if len(flipped_list_clean) > 0:
            if flipped_list_clean[0] > max_first_flip:
                continue
        if len(flipped_list_corrupt) > 0:
            if flipped_list_corrupt[0] > max_first_flip:
                continue
        if same_flips and not flipped_list_clean == flipped_list_corrupt:
            continue
        if not same_flips and set(flipped_list_clean) & set(flipped_list_corrupt):
            continue
        patch_info = dict()
        patch_info["board_history_clean"] = board_history_clean
        patch_info["board_history_corrupt"] = board_history_corrupt
        patch_info["corrupted_square"] = corrupted_square
        patch_info["corrupted_move"] = corrupted_move
        patch_info["length"] = length
        patch_info["flipped_list_clean"] = flipped_list_clean
        patch_info["flipped_list_corrupt"] = flipped_list_corrupt
        return patch_info
        

if __name__ == "__main__":
    patch_info = generate_patch()
    print(patch_info)
    print([1, 2] == [1, 2])

    flipped_list = get_flipped_list([37, 45, 26, 38, 39, 31, 53, 52, 51, 34, 43, 20, 42, 59,
                         44, 61, 30, 50, 12, 18, 17, 16, 8, 13, 14, 25, 41, 0,
                         33, 19, 57, 24, 49, 5, 54, 48, 9, 55, 60, 15, 4, 22,
                         58, 10, 2, 46, 1, 32, 21, 56, 11, 3, 6, 40, 62, 23, 7,
                         47, 63], 38)
    assert flipped_list[0] == 4
    assert flipped_list[1] == 5