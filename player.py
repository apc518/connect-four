"""
multi-threaded minimax connect four player
"""

__author__ = "Andy Chamberlain" # replace my name with yours
__license__ = "MIT"
__date__ = "February 2022"

from unittest.util import _count_diff_all_purpose
from connect4 import find_win

SUCCESS = 0
ERROR = 1

RED = 1
BLUE = 2

SMALL_INF = 2**16 # used as an exclusive lower bound for terminal evaluations
INF = 2**24 # used to evaluate winning quartets
BIG_INF = 2**32 # used as an exclusive upper bound for terminal evaluations 

def deep_equals(r1, r2):
    if len(r1) != len(r2):
        return False
    if len(r1[0]) != len(r2[0]):
        return False
    for i in range(0, len(r1)):
        for k in range(0, len(r1[0])):
            if r1[i][k] != r2[i][k]:
                return False
    return True

def increment_base3(digits):
    """
    increments a number composed of digits in base 3 
    returns 0 for normal operation, or 1 for overflow
    """
    for i in range(0, len(digits)):
        idx = len(digits) - i - 1
        if digits[idx] == 2:
            if idx == 0:
                return ERROR # number has reached its max
            continue
        else:
            digits[idx] += 1
            for k in range(idx+1, len(digits)):
                digits[k] = 0
            return SUCCESS


class ComputerPlayer:
    def __init__(self, id, difficulty_level=1):
        """
        Constructor, takes a difficulty level (likely the # of plies to look
        ahead), and a player ID that's either 1 or 2 that tells the player what
        its number is.
        """
        self.id = id
        self.difficulty_level = difficulty_level

        print(f"new ComputerPlayer created with difficulty {self.difficulty_level} and id {self.id}")

        # count total calls to self.eval()
        self.total_evals = 0

        # remember rack's we've evaluated that were terminal
        self.terminal_racks = {}
        self.non_terminal_racks = {}

        self.trans_table = {} # transposition table

        # store quartet evaluations in a table to avoid having to compute them repeatedly
        self.quartet_table = {}
        # populate the quartet evaluation table
        # take care of the very first one before the loop
        q = [0,0,0,0]
        self.quartet_table[tuple(q)] = 0
        piece_count_values = {1: 1, 2: 10, 3: 100, 4: INF}
        while increment_base3(q) == 0:
            opp_count = 0 # how many pieces in this quartet are the opponent's
            ai_count = 0 # how many pieces in this quarter are the ai's

            for item in q:
                if item == self.id:
                    ai_count += 1
                elif item != 0:
                    opp_count += 1
            
            sign = 1 if ai_count > opp_count else -1

            # if at least one of each color is present, this quartet is worth 0
            if opp_count > 0 and ai_count > 0:
                self.quartet_table[tuple(q)] = 0
                continue
            
            max_count = max(opp_count, ai_count)

            self.quartet_table[tuple(q)] = sign * piece_count_values[max_count]


    def possible_descendant(self, current_rack, other_rack):
        """ returns whether other_rack is a possible descendant of current_rack """

        for col_idx, col in enumerate(other_rack):
            for row_idx, val in enumerate(col):
                current_val = current_rack[col_idx][row_idx]
                if current_val != 0 and val != current_val:
                    return False
        return True


    def cleanup_trans_table(self, current_rack):
        """ removes positions from the transposition table that we know will never be used again in this game """

        keys_to_remove = []
        for key in self.trans_table:
            if not self.possible_descendant(current_rack, key):
                keys_to_remove.append(key)
        
        for k in keys_to_remove:
            del self.trans_table[k]


    def eval(self, rack):
        """ go through all vertical, horizontal, and diagonal quartets and add their values """

        self.total_evals += 1

        rack_tuple = tuple(map(tuple, rack))

        # use transposition table if possible
        if rack_tuple in self.trans_table:
            return self.trans_table[rack_tuple]

        total = 0

        # vertical quartets
        for col in rack:
            for i in range(0, len(col) - 3):
                quartet = col[i:i+4]
                total += self.quartet_table[tuple(quartet)]

        # horizontal quartets
        for row_idx in range(0, len(rack[0])):
            for col_idx in range(0, len(rack) - 3):
                pieces = (
                    rack[col_idx][row_idx],
                    rack[col_idx+1][row_idx],
                    rack[col_idx+2][row_idx],
                    rack[col_idx+3][row_idx]
                )
                total += self.quartet_table[pieces]
    
        # diagonal quartets
        for row_idx in range(0, len(rack[0]) - 3):
            for col_idx in range(0, len(rack) - 3):
                forward_slash_pieces = (
                    rack[col_idx][row_idx],
                    rack[col_idx+1][row_idx+1],
                    rack[col_idx+2][row_idx+2],
                    rack[col_idx+3][row_idx+3]
                )
                total += self.quartet_table[forward_slash_pieces]

                back_slash_pieces = (
                    rack[col_idx][len(rack[0]) - row_idx - 1],
                    rack[col_idx+1][len(rack[0]) - row_idx - 2],
                    rack[col_idx+2][len(rack[0]) - row_idx - 3],
                    rack[col_idx+3][len(rack[0]) - row_idx - 4]
                )
                total += self.quartet_table[back_slash_pieces]

        self.trans_table[rack_tuple] = total

        return total


    def make_move(self, rack, move, player_id):
        """ 
        makes a move in place on the rack
        returns: 0 for successful move, 1 for failed move
        """

        for row_idx in range(0, len(rack[0])):
            if rack[move][row_idx] == 0:
                rack[move][row_idx] = player_id
                return SUCCESS
        
        return ERROR


    def get_children(self, rack, player_id):
        """ returns all children of the given rack, assuming the given player is moving """

        # also orders the moves according the heuristic eval

        children = []
        
        for col_idx in range(0, len(rack)):
            for row_idx in range(0, len(rack[0])):
                if rack[col_idx][row_idx] == 0:
                    child = [x[:] for x in rack]
                    child[col_idx][row_idx] = player_id
                    children.append((self.eval(child), child))
                    break

        children.sort()
        if player_id == self.id:
            children = children[::-1]

        return [child for _, child in children]


    def minimax(self, rack, player_id, depth, alpha, beta):
        """ returns the evaluation of the rack """

        # TODO: move order beforehand, both for increased performance and 
        # to ensure that if this state is terminal, we will have recorded it in `self.terminal_racks`

        ### Leaf nodes (depth 0 and terminal states)

        heuristic_val = self.eval(rack)

        # if this rack is terminal or we are at max depth, return immediately
        # small_inf is a lower bound on terminal evaluations
        if abs(heuristic_val) >= SMALL_INF or depth == 0:
            return heuristic_val


        ### Alpha-beta pruning

        if player_id == self.id:
            val = -INF
            for child in self.get_children(rack, player_id):
                val = max(
                    val,
                    self.minimax(child, (BLUE+RED) - player_id, depth - 1, alpha, beta)
                )
                alpha = max(alpha, val)
                if val >= beta:
                    break
            
            return val
        else:
            val = INF
            for child in self.get_children(rack, player_id):
                val = min(
                    val,
                    self.minimax(child, (BLUE+RED) - player_id, depth - 1, alpha, beta)
                )
                beta = min(beta, val)
                if val <= alpha:
                    break
            
            return val


    def dispatch_job(self, rack_list, move, move_evals):
        new_rack = [x[:] for x in rack_list]
        self.make_move(new_rack, move, self.id)

        # we subtract 1 from the difficulty because this function is itself
        # called multiple times as the first level of search, by self.pick_move
        move_evals.append(
            self.minimax(
                new_rack,
                (BLUE + RED) - self.id,
                self.difficulty_level - 1,
                -INF,
                INF
            )
        )
    

    def pick_move(self, rack):
        """
        Pick the move to make. It will be passed a rack with the current board
        layout, column-major. A 0 indicates no token is there, and 1 or 2
        indicate discs from the two players. Column 0 is on the left, and row 0 
        is on the bottom. It must return an int indicating in which column to 
        drop a disc. The player current just pauses for half a second (for 
        effect), and then chooses a random valid move.
        """
        rack_list = list(map(list, rack))
        move_evals = []

        self.total_evals = 0

        self.cleanup_trans_table(rack_list)
        
        for move in range(len(rack)):
            self.dispatch_job(rack_list, move, move_evals)

        # print(f"Level {self.difficulty_level} Evals: ", end="")
        # print([x for x in move_evals])

        for col_idx, col in enumerate(rack):
            if col[-1] != 0:
                move_evals[col_idx] = -BIG_INF # never pick the move if its impossible

        # print(f"total evals: {self.total_evals}")
        # print(f"transposition table size: {len(self.trans_table)}")

        # input("next move...")

        return move_evals.index(max(move_evals))