"""
multi-threaded minimax connect four player
"""

__author__ = "Andy Chamberlain" # replace my name with yours
__license__ = "MIT"
__date__ = "February 2022"

from connect4 import find_win

SUCCESS = 0
ERROR = 1

RED = 1
BLUE = 2

INF = 2**20

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

        # count total calls to self.eval()
        self.total_evals = 0
        self.total_evals_lock = None

        # remember rack's we've evaluated that were terminal
        self.terminal_racks = {}
        self.non_terminal_racks = {}

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


    def eval(self, rack):
        """ go through all vertical, horizontal, and diagonal quartets and add their values """

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

        # if this state was terminal, remember that
        if abs(total) == INF:
            # print(f"board evaluated terminal: {rack}")
            self.terminal_racks[tuple(map(tuple, rack))] = total

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


    def children(self, rack, player_id):
        """ returns all children of the rack given the player to move """

        children = []
        
        for col_idx in range(0, len(rack)):
            for row_idx in range(0, len(rack[0])):
                if rack[col_idx][row_idx] == 0:
                    child = [x[:] for x in rack]
                    child[col_idx][row_idx] = player_id
                    children.append(child)
                    break
        
        return children


    def minimax(self, rack, player_id, depth, alpha, beta):
        """ returns the evaluation of the rack """

        # TODO: move order beforehand, both for increased performance and 
        # to ensure that if this state is terminal, we will have recorded it in `self.terminal_racks`

        # check if we've seen it before and its a terminal state
        rack_tuple = tuple(map(tuple, rack))
        if rack_tuple in self.terminal_racks:
            return self.terminal_racks[rack_tuple]
        
        if depth == 0:
            return self.eval(rack)
        
        # if we have not seen it before as a terminal state
        if rack_tuple not in self.non_terminal_racks and find_win(rack_tuple) is not None:
            e = self.eval(rack)
            self.terminal_racks[rack_tuple] = e
            return e
        
        self.non_terminal_racks[rack_tuple] = 1

        if player_id == self.id:
            val = -INF
            for child in self.children(rack, player_id):
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
            for child in self.children(rack, player_id):
                mmx_res = self.minimax(child, (BLUE+RED) - player_id, depth - 1, alpha, beta)
                if mmx_res is None:
                    raise Exception("minimax returned None")
                val = min(
                    val,
                    mmx_res
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
        
        for move in range(7):
            self.dispatch_job(rack_list, move, move_evals)

        print("Evals: ", end="")
        print([x for x in move_evals])

        return move_evals.index(max(move_evals))