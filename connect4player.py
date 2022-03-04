"""
minimax connect four player with a-b pruning, move ordering, and
optional transposition table and multithreading

since command line args are handled by connect4.py, which is not my code,
options are simply defined as constants at the top of this file.

In my testing, having the transposition table off and multithreading on yields
the best performance. However, with multithreading off, using the transposition
table does yield much better performance than having it off.

Time taken for level 9 to respond to an opening move of column 4 (index 3):
            parallel on	    parallel off
table on    23.0            3.3
table off   1.6             6.9

I expect that the mutex lock that Manager uses to syncronize the dictionary
is what causes the slowdown when multithreading is enabled. Since all the threads
are trying to access the table all the time, it results in more time being
spent syncronizing them than on anything else.

This is supported by the results of timing pick_move on an empty rack:

I issued two `time` commands, first with the transposition table off, then on. In both
cases, multithreading was on.

```
> time python test.py

real    0m1.496s
user    0m7.547s
sys     0m0.050s
> time py test.py

real    0m22.677s
user    0m41.559s
sys     0m24.732s
```

test.py contained the following:

```python
from connect4player import ComputerPlayer, RED

empty_rack = ((0,)*6,)*7
player = ComputerPlayer(RED, difficulty_level=9)
player.pick_move(empty_rack)
```

So, with the syncronized Manager.dict() transposition table, the proportion 
of time spent in kernel mode by the OS was much higher, which is what we'd
expect from waiting around for mutex locks.


A final note on performance: the transposition table being enabled is probably
faster than multithreading if you have less than 4 cpu cores. On my 6-core machine,
the speedup when using multithreading is roughly 3-4x, and on a machine with 8+ cores
the speedup would probably be a bit better. However, with fewer than 4 cores,
especially a dual or single core machine, the speedup benefit from threading will be
marginal at best. However, the transposition table will boost performance for
single threaded processing by about 2x regardless of how many cores your machine has.

"""

__author__ = "Andy Chamberlain"
__license__ = "MIT"
__date__ = "February 2022"

from multiprocessing import Manager, Process
import time

import numpy as np
from numba import njit

## OPTIONS
DO_MULTIPROCESSING = True
USE_TRANSPOSITION_TABLE = False


### CONSTANTS
SUCCESS = 0
ERROR = 1

RED = 1
BLUE = 2

SMALL_INF = 2**16 # used as a lower bound for terminal evaluations
INF = 2**24 # used to evaluate winning quartets
BIG_INF = 2**32 # used as an upper bound for terminal evaluations 

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


# rack and quartet_table should be numpy arrays
@njit
def eval_jit(rack : np.ndarray, quartet_table: np.ndarray):
    total = 0

    # vertical quartets
    for col_idx in range(len(rack)):
        for i in range(len(rack[col_idx]) - 3):
            idx = rack[col_idx][i] * 27 + rack[col_idx][i+1] * 9 + rack[col_idx][i+2] * 3 + rack[col_idx][i+3]
            total += quartet_table[idx]

    # horizontal quartets
    for row_idx in range(len(rack[0])):
        for col_idx in range(len(rack) - 3):
            idx = rack[col_idx][row_idx] * 27 + rack[col_idx+1][row_idx] * 9 + \
                rack[col_idx+2][row_idx] * 3 + rack[col_idx+3][row_idx]

            total += quartet_table[idx]

    # diagonal quartets
    for row_idx in range(len(rack[0]) - 3):
        for col_idx in range(len(rack) - 3):
            forward_slash_idx = rack[col_idx][row_idx] * 27 + rack[col_idx+1][row_idx+1] * 9 + \
                rack[col_idx+2][row_idx+2] * 3 + rack[col_idx+3][row_idx+3]

            total += quartet_table[forward_slash_idx]

            back_slash_idx = rack[col_idx][len(rack[0]) - row_idx - 1] * 27 + \
                rack[col_idx+1][len(rack[0]) - row_idx - 2] * 9 + \
                rack[col_idx+2][len(rack[0]) - row_idx - 3] * 3 + \
                rack[col_idx+3][len(rack[0]) - row_idx - 4]

            total += quartet_table[back_slash_idx]
    
    return total


class ComputerPlayer:
    def __init__(self, id, difficulty_level=1):
        """
        Constructor, takes a difficulty level (likely the # of plies to look
        ahead), and a player ID that's either 1 or 2 that tells the player what
        its number is.
        """
        self.id = id
        self.difficulty_level = difficulty_level

        # print(f"new ComputerPlayer created with difficulty {self.difficulty_level} and id {self.id}")

        # count total calls to self.eval()
        self.total_evals = 0

        if DO_MULTIPROCESSING:
            self.manager = Manager()
            self.trans_table = self.manager.dict() # transposition table
        else:
            self.trans_table = {}

        # store quartet evals at the base-3 number index of the quartet
        self.quartet_table = [0] * 81

        # populate the quartet evaluation table
        # take care of the very first one before the loop
        q = [0,0,0,0]
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

            idx = q[0] * 27 + q[1] * 9 + q[2] * 3 + q[3]

            # if at least one of each color is present, this quartet is worth 0
            if opp_count > 0 and ai_count > 0:
                self.quartet_table[idx] = 0
                continue
            
            max_count = max(opp_count, ai_count)

            self.quartet_table[idx] = sign * piece_count_values[max_count]


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

        if USE_TRANSPOSITION_TABLE:
            rack_tuple = tuple(map(tuple, rack))

            if rack_tuple in self.trans_table:
                return self.trans_table[rack_tuple]

        total = eval_jit(np.array(rack), np.array(self.quartet_table))

        if USE_TRANSPOSITION_TABLE:
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
        """ returns all children of the given rack, assuming the given player is moving 
        items in the returned list are tuples of the form (rack, move, eval)
        they are sorted by eval, according to whether the given player_id
        is the opponent or is self.id
        So, if player_id == self.id, we sort the children in descending order by eval
        otherwise, sort ascending
        """

        children = []
        
        for move in range(len(rack)):
            for row_idx in range(len(rack[0])):
                if rack[move][row_idx] == 0:
                    child = [x[:] for x in rack]
                    child[move][row_idx] = player_id
                    children.append((self.eval(child), move, child))
                    break

        # move order best-first
        children.sort()
        if player_id == self.id:
            children = children[::-1]

        return [(child, move, val) for val, move, child in children]


    def minimax(self, rack, player_id, depth, alpha, beta, heuristic_val):
        """ returns the evaluation of the rack """

        # if we are at max depth or this rack is terminal, return immediately
        # small_inf is a lower bound on terminal evaluations
        if depth == 0 or abs(heuristic_val) >= SMALL_INF:
            return heuristic_val


        ### Alpha-beta pruning

        if player_id == self.id:
            val = -INF
            for child, move, heur_val in self.get_children(rack, player_id):
                val = max(
                    val,
                    self.minimax(child, (BLUE+RED) - player_id, depth - 1, alpha, beta, heur_val)
                )
                alpha = max(alpha, val)
                if val >= beta:
                    break
            
            return val
        else:
            val = INF
            for child, move, heur_val in self.get_children(rack, player_id):
                val = min(
                    val,
                    self.minimax(child, (BLUE+RED) - player_id, depth - 1, alpha, beta, heur_val)
                )
                beta = min(beta, val)
                if val <= alpha:
                    break
            
            return val


    def eval_move(self, rack, move, move_evals, heur_val):
        """ 
        updates the move_evals collection at the given move
        with an evaluation of the given rack

        used exclusively for the very top level of the tree
        """

        # we subtract 1 from the difficulty because this function is itself
        # called multiple times as the first level of search, by self.pick_move
        move_evals[move] = self.minimax(
            rack,
            (BLUE + RED) - self.id,
            self.difficulty_level - 1,
            -INF,
            INF,
            heur_val
        )


    def prune(self, jobs, move_dict):
        """
        periodically check to see if the current move evaluations merit an immediate stop

        This function is blocking so it should always be called on its own thread/process.

        this is not alpha beta pruning but it has the same function, that is, it will never
        change the outcome of a search but it may make the search faster

        the two conditions it checks for are: 
        
        1. if any of the moves has been evaluated as a guaranteed win, that is, with a score
        of greater than SMALL_INF

        *2. if all possible moves except for one have been evaluated as guaranteed losses,
        then we know to immediately stop searching and pick the move that has not yet
        been marked a guaranteed loss

        *note that a final position that hasnt been evaluated as a loss yet may turn out to
        also be a guaranteed loss, but it will probably be the latest/deepest in the case,
        so it's still probably the best move.
        """

        def kill_jobs():
            for j, _ in jobs:
                j.kill()

        def try_prune():
            # check if move_dict is in a state such that we can immediately finish
            items = list(move_dict.items())
            keys = [k for k, v in items]
            for k, v in items:
                if v > SMALL_INF:
                    kill_jobs()
                    return


            ### now check for all losses but one move is missing

            # ensure there is exactly one job that hasnt finished
            if len(items) != len(jobs) - 1:
                return

            # ensure all jobs finished so far returned a guaranteed loss
            for k, v in items:
                # if we find a non loss, we cant prune anything, so just return
                if v > -SMALL_INF:
                    return
            
            kill_jobs()
            
            # find the move whose job has not finished yet, set its value as 0 and return
            for j, move in jobs:
                if move not in keys:
                    move_dict[move] = 0
                    return


        while True:
            time.sleep(0.1) # sleep for 0.1 seconds
            try_prune()



    def pick_move(self, rack):
        """
        Pick the move to make. It will be passed a rack with the current board
        layout, column-major. A 0 indicates no token is there, and 1 or 2
        indicate discs from the two players. Column 0 is on the left, and row 0 
        is on the bottom. It must return an int indicating in which column to 
        drop a disc. The player current just pauses for half a second (for 
        effect), and then chooses a random valid move.
        """
        start_time = time.time()

        rack_list = list(map(list, rack))
        move_evals_list = [-BIG_INF for _ in range(len(rack))]

        self.total_evals = 0

        self.cleanup_trans_table(rack_list)

        if DO_MULTIPROCESSING:
            move_eval_dict = self.manager.dict()

            # one job per possible move, as a tuple of the job itself and the move it processes
            jobs = []
            for child, move, heur_val in self.get_children(rack_list, self.id):
                # if this move is an instant win, just return it immediately
                if heur_val >= SMALL_INF:
                    return move

                j = Process(target=self.eval_move, args=(child, move, move_eval_dict, heur_val))
                jobs.append((j, move))
                j.start()

            # avoid unnecessary computation on forced moves or shallow guaranteed wins
            if self.id == RED:
                pruner = Process(target=self.prune, args=(jobs, move_eval_dict))
                pruner.start()

            for j, _ in jobs:
                j.join()

            if self.id == RED:
                pruner.kill()

            for k in move_eval_dict:
                move_evals_list[k] = move_eval_dict[k]
        else:
            for child, move, heur_val in self.get_children(rack_list, self.id):
                # if this move is an instant win, just return it immediately
                if heur_val >= SMALL_INF:
                    return move
                
                self.eval_move(child, move, move_evals_list, heur_val)

        for col_idx, col in enumerate(rack):
            if col[-1] != 0:
                move_evals_list[col_idx] = -BIG_INF # never pick the move if its impossible

        decision = move_evals_list.index(max(move_evals_list))

        print(f"Decided on move {decision+1} after {(time.time() - start_time):.03f}s")
        # print(f"{move_evals_list=}")

        return decision