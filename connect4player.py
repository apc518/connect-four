"""
minimax connect four player with a-b pruning, move ordering, and
optional multithreading and thread pruning

since command line args are handled by connect4.py, which is not my code,
options are simply defined as constants at the top of this file.
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


### CONSTANTS
SUCCESS = 0
ERROR = 1

RED = 1
BLUE = 2

SMALL_INF = 2**16 # used as a lower bound for terminal evaluations
INF = 2**24 # used to evaluate winning quartets
BIG_INF = 2**32 # used as an upper bound for terminal evaluations 


@njit
def eval_jit(rack : np.ndarray, quartet_table: np.ndarray):
    """
    returns a heuristic evaluation of the rack

    quartet_table must be a linearly indexable collection containing
    numerical items, and have a length of >= 81. Preferably, a length-81 numpy
    array of ints.

    goes through all vertical, horizontal, and diagonal quartets, adding up
    the evaluations of each quartet (group of four connected slots), as
    specified by quartet_table
    """

    total = 0

    # vertical quartets
    for col_idx in range(len(rack)):
        for i in range(len(rack[col_idx]) - 3):
            idx = rack[col_idx][i] * 27 + rack[col_idx][i+1] * 9 + \
                rack[col_idx][i+2] * 3 + rack[col_idx][i+3]
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
            forward_slash_idx = rack[col_idx][row_idx] * 27 + \
                rack[col_idx+1][row_idx+1] * 9 + \
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

        if DO_MULTIPROCESSING:
            self.manager = Manager()

        # generate all possible quartets
        tv = [0,1,2] # possible tile values
        all_quartets = [[x,y,w,z] for x in tv for y in tv for w in tv for z in tv]

        piece_count_values = {1: 1, 2: 10, 3: 100, 4: INF}

        # quartet_table will store the evaluation of each quartet at the index
        # corresponding to the base-3 number of the quartet.
        # so, for a quartet [1, 2, 0, 1], its evaluation would be stored at
        # index 1 * 3**3 + 2 * 3**2 + 0 * 3**1 + 1 * 3**0, aka 46 in decimal
        self.quartet_table = [0] * 81

        # populate quartet_table
        for q in all_quartets[1:]: # skip the first one since its already 0
            """
            Each quartet is evaluated according to the following rules:

            Point value is positive if it favors ComputerPlayer, negative otherwise.
            Contains at least one disc of each color: 0
            Contains 4 discs of the same color: ??INF (since one player has won).
            Contains 3 discs of the same color (and 1 empty): ??100.
            Contains 2 discs of the same color (and 2 empties): ??10.
            Contains 1 disc (and 3 empties): ??1.
            """

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
            # table is filled with 0s by default so no action is needed
            if opp_count > 0 and ai_count > 0:
                continue
            
            max_count = max(opp_count, ai_count)

            self.quartet_table[idx] = sign * piece_count_values[max_count]
        
        self.np_quartet_table = np.array(self.quartet_table)

        # call eval once so numba compiles it
        eval_jit(np.array([[0]*6]*7), self.np_quartet_table)


    def eval_heur(self, rack):
        return eval_jit(rack, self.np_quartet_table)


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
        """ returns all children of the given rack, assuming the given player 
        is moving items in the returned list are tuples of the form
        (rack, move, eval)
        they are sorted by eval, according to whether the given player_id
        is the opponent or is self.id
        So, if player_id == self.id, we sort the children in descending order
        by eval otherwise, sort ascending
        """

        children = []
        
        for move in range(len(rack)):
            for row_idx in range(len(rack[0])):
                if rack[move][row_idx] == 0:
                    child = np.copy(rack)
                    child[move][row_idx] = player_id
                    children.append((self.eval_heur(child), move, child))
                    break

        # move order best-first
        children.sort()
        if player_id == self.id:
            children = children[::-1]

        return [(child, move, val) for val, move, child in children]


    def minimax(self, rack, player_id, depth, alpha, beta, heuristic_val):
        """
        use the minimax algorithm to evaluate a position with alpha beta pruning

        returns: a number representing the heuristic evaluation based on
        the player inputted
        """

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
                    self.minimax(child, (BLUE+RED) - player_id, 
                        depth - 1, alpha, beta, heur_val)
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
                    self.minimax(child, (BLUE+RED) - player_id,
                        depth - 1, alpha, beta, heur_val)
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


    def prune_threads(self, jobs, move_dict):
        """
        Periodically check to see if the current move evaluations merit an 
        immediate stop.

        This function is blocking so it should always be called on its own
        thread/process.

        ### Arguments
        jobs -- list of tuples (Process job, int move)
        move_dict -- a Manager dict, of the form { int move : int evaluation }


        ### Details

        this is not alpha beta pruning but it has the same function, that is,
        it will never change the outcome of a search but it may make the
        search faster

        the two conditions it checks for are: 
        
        1. if any of the moves has been evaluated as a guaranteed win, that is,
        with a score of greater than SMALL_INF

        *2. if all possible moves except for one have been evaluated as
        guaranteed losses, then we know to immediately stop searching and pick
        the move that has not yet been marked a guaranteed loss

        *note that a final position that hasnt been evaluated as a loss yet may
        turn out to also be a guaranteed loss, but it will probably be the
        latest/deepest in the case, so it's still probably the best move.
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

            # ensure that exactly one job hasnt finished
            if len(items) != len(jobs) - 1:
                return

            # ensure all jobs finished so far returned a guaranteed loss
            for k, v in items:
                # if we find a non loss, we cant prune anything, so just return
                if v > -SMALL_INF:
                    return
            
            # at this point, the guard clauses above not returning confirms
            # that we have indeed evaluated all but one possible move, and
            # that all so far are guaranteed losses, so we can finish early
            kill_jobs()
            
            # find the move whose job has not finished yet
            for j, move in jobs:
                if move not in keys:
                    # all other moves are -INF, so 0 will be chosen over them
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
        is on the bottom. It must return an integer indicating in which column to 
        drop a disc.
        """
        start_time = time.time()

        rack_list = list(map(list, rack))
        np_rack = np.array(rack_list)
        move_evals_list = [-BIG_INF for _ in range(len(rack))]

        self.total_evals = 0

        if DO_MULTIPROCESSING:
            move_eval_dict = self.manager.dict()

            # one job per possible move
            # as a tuple of the job itself and the move it processes
            jobs = []
            for child, move, heur_val in self.get_children(np_rack, self.id):
                # if this move is an instant win, just return it immediately
                if heur_val >= SMALL_INF:
                    return move

                j = Process(target=self.eval_move,
                    args=(child, move, move_eval_dict, heur_val))
                jobs.append((j, move))
                j.start()

            # avoid unnecessary computation on forced moves
            # or shallow guaranteed wins
            pruner = Process(target=self.prune_threads, args=(jobs, move_eval_dict))
            pruner.start()

            for j, _ in jobs:
                j.join()

            pruner.kill()

            for k in move_eval_dict:
                move_evals_list[k] = move_eval_dict[k]
        else:
            for child, move, heur_val in self.get_children(np_rack, self.id):
                # if this move is an instant win, just return it immediately
                if heur_val >= SMALL_INF:
                    return move
                
                self.eval_move(child, move, move_evals_list, heur_val)

        for col_idx, col in enumerate(rack):
            if col[-1] != 0:
                # never pick the move if its impossible
                move_evals_list[col_idx] = -BIG_INF

        decision = move_evals_list.index(max(move_evals_list))

        # print(f"Decided on move {decision+1} after {(time.time() - start_time):.03f}s")
        # print(f"{move_evals_list=}")

        return decision