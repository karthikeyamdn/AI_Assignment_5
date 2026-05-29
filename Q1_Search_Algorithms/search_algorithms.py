"""
Q1: Game Tree Search Algorithms
================================
Implements four search algorithms used in adversarial game playing:
  1. Minimax Search
  2. Alpha-Beta Pruning
  3. Heuristic Alpha-Beta (depth-limited with evaluation function)
  4. Monte-Carlo Tree Search (MCTS)

Each algorithm is demonstrated on Tic-Tac-Toe, a classic two-player zero-sum game.
"""

import math
import random
import time
from copy import deepcopy
from collections import defaultdict

# ─────────────────────────────────────────────────────────
#  TIC-TAC-TOE GAME ENGINE  (shared by all four algorithms)
# ─────────────────────────────────────────────────────────

class TicTacToe:
    """
    Board representation for Tic-Tac-Toe.
      'X'  – maximising player
      'O'  – minimising player
      ' '  – empty cell
    """

    def __init__(self):
        self.board = [' '] * 9          # flat 3×3 grid
        self.current_player = 'X'

    def clone(self):
        """Return a deep copy of the current game state."""
        g = TicTacToe()
        g.board = self.board[:]
        g.current_player = self.current_player
        return g

    def get_legal_moves(self):
        """Return indices of all empty cells."""
        return [i for i, v in enumerate(self.board) if v == ' ']

    def make_move(self, index):
        """Place the current player's mark at *index* and switch turns."""
        if self.board[index] != ' ':
            raise ValueError(f"Cell {index} is already occupied.")
        self.board[index] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def check_winner(self):
        """Return 'X', 'O', 'Draw', or None (game still in progress)."""
        wins = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),   # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),   # columns
            (0, 4, 8), (2, 4, 6),               # diagonals
        ]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]
        if ' ' not in self.board:
            return 'Draw'
        return None

    def is_terminal(self):
        return self.check_winner() is not None

    def display(self):
        """Pretty-print the board."""
        b = self.board
        print(f"\n {b[0]} | {b[1]} | {b[2]}")
        print("───┼───┼───")
        print(f" {b[3]} | {b[4]} | {b[5]}")
        print("───┼───┼───")
        print(f" {b[6]} | {b[7]} | {b[8]}\n")


# ─────────────────────────────────────────────────────────
#  ALGORITHM 1 – MINIMAX SEARCH
# ─────────────────────────────────────────────────────────

class MinimaxAgent:
    """
    Classic Minimax algorithm (no pruning).

    • MAX player ('X') tries to maximise the score.
    • MIN player ('O') tries to minimise the score.
    • Terminal scores: +1 (X wins), -1 (O wins), 0 (draw).
    • Every node in the game tree is visited (exponential in depth).
    """

    def __init__(self):
        self.nodes_explored = 0

    def minimax(self, game: TicTacToe, is_maximising: bool) -> int:
        """
        Recursively compute the minimax value of *game*.

        Parameters
        ----------
        game           : current game state
        is_maximising  : True when it is MAX's (X's) turn

        Returns
        -------
        int : +1 if X wins, -1 if O wins, 0 for draw
        """
        self.nodes_explored += 1

        winner = game.check_winner()
        if winner == 'X':
            return 1
        if winner == 'O':
            return -1
        if winner == 'Draw':
            return 0

        moves = game.get_legal_moves()

        if is_maximising:
            best = -math.inf
            for move in moves:
                child = game.clone()
                child.make_move(move)
                score = self.minimax(child, False)
                best = max(best, score)
            return best
        else:
            best = math.inf
            for move in moves:
                child = game.clone()
                child.make_move(move)
                score = self.minimax(child, True)
                best = min(best, score)
            return best

    def best_move(self, game: TicTacToe) -> int:
        """Return the index of the best move for the current player."""
        self.nodes_explored = 0
        is_max = (game.current_player == 'X')
        best_score = -math.inf if is_max else math.inf
        best_mv = None

        for move in game.get_legal_moves():
            child = game.clone()
            child.make_move(move)
            score = self.minimax(child, not is_max)
            if is_max and score > best_score:
                best_score, best_mv = score, move
            elif not is_max and score < best_score:
                best_score, best_mv = score, move

        return best_mv


# ─────────────────────────────────────────────────────────
#  ALGORITHM 2 – ALPHA-BETA PRUNING
# ─────────────────────────────────────────────────────────

class AlphaBetaAgent:
    """
    Minimax with Alpha-Beta pruning.

    Alpha (α) – best value MAX can guarantee so far (lower bound).
    Beta  (β) – best value MIN can guarantee so far (upper bound).

    A branch is pruned when α ≥ β, because neither player would
    ever choose to play into that subtree.
    """

    def __init__(self):
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def alpha_beta(self, game: TicTacToe, alpha: float,
                   beta: float, is_maximising: bool) -> int:
        """
        Alpha-Beta minimax search.

        Parameters
        ----------
        game          : current game state
        alpha         : best score MAX can guarantee on this path
        beta          : best score MIN can guarantee on this path
        is_maximising : True when it is MAX's (X's) turn

        Returns
        -------
        int : minimax value (identical to plain minimax, computed faster)
        """
        self.nodes_explored += 1

        winner = game.check_winner()
        if winner == 'X':
            return 1
        if winner == 'O':
            return -1
        if winner == 'Draw':
            return 0

        moves = game.get_legal_moves()

        if is_maximising:
            value = -math.inf
            for move in moves:
                child = game.clone()
                child.make_move(move)
                value = max(value, self.alpha_beta(child, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:          # β-cutoff
                    self.nodes_pruned += 1
                    break
            return value
        else:
            value = math.inf
            for move in moves:
                child = game.clone()
                child.make_move(move)
                value = min(value, self.alpha_beta(child, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:          # α-cutoff
                    self.nodes_pruned += 1
                    break
            return value

    def best_move(self, game: TicTacToe) -> int:
        """Return the index of the best move for the current player."""
        self.nodes_explored = 0
        self.nodes_pruned = 0
        is_max = (game.current_player == 'X')
        best_score = -math.inf if is_max else math.inf
        best_mv = None

        for move in game.get_legal_moves():
            child = game.clone()
            child.make_move(move)
            score = self.alpha_beta(child, -math.inf, math.inf, not is_max)
            if is_max and score > best_score:
                best_score, best_mv = score, move
            elif not is_max and score < best_score:
                best_score, best_mv = score, move

        return best_mv


# ─────────────────────────────────────────────────────────
#  ALGORITHM 3 – HEURISTIC ALPHA-BETA (depth-limited)
# ─────────────────────────────────────────────────────────

class HeuristicAlphaBetaAgent:
    """
    Depth-limited Alpha-Beta with a heuristic evaluation function.

    When the search reaches *max_depth* without finding a terminal
    state, it calls evaluate() to estimate the position's value.
    This makes the algorithm practical for large state spaces where
    searching to terminal nodes would be too slow.

    Heuristic evaluation for Tic-Tac-Toe
    ────────────────────────────────────
    For each of the 8 lines (rows, columns, diagonals):
      +10  if the line contains 2 X's and no O  (X one away from winning)
      +  1 if the line contains 1 X  and no O
      - 10 if the line contains 2 O's and no X  (O one away from winning)
      -  1 if the line contains 1 O  and no X
         0 if the line contains both X and O     (blocked, useless)
    """

    WIN_LINES = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def evaluate(self, game: TicTacToe) -> float:
        """
        Static heuristic evaluation of a non-terminal position.
        Positive values favour X; negative values favour O.
        """
        score = 0
        for a, b, c in self.WIN_LINES:
            line = [game.board[a], game.board[b], game.board[c]]
            x_count = line.count('X')
            o_count = line.count('O')
            if o_count == 0:           # open for X
                score += 10 ** x_count   # 1, 10, or 100
            if x_count == 0:           # open for O
                score -= 10 ** o_count
        return score

    def heuristic_alpha_beta(self, game: TicTacToe, depth: int,
                              alpha: float, beta: float,
                              is_maximising: bool) -> float:
        """
        Depth-limited alpha-beta with heuristic cutoff.

        Parameters
        ----------
        game          : current game state
        depth         : remaining search depth
        alpha, beta   : pruning bounds
        is_maximising : True when it is MAX's turn

        Returns
        -------
        float : exact minimax value (if terminal) or heuristic estimate
        """
        self.nodes_explored += 1

        winner = game.check_winner()
        if winner == 'X':
            return 1000        # big positive (exact win)
        if winner == 'O':
            return -1000       # big negative (exact loss)
        if winner == 'Draw':
            return 0

        if depth == 0:
            return self.evaluate(game)   # heuristic cut-off

        moves = game.get_legal_moves()

        if is_maximising:
            value = -math.inf
            for move in moves:
                child = game.clone()
                child.make_move(move)
                value = max(value,
                            self.heuristic_alpha_beta(
                                child, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    self.nodes_pruned += 1
                    break
            return value
        else:
            value = math.inf
            for move in moves:
                child = game.clone()
                child.make_move(move)
                value = min(value,
                            self.heuristic_alpha_beta(
                                child, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    self.nodes_pruned += 1
                    break
            return value

    def best_move(self, game: TicTacToe) -> int:
        """Return the index of the best move for the current player."""
        self.nodes_explored = 0
        self.nodes_pruned = 0
        is_max = (game.current_player == 'X')
        best_score = -math.inf if is_max else math.inf
        best_mv = None

        for move in game.get_legal_moves():
            child = game.clone()
            child.make_move(move)
            score = self.heuristic_alpha_beta(
                child, self.max_depth, -math.inf, math.inf, not is_max)
            if is_max and score > best_score:
                best_score, best_mv = score, move
            elif not is_max and score < best_score:
                best_score, best_mv = score, move

        return best_mv


# ─────────────────────────────────────────────────────────
#  ALGORITHM 4 – MONTE-CARLO TREE SEARCH (MCTS)
# ─────────────────────────────────────────────────────────

class MCTSNode:
    """
    A single node in the MCTS search tree.

    Attributes
    ----------
    game      : game state at this node
    parent    : parent MCTSNode (None for root)
    move      : the move that led to this node from its parent
    children  : list of child MCTSNode objects (expanded lazily)
    wins      : total reward accumulated through this node
    visits    : number of times this node has been visited
    untried   : legal moves not yet expanded into children
    """

    def __init__(self, game: TicTacToe, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried = game.get_legal_moves()

    def is_fully_expanded(self):
        return len(self.untried) == 0

    def is_terminal(self):
        return self.game.is_terminal()

    def ucb1(self, c: float = 1.414) -> float:
        """
        Upper Confidence Bound (UCB1) formula used to select nodes.

            UCB1 = (wins / visits) + c * sqrt(ln(parent.visits) / visits)

        The exploitation term (wins/visits) favours nodes with high
        win rates; the exploration term pushes towards less-visited nodes.
        """
        if self.visits == 0:
            return float('inf')
        exploit = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, c: float = 1.414):
        """Return the child with the highest UCB1 value."""
        return max(self.children, key=lambda n: n.ucb1(c))

    def expand(self):
        """Expand one untried move and return the new child node."""
        move = self.untried.pop(random.randint(0, len(self.untried) - 1))
        child_game = self.game.clone()
        child_game.make_move(move)
        child = MCTSNode(child_game, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result: float):
        """Back-propagate the simulation result up the tree."""
        self.visits += 1
        self.wins += result


class MCTSAgent:
    """
    Monte-Carlo Tree Search.

    The algorithm iterates four phases until the time/iteration budget
    is exhausted:

    1. Selection   – walk the tree using UCB1 until a node that is not
                     fully expanded (or is terminal) is found.
    2. Expansion   – add one new child for an untried move.
    3. Simulation  – play out the game randomly (rollout policy).
    4. Back-prop   – propagate the result back to the root,
                     updating visit counts and win totals.

    After all iterations, the child of the root with the most visits
    (robust child) is chosen as the best move.
    """

    def __init__(self, iterations: int = 1000, time_limit: float = None):
        """
        Parameters
        ----------
        iterations : maximum number of MCTS iterations
        time_limit : optional wall-clock limit in seconds (overrides iterations)
        """
        self.iterations = iterations
        self.time_limit = time_limit

    # ── Phase 1: Selection ──────────────────────────────────
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Descend the tree using UCB1 until we reach a node that is
        not fully expanded or is a terminal state.
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    # ── Phase 3: Simulation (random rollout) ────────────────
    def _simulate(self, game: TicTacToe) -> float:
        """
        Play random moves until the game ends and return the reward
        from the perspective of the ROOT player ('X'):
          +1  if X wins
          -1  if O wins
           0  for a draw
        """
        sim = game.clone()
        while not sim.is_terminal():
            move = random.choice(sim.get_legal_moves())
            sim.make_move(move)
        winner = sim.check_winner()
        if winner == 'X':
            return 1.0
        if winner == 'O':
            return -1.0
        return 0.0

    # ── Phase 4: Back-propagation ────────────────────────────
    def _backpropagate(self, node: MCTSNode, result: float):
        """
        Walk back to the root and update each node's statistics.
        The reward is negated at each level because the two players
        have opposing objectives.
        """
        while node is not None:
            node.update(result)
            result = -result      # flip perspective at each level
            node = node.parent

    def best_move(self, game: TicTacToe) -> int:
        """Run MCTS and return the best move index."""
        root = MCTSNode(game.clone())
        start = time.time()

        for _ in range(self.iterations):
            if self.time_limit and (time.time() - start) > self.time_limit:
                break

            # 1. Selection
            node = self._select(root)

            # 2. Expansion (if node is not terminal)
            if not node.is_terminal():
                node = node.expand()

            # 3. Simulation
            result = self._simulate(node.game)

            # 4. Back-propagation
            self._backpropagate(node, result)

        # Choose the most-visited child (robust child criterion)
        best = max(root.children, key=lambda n: n.visits)
        return best.move


# ─────────────────────────────────────────────────────────
#  HELPER – play a full game between two agents
# ─────────────────────────────────────────────────────────

def play_game(x_agent, o_agent, verbose: bool = True) -> str:
    """
    Run a full Tic-Tac-Toe game.

    Parameters
    ----------
    x_agent : agent with a best_move(game) method (plays 'X')
    o_agent : agent with a best_move(game) method (plays 'O')
    verbose : print moves and board if True

    Returns
    -------
    str : 'X', 'O', or 'Draw'
    """
    game = TicTacToe()
    agents = {'X': x_agent, 'O': o_agent}

    while not game.is_terminal():
        current = game.current_player
        move = agents[current].best_move(game)
        game.make_move(move)
        if verbose:
            print(f"Player {current} plays cell {move}")
            game.display()

    result = game.check_winner()
    if verbose:
        print(f"Result: {result}")
    return result


# ─────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  Minimax  vs  Alpha-Beta  (should agree on every move)")
    print("=" * 55)
    mm  = MinimaxAgent()
    ab  = AlphaBetaAgent()
    hab = HeuristicAlphaBetaAgent(max_depth=6)
    mcts = MCTSAgent(iterations=500)

    game = TicTacToe()
    game.display()

    mm_move  = mm.best_move(game)
    ab_move  = ab.best_move(game)
    hab_move = hab.best_move(game)
    mcts_move = mcts.best_move(game)

    print(f"Minimax        best move: {mm_move}  (nodes explored: {mm.nodes_explored})")
    print(f"Alpha-Beta     best move: {ab_move}  (nodes explored: {ab.nodes_explored}, pruned: {ab.nodes_pruned})")
    print(f"Heuristic A-B  best move: {hab_move}  (nodes explored: {hab.nodes_explored}, pruned: {hab.nodes_pruned})")
    print(f"MCTS           best move: {mcts_move}  (iterations: 500)")

    print("\n" + "=" * 55)
    print("  Full game: Minimax (X) vs Alpha-Beta (O)")
    print("=" * 55)
    play_game(MinimaxAgent(), AlphaBetaAgent(), verbose=True)

    print("\n" + "=" * 55)
    print("  Full game: MCTS (X) vs Heuristic Alpha-Beta (O)")
    print("=" * 55)
    play_game(MCTSAgent(iterations=800), HeuristicAlphaBetaAgent(max_depth=4), verbose=True)
