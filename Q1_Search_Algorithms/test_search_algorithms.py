"""
test_search_algorithms.py
=========================
Unit tests for Q1: Minimax, Alpha-Beta, Heuristic Alpha-Beta, MCTS.

Run with:   python -m pytest test_search_algorithms.py -v
  or:       python test_search_algorithms.py
"""

import sys, math, random
sys.path.insert(0, '.')
from search_algorithms import (
    TicTacToe, MinimaxAgent, AlphaBetaAgent,
    HeuristicAlphaBetaAgent, MCTSAgent, play_game
)

# ── helpers ────────────────────────────────────────────────

def board_from(cells: str, player: str = 'X') -> TicTacToe:
    """
    Build a TicTacToe state from a 9-char string.
    Use 'X', 'O', '.' (empty).

    Example:  board_from("XOX O   X", 'O')
    """
    g = TicTacToe()
    g.board = [c if c != '.' else ' ' for c in cells]
    g.current_player = player
    return g


def run(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}  —  {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}  —  {e}")
        return False


# ══════════════════════════════════════════════════════════
#  TEST SUITE
# ══════════════════════════════════════════════════════════

class TestTicTacToe:

    def test_winner_row(self):
        g = board_from("XXX......", 'O')
        assert g.check_winner() == 'X', "Row 0 should be X's win"

    def test_winner_col(self):
        g = board_from("X..X..X..", 'O')
        assert g.check_winner() == 'X', "Column 0 should be X's win"

    def test_winner_diag(self):
        g = board_from("O..OO..OX", 'X')   # not a win yet
        g2 = board_from("O...O...O", 'X')
        assert g2.check_winner() == 'O', "Diagonal should be O's win"

    def test_draw(self):
        g = board_from("XOXOOXXXO", 'X')
        assert g.check_winner() == 'Draw', "Fully filled board should be draw"

    def test_in_progress(self):
        g = board_from("XO.......", 'X')
        assert g.check_winner() is None, "Game in progress should return None"

    def test_legal_moves(self):
        g = board_from("XO.X.....", 'X')
        moves = g.get_legal_moves()
        assert 2 in moves and 4 in moves, "Cells 2 and 4 should be legal"
        assert 0 not in moves and 1 not in moves, "Cells 0,1 already taken"

    def test_clone_independence(self):
        g = TicTacToe()
        clone = g.clone()
        clone.make_move(0)
        assert g.board[0] == ' ', "Original board must not be mutated by clone"


class TestMinimax:
    agent = MinimaxAgent()

    def test_win_in_one(self):
        # X can win immediately at cell 2
        g = board_from("XX.OO....", 'X')
        move = self.agent.best_move(g)
        assert move == 2, f"Minimax must take the winning move 2, got {move}"

    def test_block_opponent(self):
        # O is about to win at cell 8; X must block
        g = board_from("X..OO....", 'X')
        move = self.agent.best_move(g)
        assert move == 5, f"Minimax must block at 5, got {move}"

    def test_returns_valid_move(self):
        g = TicTacToe()
        move = self.agent.best_move(g)
        assert move in g.get_legal_moves(), "Move must be a legal cell index"

    def test_nodes_explored_positive(self):
        g = TicTacToe()
        self.agent.best_move(g)
        assert self.agent.nodes_explored > 0, "nodes_explored must be > 0"

    def test_never_loses(self):
        """Minimax X playing against random O should never lose."""
        random.seed(42)
        for _ in range(20):
            g = TicTacToe()
            while not g.is_terminal():
                if g.current_player == 'X':
                    mv = MinimaxAgent().best_move(g)
                else:
                    mv = random.choice(g.get_legal_moves())
                g.make_move(mv)
            assert g.check_winner() != 'O', "Minimax should never lose to random"


class TestAlphaBeta:
    mm  = MinimaxAgent()
    ab  = AlphaBetaAgent()

    def test_same_moves_as_minimax_empty(self):
        """Alpha-Beta must choose the same move as Minimax on empty board."""
        g = TicTacToe()
        assert self.ab.best_move(g) == self.mm.best_move(g)

    def test_same_moves_as_minimax_mid_game(self):
        """Alpha-Beta must agree with Minimax mid-game."""
        g = board_from("XO.X.....", 'O')
        assert self.ab.best_move(g) == self.mm.best_move(g)

    def test_pruning_reduces_nodes(self):
        """Alpha-Beta must explore fewer nodes than plain Minimax."""
        g = TicTacToe()
        self.mm.best_move(g)
        self.ab.best_move(g)
        assert self.ab.nodes_explored < self.mm.nodes_explored, \
            "Alpha-Beta must prune at least some nodes"

    def test_win_in_one(self):
        g = board_from("XX.OO....", 'X')
        assert self.ab.best_move(g) == 2

    def test_nodes_pruned_positive(self):
        g = TicTacToe()
        self.ab.best_move(g)
        assert self.ab.nodes_pruned > 0, "Some nodes must be pruned"


class TestHeuristicAlphaBeta:
    agent = HeuristicAlphaBetaAgent(max_depth=6)

    def test_win_in_one(self):
        g = board_from("XX.OO....", 'X')
        assert self.agent.best_move(g) == 2

    def test_returns_valid_move(self):
        g = TicTacToe()
        move = self.agent.best_move(g)
        assert move in g.get_legal_moves()

    def test_heuristic_positive_for_X_advantage(self):
        g = board_from("X.X......", 'O')     # X has two in a row
        score = self.agent.evaluate(g)
        assert score > 0, "X has advantage; heuristic must be positive"

    def test_heuristic_negative_for_O_advantage(self):
        g = board_from("...O.O...", 'X')     # O has two in a row
        score = self.agent.evaluate(g)
        assert score < 0, "O has advantage; heuristic must be negative"

    def test_depth_limit_explored_less_than_full(self):
        """Depth-3 search must explore fewer nodes than full minimax."""
        g = TicTacToe()
        hab = HeuristicAlphaBetaAgent(max_depth=3)
        mm  = MinimaxAgent()
        hab.best_move(g)
        mm.best_move(g)
        assert hab.nodes_explored < mm.nodes_explored


class TestMCTS:
    agent = MCTSAgent(iterations=500)

    def test_win_in_one(self):
        """MCTS must take an immediate winning move."""
        g = board_from("XX.OO....", 'X')
        move = self.agent.best_move(g)
        assert move == 2, f"MCTS must take winning move 2, got {move}"

    def test_returns_valid_move(self):
        g = TicTacToe()
        move = self.agent.best_move(g)
        assert move in g.get_legal_moves()

    def test_never_picks_occupied_cell(self):
        random.seed(0)
        for _ in range(10):
            g = TicTacToe()
            while not g.is_terminal():
                mv = MCTSAgent(iterations=200).best_move(g)
                assert g.board[mv] == ' ', "MCTS must not pick an occupied cell"
                g.make_move(mv)

    def test_win_rate_vs_random(self):
        """MCTS (X) should win or draw ≥ 80 % of games against random (O)."""
        random.seed(7)
        wins = draws = 0
        total = 30
        for _ in range(total):
            g = TicTacToe()
            while not g.is_terminal():
                if g.current_player == 'X':
                    mv = MCTSAgent(iterations=300).best_move(g)
                else:
                    mv = random.choice(g.get_legal_moves())
                g.make_move(mv)
            r = g.check_winner()
            if r == 'X':   wins  += 1
            if r == 'Draw': draws += 1
        rate = (wins + draws) / total
        assert rate >= 0.80, f"MCTS win+draw rate {rate:.0%} < 80 %"


# ── integration test ───────────────────────────────────────

class TestIntegration:

    def test_minimax_vs_alphabeta_same_outcome(self):
        """Full game: Minimax X vs Alpha-Beta O should always draw (both optimal)."""
        result = play_game(MinimaxAgent(), AlphaBetaAgent(), verbose=False)
        assert result == 'Draw', f"Two optimal agents must draw, got {result}"

    def test_game_terminates(self):
        """Any game between two agents must terminate."""
        result = play_game(MCTSAgent(iterations=200),
                           HeuristicAlphaBetaAgent(max_depth=3),
                           verbose=False)
        assert result in ('X', 'O', 'Draw')


# ── runner ─────────────────────────────────────────────────

if __name__ == '__main__':
    suites = [
        TestTicTacToe, TestMinimax, TestAlphaBeta,
        TestHeuristicAlphaBeta, TestMCTS, TestIntegration
    ]

    total = passed = 0
    for suite_cls in suites:
        suite = suite_cls()
        print(f"\n{'─'*50}")
        print(f"  {suite_cls.__name__}")
        print(f"{'─'*50}")
        for name in dir(suite):
            if name.startswith('test_'):
                total += 1
                ok = run(name, getattr(suite, name))
                if ok:
                    passed += 1

    print(f"\n{'═'*50}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'═'*50}")
