from policy_value_isomorph.tictactoe import TicTacToeState, check_winner, state_from_rows


def test_legal_move_generation_initial_state():
    s = TicTacToeState.initial()
    assert s.legal_moves() == list(range(9))


def test_terminal_state_detection_draw():
    s = state_from_rows(["XOX", "XXO", "OXO"], to_move=1)
    assert s.winner() is None
    assert s.is_terminal()
    assert s.legal_moves() == []


def test_winner_detection_row_column_diagonal():
    row_win = state_from_rows(["XXX", "O..", "O.."], to_move=-1)
    col_win = state_from_rows(["XO.", "XO.", "X.."], to_move=-1)
    diag_win = state_from_rows(["X.O", ".X.", "O.X"], to_move=-1)
    assert row_win.winner() == 1
    assert col_win.winner() == 1
    assert diag_win.winner() == 1


def test_check_winner_none_on_non_terminal_partial_position():
    board = (1, 0, -1, 0, 1, 0, -1, 0, 0)
    assert check_winner(board) is None
