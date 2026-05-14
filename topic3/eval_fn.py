from __future__ import annotations

from typing import List

from game_state import Board, EMPTY, PLAYER_O, PLAYER_X


WIN_SCORE = 100
TWO_IN_ROW = 10
ONE_IN_ROW = 1


def evaluate(board: Board, max_player: str) -> int:
    winner = board.winner()
    if winner == max_player:
        return WIN_SCORE
    if winner is not None and winner != max_player:
        return -WIN_SCORE
    if board.is_terminal():
        return 0

    score = 0
    for line in board.as_lines():
        score += line_score(line, max_player)
    return score


def line_score(line: List[str], max_player: str) -> int:
    opp = PLAYER_O if max_player == PLAYER_X else PLAYER_X
    max_count = line.count(max_player)
    opp_count = line.count(opp)
    empty_count = line.count(EMPTY)

    if max_count > 0 and opp_count > 0:
        return 0
    if max_count == 2 and empty_count == 1:
        return TWO_IN_ROW
    if max_count == 1 and empty_count == 2:
        return ONE_IN_ROW
    if opp_count == 2 and empty_count == 1:
        return -TWO_IN_ROW
    if opp_count == 1 and empty_count == 2:
        return -ONE_IN_ROW
    return 0

