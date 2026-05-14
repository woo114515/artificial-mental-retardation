from __future__ import annotations

from typing import Dict, Optional, Tuple

from game_state import Board, PLAYER_O, PLAYER_X
from eval_fn import evaluate


def minimax(
    board: Board,
    player: str,
    depth: int,
    max_player: str,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[int, Optional[Tuple[int, int]]]:
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1

    if depth == 0 or board.is_terminal():
        return evaluate(board, max_player), None

    best_move: Optional[Tuple[int, int]] = None
    if player == max_player:
        best_score = -10_000
        for move in board.legal_moves():
            next_board = board.copy()
            next_board.apply_move(move, player)
            score, _ = minimax(next_board, _opponent(player), depth - 1, max_player, stats)
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

    best_score = 10_000
    for move in board.legal_moves():
        next_board = board.copy()
        next_board.apply_move(move, player)
        score, _ = minimax(next_board, _opponent(player), depth - 1, max_player, stats)
        if score < best_score:
            best_score = score
            best_move = move
    return best_score, best_move


def alphabeta(
    board: Board,
    player: str,
    depth: int,
    alpha: int,
    beta: int,
    max_player: str,
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[int, Optional[Tuple[int, int]]]:
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + 1

    if depth == 0 or board.is_terminal():
        return evaluate(board, max_player), None

    best_move: Optional[Tuple[int, int]] = None
    if player == max_player:
        value = -10_000
        for move in board.legal_moves():
            next_board = board.copy()
            next_board.apply_move(move, player)
            score, _ = alphabeta(
                next_board,
                _opponent(player),
                depth - 1,
                alpha,
                beta,
                max_player,
                stats,
            )
            if score > value:
                value = score
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                if stats is not None:
                    stats["cutoffs"] = stats.get("cutoffs", 0) + 1
                break
        return value, best_move

    value = 10_000
    for move in board.legal_moves():
        next_board = board.copy()
        next_board.apply_move(move, player)
        score, _ = alphabeta(
            next_board,
            _opponent(player),
            depth - 1,
            alpha,
            beta,
            max_player,
            stats,
        )
        if score < value:
            value = score
            best_move = move
        beta = min(beta, value)
        if beta <= alpha:
            if stats is not None:
                stats["cutoffs"] = stats.get("cutoffs", 0) + 1
            break
    return value, best_move


def best_move(
    board: Board,
    player: str,
    depth: int,
    use_ab: bool,
    stats: Optional[Dict[str, int]] = None,
) -> Optional[Tuple[int, int]]:
    if use_ab:
        _, move = alphabeta(board, player, depth, -10_000, 10_000, player, stats)
    else:
        _, move = minimax(board, player, depth, player, stats)
    return move


def _opponent(player: str) -> str:
    return PLAYER_O if player == PLAYER_X else PLAYER_X

