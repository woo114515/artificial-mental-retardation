from __future__ import annotations

import time

import pygame

from ai_minimax import best_move
from game_state import Board, PLAYER_O, PLAYER_X
from gui import GameGUI


AI_PLAYER = PLAYER_O
HUMAN_PLAYER = PLAYER_X
SEARCH_DEPTH = 9
USE_ALPHA_BETA = True


def run_game() -> None:
    board = Board()
    gui = GameGUI(board)
    current = HUMAN_PLAYER
    clock = pygame.time.Clock()
    pending_ai = False
    last_stats = {"nodes": 0, "cutoffs": 0, "ms": 0}

    running = True
    while running:
        gui.set_status(_status_text(board, current, last_stats))
        gui.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    board.reset()
                    current = HUMAN_PLAYER
                    pending_ai = False
                    last_stats = {"nodes": 0, "cutoffs": 0, "ms": 0}
            elif event.type == pygame.MOUSEBUTTONDOWN and current == HUMAN_PLAYER:
                move = gui.handle_click(event.pos)
                if move and move in board.legal_moves():
                    board.apply_move(move, HUMAN_PLAYER)
                    current = AI_PLAYER
                    pending_ai = True

        if not board.is_terminal() and current == AI_PLAYER and pending_ai:
            stats = {"nodes": 0, "cutoffs": 0}
            start = time.perf_counter()
            move = best_move(board, AI_PLAYER, SEARCH_DEPTH, USE_ALPHA_BETA, stats)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            if move:
                board.apply_move(move, AI_PLAYER)
            current = HUMAN_PLAYER
            pending_ai = False
            stats["ms"] = elapsed_ms
            last_stats = stats

        if board.is_terminal():
            pending_ai = False

        clock.tick(60)

    pygame.quit()


def _status_text(board: Board, current: str, stats: dict) -> str:
    winner = board.winner()
    if winner:
        return f"Winner: {winner} | Press R to restart"
    if board.is_terminal():
        return "Draw | Press R to restart"
    if current == HUMAN_PLAYER:
        return "Your turn (X)"
    return f"AI thinking... nodes={stats.get('nodes', 0)} cutoffs={stats.get('cutoffs', 0)} time={stats.get('ms', 0)}ms"


if __name__ == "__main__":
    run_game()

