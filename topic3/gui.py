from __future__ import annotations

from typing import Optional, Tuple

import pygame

from game_state import Board, EMPTY, PLAYER_O, PLAYER_X


WHITE = (250, 248, 243)
BLACK = (20, 20, 20)
LINE_COLOR = (30, 30, 30)
X_COLOR = (66, 86, 180)
O_COLOR = (203, 92, 92)


class GameGUI:
    def __init__(self, board: Board) -> None:
        pygame.init()
        self.board = board
        self.size = 360
        self.cell = self.size // 3
        self.screen = pygame.display.set_mode((self.size, self.size + 60))
        pygame.display.set_caption("Tic-Tac-Toe")
        self.font = pygame.font.SysFont("DejaVu Sans", 48)
        self.status_font = pygame.font.SysFont("DejaVu Sans", 22)
        self.status_text = ""

    def draw(self) -> None:
        self.screen.fill(WHITE)
        for i in range(1, 3):
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (i * self.cell, 0),
                (i * self.cell, self.size),
                4,
            )
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (0, i * self.cell),
                (self.size, i * self.cell),
                4,
            )

        for r in range(3):
            for c in range(3):
                val = self.board.grid[r][c]
                if val == EMPTY:
                    continue
                color = X_COLOR if val == PLAYER_X else O_COLOR
                text = self.font.render(val, True, color)
                rect = text.get_rect(center=(c * self.cell + self.cell // 2, r * self.cell + self.cell // 2))
                self.screen.blit(text, rect)

        status_bg = pygame.Rect(0, self.size, self.size, 60)
        pygame.draw.rect(self.screen, BLACK, status_bg)
        status = self.status_font.render(self.status_text, True, WHITE)
        status_rect = status.get_rect(center=(self.size // 2, self.size + 30))
        self.screen.blit(status, status_rect)
        pygame.display.flip()

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        if y >= self.size:
            return None
        row = y // self.cell
        col = x // self.cell
        if row < 0 or row > 2 or col < 0 or col > 2:
            return None
        return row, col

    def set_status(self, text: str) -> None:
        self.status_text = text

