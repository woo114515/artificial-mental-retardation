from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


EMPTY = "."
PLAYER_X = "X"
PLAYER_O = "O"


@dataclass
class Board:
    grid: List[List[str]]

    def __init__(self) -> None:
        self.grid = [[EMPTY for _ in range(3)] for _ in range(3)]

    def copy(self) -> "Board":
        new_board = Board()
        new_board.grid = [row[:] for row in self.grid]
        return new_board

    def legal_moves(self) -> List[Tuple[int, int]]:
        moves: List[Tuple[int, int]] = []
        for r in range(3):
            for c in range(3):
                if self.grid[r][c] == EMPTY:
                    moves.append((r, c))
        return moves

    def apply_move(self, move: Tuple[int, int], player: str) -> None:
        r, c = move
        if self.grid[r][c] != EMPTY:
            raise ValueError("Cell already occupied")
        self.grid[r][c] = player

    def is_terminal(self) -> bool:
        return self.winner() is not None or not self.legal_moves()

    def winner(self) -> Optional[str]:
        lines = self.as_lines()
        for line in lines:
            if line[0] != EMPTY and line.count(line[0]) == 3:
                return line[0]
        return None

    def as_lines(self) -> List[List[str]]:
        lines: List[List[str]] = []
        # Rows
        lines.extend([row[:] for row in self.grid])
        # Columns
        for c in range(3):
            lines.append([self.grid[r][c] for r in range(3)])
        # Diagonals
        lines.append([self.grid[i][i] for i in range(3)])
        lines.append([self.grid[i][2 - i] for i in range(3)])
        return lines

    def reset(self) -> None:
        for r in range(3):
            for c in range(3):
                self.grid[r][c] = EMPTY

