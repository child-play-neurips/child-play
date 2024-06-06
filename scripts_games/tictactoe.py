from ast import List
import random

class TicTacToe:
    def __init__(self, options=None):
        if options is None:
            self.board_size = 3  # TicTacToe is always 3x3
            options = {}
        else:
            self.board_size = options.get("rows", 3)
            self.debug = options.get("debug", False)
        self.reset_game()
        self.name = "tictactoe"
        self.prompt = "Tic-Tac-Toe is a two-player game played on a 3x3 grid. Players take turns placing their mark, X or O, in an empty square. The first player to place three of their marks in a horizontal, vertical, or diagonal row wins the game. You will play as player 1, therefore you play with X while your adversary plays with the symbol O. Your input is then a number (from 0 to 2) for the row followed by a space and another number (from 0 to 2) for the column, nothing else. Do not output anything else but the row col values else you lose."

    def reset_game(self):
        self.board = [[" " for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_player = "P1"
        self.moves_made = []
        self.game_over = False

    def get_text_state(self, player_index=None):
        """Generates a textual representation of the game state."""
        out = f""
        # player = "Player 1" if player_index == 0 else "Player 2"
        # out = f"\n{player}'s turn (You are '{self.current_player}'):\n\n"
        
        column_headers = "    " + "   ".join(str(i) for i in range(self.board_size)) + "\n"
        divider = "  +" + "---+" * self.board_size + "\n"

        out += column_headers + divider
        
        for index, row in enumerate(self.board):
            row_str = f"{index} | " + " | ".join(row) + " |\n"
            out += row_str + divider
        
        return out

    def guess(self, index, guess, playerobj):
        """Processes a player's guess, treating it as a move in TicTacToe."""
        row, col = guess
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return "Invalid move. Out of board range.", False
        if self.board[row][col] != " ":
            return "Invalid move. Position already taken.", False
        
        self.board[row][col] = "X" if index == 0 else "O"

        if self.check_win():
            self.game_over = True
            return "Win", True
        if self.check_tie():
            self.game_over = True
            return "Tie", True
        self.switch_player()
        return "Valid move", True

    def check_win(self) -> bool:
        """Checks if the current player has won the game."""
        symbol = "X" if self.current_player == "P1" else "O"
        win_conditions = [
            [(i, j) for i in range(self.board_size)] for j in range(self.board_size)  # Vertical
        ] + [
            [(j, i) for i in range(self.board_size)] for j in range(self.board_size)  # Horizontal
        ] + [
            [(i, i) for i in range(self.board_size)],  # Diagonal \
            [(i, self.board_size - 1 - i) for i in range(self.board_size)]  # Diagonal /
        ]

        for condition in win_conditions:
            if all(self.board[row][col] == symbol for row, col in condition):
                return True
        return False

    def check_loss(self) -> bool:
        return False

    def check_tie(self) -> bool:
        """Checks if the game has ended in a tie."""
        return all(self.board[row][col] != " " for row in range(self.board_size) for col in range(self.board_size)) and not self.check_win()
    
    def switch_player(self):
        """Switches the current player."""
        self.current_player = "P2" if self.current_player == "P1" else "P1"
