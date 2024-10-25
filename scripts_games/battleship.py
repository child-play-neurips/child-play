from typing import List, Tuple, Callable
import random

class BattleShip:
    def __init__(self, options=None):
        self.initialize_game(options)

    def initialize_game(self, options=None):
        if options is None:
            self.board_size = 5
            options = {}
        else:
            self.board_size = options.get("board_size", 5)
            self.debug = options.get("debug", False)
            
        self.name = "battleship"
        self.ship_board_p1 = [["~" for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.guess_board_p1 = [["~" for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.ship_board_p2 = [["~" for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.guess_board_p2 = [["~" for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.ships_p1 = []
        self.ships_p2 = []
        self.ship_types = {
            'Cruiser': 2,
            'Destroyer': 1
        }
        max_ship_size = self.board_size
        for i in range(max_ship_size, 0, -1):
            self.ship_types[f'Ship_{i}'] = i
        self.place_ships(self.ship_board_p1, self.ships_p1)
        self.place_ships(self.ship_board_p2, self.ships_p2)
        self.game_over = False
        self.current_player = "P1"
        self.prompt = f"Battleship is a two-player guessing game where each player has a fleet of ships on a secret grid and then takes turns guessing the locations of the opponent's ships. The objective is to sink all of the opponent's ships by correctly guessing their locations. O's in a board mean that the player selected a square to attack and there was no ship there - it's a miss. Had there been a ship there, instead of a O you would see an X. In your board, an <S> signifies a ship position, and a <~> signifies sea. Your input is just two numbers with a space in between, one for the row (from 0 to {self.board_size-1}) and one for the column (from 0 to {self.board_size-1}), like: 0 0, nothing else. Do not output anything else but the row col values."

    @property
    def board(self):
        return self.guess_board_p1 if self.current_player == "P1" else self.guess_board_p2
    
    def reset_board(self):
        """Resets the game to its initial state."""
        self.initialize_game()

    def place_ships(self, ship_board: List[List[str]], ships_list: List[Tuple[int, int]]) -> None:
        """
        Places ships on the board for a player.

        Parameters:
            ship_board (List[List[str]]): The player's ship board.
            ships_list (List[Tuple[int, int]]): List to store ship positions.

        Returns:
            None
        """
        for ship, size in self.ship_types.items():
            placed = False
            attempts = 0
            max_attempts = 5
            while not placed and attempts < max_attempts:
                attempts += 1
                orientation = random.choice(['H', 'V'])
                if orientation == 'H':
                    row = random.randint(0, self.board_size - 1)
                    col = random.randint(0, max(self.board_size - size, 0))  # Prevent negative range
                else:
                    row = random.randint(0, max(self.board_size - size, 0))  # Prevent negative range
                    col = random.randint(0, self.board_size - 1)
                if self.is_space_free(ship_board, row, col, size, orientation):
                    for i in range(size):
                        r, c = (row, col + i) if orientation == 'H' else (row + i, col)
                        ship_board[r][c] = 'S'
                        ships_list.append((r, c))
                    placed = True

    def is_space_free(self, ship_board: List[List[str]], row: int, col: int, size: int, orientation: str) -> bool:
        """
        Checks if space is free to place a ship.

        Parameters:
            ship_board (List[List[str]]): The player's ship board.
            row (int): The starting row index.
            col (int): The starting column index.
            size (int): The size of the ship.
            orientation (str): The orientation of the ship ('H' or 'V').

        Returns:
            bool: True if space is free, False otherwise.
        """
        for i in range(size):
            r, c = (row, col + i) if orientation == 'H' else (row + i, col)
            if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size or ship_board[r][c] == 'S':
                return False
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and ship_board[nr][nc] == 'S':
                        return False
        return True

    def get_text_state(self, player=0) -> None:
        """
        Prints both player's boards.

        Parameters:
            player (int): The current player (1 or 2).

        Returns:
            None
        """
        out = ""

        if player == 0:
            own_board, guess_board = self.ship_board_p1, self.guess_board_p1
        else:
            own_board, guess_board = self.ship_board_p2, self.guess_board_p2

        out += "Your Ships:" + " " * (self.board_size - 2) + "Opponent's Board:" + "\n"
        out += "  " + " ".join(str(i) for i in range(self.board_size)) + "   " + "  " + " ".join(str(i) for i in range(self.board_size)) + "\n"
        for i in range(self.board_size):
            out += f"{i} {' '.join(own_board[i])} | {i} {' '.join(guess_board[i])}" + "\n"

        return out

    def guess(self, player: int, guess: Tuple[int, int], playerobj) -> bool:
        row, col = guess

        guess_board = self.guess_board_p1 if player == 0 else self.guess_board_p2
        target_board = self.ship_board_p2 if player == 0 else self.ship_board_p1

        hit_marker = "X" if self.current_player == "P1" else "X"
        miss_marker = "O" if self.current_player == "P1" else "O"

        if target_board[row][col] == 'S':
            guess_board[row][col] = hit_marker
            target_board[row][col] = hit_marker
            if self.check_win():
                self.game_over = True
                return "Win", True
        else:
            guess_board[row][col] = miss_marker
            
        self.switch_player()
        return "Valid move", True
    
    def check_loss(self) -> bool:
        return False

    def check_tie(self) -> bool:
        return False

    def check_win(self) -> bool:
        if len(self.ships_p2) == sum(1 for (r, c) in self.ships_p2 if self.guess_board_p1[r][c] == 'X'):
            return True
        elif len(self.ships_p1) == sum(1 for (r, c) in self.ships_p1 if self.guess_board_p2[r][c] == 'X'):
            return True
        else:
            return False
        
    def switch_player(self):
        """Switches the current player."""
        self.current_player = "P2" if self.current_player == "P1" else "P1"