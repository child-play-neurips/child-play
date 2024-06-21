class ConnectFour:
    def __init__(self, options=None):
        if options is None:
            self.rows = 7
            self.cols = 7
            options = {}
        else:
            self.debug = options.get("debug", False)
            self.rows = options.get("rows", 7)
            self.cols = options.get("cols", 7)
        
        self.name = "connectfour"
        self.reset_board()
        self.last_move = (-1, -1)
        self.game_over = False
        self.current_player = "P1"  # Assuming P1 starts the game
        self.prompt = "Connect-Four is a two-player game. The pieces fall straight down, occupying the next available space within a column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. In a board, player 1, you, plays with symbol X, while player 2, your opponent, plays with symbol O. Your input is just a number from 0 to 6, nothing else.  Do not output anything else but the col value else you lose."

    def reset_board(self):
        self.board = [["." for _ in range(self.cols)] for _ in range(self.rows)]

    def check_tie(self):
        return all(self.board[0][col] != '.' for col in range(self.cols)) and not self.check_win()

    def check_win(self):
        row, col = self.last_move
        if row == -1:
            return False
        player = self.board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                r, c = row, col
                while 0 <= r + d*dr < self.rows and 0 <= c + d*dc < self.cols and self.board[r + d*dr][c + d*dc] == player:
                    count += 1
                    r += d*dr
                    c += d*dc
                    if count >= 4:
                        return True
        return False

    def guess(self, player_index, guess, player):
        col = guess
        if col < 0 or col >= self.cols:
            return False
        for row in reversed(range(self.rows)):
            if self.board[row][col] == ".":
                self.board[row][col] = "X" if player_index == 0 else "O"
                self.last_move = (row, col)
                if self.check_win():
                    self.game_over = True
                    return "Win", True
                if self.check_tie():
                    self.game_over = True
                    return "Tie", True
                
                self.switch_player()
                return "Valid move", True
        return "Invalid move.", False
    
    def get_text_state(self, player_index=None):
        red = "\033[91mX\033[0m"
        yellow = "\033[32mO\033[0m"
        state_lines = [" 0 1 2 3 4 5 6"]
        for row in self.board:
            row_str = "|"
            for cell in row:
                if cell == "X":
                    row_str += red + "|"
                elif cell == "O":
                    row_str += yellow + "|"
                else:
                    row_str += cell + "|"
            state_lines.append(row_str)
        return "\n".join(state_lines)

    def switch_player(self):
        """Switches the current player."""
        self.current_player = "P2" if self.current_player == "P1" else "P1"

    @property
    def board_size(self):
        return self.cols  