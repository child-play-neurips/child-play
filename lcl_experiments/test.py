import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast

class ConstructGame:
    def __init__(self):
        self.valid_colors = ['blue', 'yellow', 'red']
        self.pieces = []

    def random_or_predefined_assembly(self, n=5):
        if random.choice([True, False]):
            # Generate a random assembly with a high likelihood of overlap
            pieces = [(random.randint(0, 5) * 2, random.randint(0, 3), self.valid_colors[random.randint(0, len(self.valid_colors) - 1)]) for _ in range(n)]
            # Ensure overlap by duplicating coordinates
            if pieces:  # Check if pieces list is not empty
                overlap_index = random.randint(0, len(pieces) - 1)  # Adjust index to be within the length of pieces
                pieces.append(pieces[overlap_index])  # Duplicate one piece to guarantee overlap
            return pieces
        else:
            # Return another predefined assembly
            return [(0, 0, 'blue'), (2, 0, 'red'), (4, 1, 'green')]

    def draw_piece(self, ax, position, color='blue'):
        ax.add_patch(patches.Rectangle(position, 4, 1, edgecolor='black', facecolor=color))

    def display_construct(self, pieces, filename):
        fig, ax = plt.subplots()
        for piece in pieces:
            self.draw_piece(ax, (piece[0], piece[1]), piece[2])
        
        min_x = min(p[0] for p in pieces)
        max_x = max(p[0] for p in pieces) + 4  # Account for piece width
        min_y = min(p[1] for p in pieces)
        max_y = max(p[1] for p in pieces) + 1

        ax.set_xticks([x for x in range(int(min_x), int(max_x) + 1)])
        ax.set_xlim(left=min_x, right=max_x)
        ax.set_ylim(bottom=min_y, top=max_y)
        ax.set_xticklabels([x for x in range(int(min_x), int(max_x) + 1)])
        ax.set_yticks(range(min_y, max_y + 1))
        ax.set_aspect('equal', 'box')
        ax.axis('on')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Studs')
        ax.set_ylabel('Layers')
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    def is_valid_construct(self, pieces):
        occupied_positions = {}
        for x, y, color in pieces:
            # Create a set of x positions for each piece based on its width (4 studs)
            piece_positions = set(range(x, x + 4))
            if y in occupied_positions:
                # Check for any overlap in x positions with existing pieces at the same y level
                if any(pos in occupied_positions[y] for pos in piece_positions):
                    return False  # Overlap detected
                occupied_positions[y].update(piece_positions)
            else:
                # Initialize occupied positions for this y level
                occupied_positions[y] = set(piece_positions)
        return True

    def can_place_piece(self, x, y, existing_pieces):
        # Check for horizontal overlap
        for ex, ey, _ in existing_pieces:
            if ey == y and not set(range(x, x + 4)).isdisjoint(set(range(ex, ex + 4))):
                return False  # Overlap detected

        # Check for support: each piece above the first layer must be supported by at least one piece directly below it
        if y > 0:
            supported = False
            for ex, ey, _ in existing_pieces:
                if ey == y - 1 and (ex <= x < ex + 4 or ex < x + 4 <= ex + 4):
                    supported = True
                    break
            if not supported:
                return False  # No support found

        # Check for connectivity: pieces must connect via overlapping pegs
        connected = False
        for ex, ey, _ in existing_pieces:
            # Check for horizontal connectivity via overlapping pegs
            if (ey == y and (ex + 2 == x or ex - 2 == x)) or \
               (ey == y + 1 and (ex <= x + 3 and ex >= x - 3)) or \
               (ey == y - 1 and (ex <= x + 3 and ex >= x - 3)):
                connected = True
                break

        if not connected and y > 0:
            return False  # Not properly connected

        # Additional check for base layer: ensure no gaps in horizontal connection without vertical support
        if y == 0:
            left_neighbor = any(ex == x - 4 and ey == y for ex, ey, _ in existing_pieces)
            right_neighbor = any(ex == x + 4 and ey == y for ex, ey, _ in existing_pieces)
            if left_neighbor or right_neighbor:
                return False  # No connecting piece above to bridge the gap

        return True

    def find_all_valid_positions(self, existing_pieces):
        valid_positions = set()
        # Check around and directly above each existing piece for all possible peg connections
        for ex, ey, _ in existing_pieces:
            candidate_positions = [
                (ex - 4, ey), (ex + 4, ey), (ex, ey + 1),
                (ex - 2, ey), (ex + 2, ey), (ex - 2, ey + 1), (ex + 2, ey + 1)
            ]
            for x, y in candidate_positions:
                if self.can_place_piece(x, y, existing_pieces):
                    valid_positions.add((x, y))
        return list(valid_positions)

    def build_random_valid_assembly(self, num_pieces):
        if num_pieces < 1:
            return []

        # Start with the first piece at (0, 0)
        self.pieces = [(0, 0, random.choice(self.valid_colors))]

        # Try to place remaining pieces at valid positions
        for _ in range(1, num_pieces):
            valid_positions = self.find_all_valid_positions(self.pieces)
            if not valid_positions:
                print("No valid positions available, stopping early")
                break
            new_x, new_y = random.choice(valid_positions)
            new_color = random.choice(self.valid_colors)
            self.pieces.append((new_x, new_y, new_color))

        return self.pieces

def parse_response(response):
    # Parse the string response into a list of tuples
    try:
        return ast.literal_eval(response)
    except Exception as e:
        print(f"Error parsing response: {response}, error: {e}")
        return []

def reassess_validity(df, game):
    # Reassess the validity of constructs in the DataFrame
    for index, row in df.iterrows():
        pieces = parse_response(row['LLM Response'])
        valid = game.is_valid_construct(pieces)

        # Update the DataFrame with new validity information
        df.at[index, 'Valid'] = valid
        df.at[index, 'Actual Validity'] = valid  # Assuming df has this column in df_validity

        # Update the 'Correct' column based on player answer and validity
        if 'Player Answer' in df.columns:
            correct = (row['Player Answer'] == 'valid' and valid) or (row['Player Answer'] == 'invalid' and not valid)
            df.at[index, 'Correct'] = correct

def main():
    game = ConstructGame()
    
    # Load data (assuming CSV format, replace with actual loading method)
    df_validity = pd.read_csv('./lcl_experiments/df_validity.csv')
    df_construct = pd.read_csv('./lcl_experiments/df_construct.csv')

    # Reassess the validity using the ConstructGame class
    reassess_validity(df_validity, game)
    reassess_validity(df_construct, game)

    # Save the updated DataFrames back to CSV
    df_validity.to_csv('./lcl_experiments/updated_df_validity.csv', index=False)
    df_construct.to_csv('./lcl_experiments/updated_df_construct.csv', index=False)

    # Optionally, print some rows to check the outputs
    print(df_validity.head())
    print(df_construct.head())

if __name__ == '__main__':
    main()

# Example usage

# game = ConstructGame()
# pieces = game.build_random_valid_assembly(5)
# validity = game.is_valid_construct(pieces)
# game.display_construct(pieces, "./lcl_experiments/test.png")
# print("Generated pieces:", pieces)
# print("Validity of the construct:", validity)
# print("Random --------------------------v")
# random_pieces = game.random_or_predefined_assembly()
# validity_rand = game.is_valid_construct(random_pieces)
# game.display_construct(random_pieces, "./lcl_experiments/test_rand.png")
# print("Generated pieces:", random_pieces)
# print("Validity of the construct:", validity_rand)

