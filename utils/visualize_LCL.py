import matplotlib.pyplot as plt
import matplotlib.patches as patches

class LCLVisualizer:
    def __init__(self, output_path='lcl_construct_axes.png'):
        self.fig, self.ax = plt.subplots()
        self.output_path = output_path

    def draw_piece(self, position, color='blue'):
        """
        Draws a 2x4 LEGO piece at the given position.

        Parameters:
        - position: A tuple (x, y) for the position of the bottom left stud.
        - color: A string indicating the color of the piece.
        """
        # Drawing a 2x4 LEGO piece, considering the width as 4 units and height as 2 units.
        self.ax.add_patch(patches.Rectangle(position, 4, 1, edgecolor='black', facecolor=color))

    def display_construct(self, pieces):
        """
        Creates and saves the construct based on the list of pieces, with dynamic axis ranges.

        Parameters:
        - pieces: A list of tuples, each representing a piece. A tuple contains the position (x, y) and the color of the piece.
        """
        for piece in pieces:
            self.draw_piece((piece[0], piece[1]), piece[2])
        
        # Calculate dynamic ranges for x and y axes based on pieces' positions
        min_x = min(p[0] for p in pieces) - 0.5
        max_x = max(p[0] for p in pieces) + 5.5  # Account for piece width
        min_y = min(p[1] for p in pieces)
        max_y = max(p[1] for p in pieces) + 1  # Account for piece height

        # Adjust plot limits to dynamically fit all pieces
        self.ax.set_xticks([x - 0.5 for x in range(int(min_x) + 1, int(max_x) + 1)])
        self.ax.set_ylim(bottom=min_y, top=max_y)

        # Adjust ticks
        self.ax.set_xticklabels([int(x) for x in range(int(min_x) + 1, int(max_x) + 1)])
        self.ax.set_yticks(range(min_y, max_y + 1))

        # Retain other settings for aspect ratio, grid, and axis labels
        self.ax.set_aspect('equal', 'box')
        self.ax.axis('on')
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.set_xlabel('Studs')
        self.ax.set_ylabel('Layers')

        # Save the figure to a file
        plt.savefig(self.output_path, bbox_inches='tight')
        plt.close()

# Example Usage
visualizer = LCLVisualizer('visualize_construct_with_axes.png')
pieces = [
    (0, 0, 'blue'),
    (4, 0, 'yellow'),
    (8, 0, 'red'),
    (2, 1, 'green'),
    (6, 1, 'orange'),
    (4, 2, 'pink')

]

visualizer.display_construct(pieces)