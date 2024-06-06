import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Tic-Tac-Toe results (provided in your prompt)
data_tictactoe = {
    "Model": ["GPT-3.5"] * 4 + ["GPT-4"] * 4,
    "Temperature": [0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5],
    "Wins": [24, 24, 24, 19, 71, 77, 69, 50],
    "Wins (Random Player)": [76, 76, 76, 81, 25, 18, 28, 46],
    "Ties": [0, 0, 0, 0, 4, 5, 3, 4],
    "Incorrect Moves": [76, 76, 76, 81, 12, 11, 15, 36],
    "Legitimate Model Losses": [0, 0, 0, 0, 13, 7, 13, 10]
}

# Connect Four results (provided in your prompt)
data_connect_four = {
    "Model": ["GPT-3.5"] * 4 + ["GPT-4"] * 4,
    "Temperature": [0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5],
    "Wins": [76, 76, 75, 75, 80, 80, 70, 39],
    "Wins (Random Player)": [24, 24, 25, 25, 20, 20, 30, 61],
    "Incorrect Moves": [22, 20, 13, 12, 19, 20, 29, 60],
    "Legitimate Model Losses": [15, 4, 12, 13, 1, 0, 1, 1]
}

# Battleship results (provided in your prompt)
data_battleship = {
    "Model": ["GPT-3.5"] * 4 + ["GPT-4"] * 4,
    "Temperature": [0, 0.5, 1, 1.5, 0, 0.5, 1, 1.5],
    "Wins": [10, 8, 3, 0, 0, 0, 0, 0],
    "Wins (Random Player)": [90, 92, 97, 100, 100, 100, 100, 100],
    "Incorrect Moves": [86, 89, 93, 99, 100, 100, 100, 100]
}

# Calculate Legitimate Model Losses for Battleship
data_battleship["Legitimate Model Losses"] = [
    100 - wins - incorrect_moves 
    for wins, incorrect_moves in zip(data_battleship["Wins"], data_battleship["Incorrect Moves"])
]

# Create DataFrames
df_connect_four = pd.DataFrame(data_connect_four)
df_battleship = pd.DataFrame(data_battleship)

# Set global font properties
plt.rcParams.update({'font.size': 12, 'font.weight': 'bold'})

# Function to add labels to the bars
def add_labels(ax):
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f'{int(width)}', xy=(width, p.get_y() + p.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha='center', va='center')

# Initialize the matplotlib figure for each game and ensure settings
for df, title in [(data_tictactoe, "Tic-Tac-Toe"), (data_connect_four, "Connect Four"), (data_battleship, "Battleship")]:
    plt.figure(figsize=(14, 10))
    ax1 = plt.subplot(221)
    sns.barplot(x="Wins", y="Temperature", hue="Model", data=df, orient='h', errorbar='sd', ax=ax1)
    ax1.set_xlabel("Wins")
    ax1.set_ylabel("Temperature")
    ax1.legend(title='Model', title_fontsize='13', loc='lower right')
    plt.title(f"Wins ({title})")
    add_labels(ax1)

    ax2 = plt.subplot(222)
    sns.barplot(x="Wins (Random Player)", y="Temperature", hue="Model", data=df, orient='h', errorbar='sd', ax=ax2)
    ax2.set_xlabel("Wins by Random Player")
    ax2.legend().set_visible(False)
    plt.title(f"Wins by Random Player")
    add_labels(ax2)

    ax3 = plt.subplot(223)
    sns.barplot(x="Incorrect Moves", y="Temperature", hue="Model", data=df, orient='h', errorbar='sd', ax=ax3)
    ax3.set_xlabel("Incorrect Moves")
    ax3.legend().set_visible(False)
    ax3.set_ylabel("Temperature")
    plt.title(f"Incorrect Moves")
    add_labels(ax3)

    ax4 = plt.subplot(224)
    sns.barplot(x="Legitimate Model Losses", y="Temperature", hue="Model", data=df, orient='h', errorbar='sd', ax=ax4)
    ax4.set_xlabel("Legitimate Model Losses")
    ax4.legend().set_visible(False)
    plt.title(f"Legitimate Model Losses")
    add_labels(ax4)

    plt.tight_layout()
    plt.show()
