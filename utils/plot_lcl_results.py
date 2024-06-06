import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_proportions(df_validity, df_construct):
    # Calculate proportions of correct answers and valid constructs
    df_validity['Correct_Proportion'] = df_validity.groupby(['Temperature', 'Model'])['Correct'].transform('mean')
    df_construct['Valid_Proportion'] = df_construct.groupby(['Temperature', 'Model'])['Valid'].transform('mean')

    # Setting the plot size and style
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Plotting Correct Proportions from df_validity
    plt.subplot(1, 2, 1)
    sns.barplot(x='Temperature', y='Correct_Proportion', hue='Model', data=df_validity)
    plt.title('Proportion of Correct Responses')
    plt.ylabel('Proportion Correct')
    plt.xlabel('Temperature')

    # Plotting Valid Proportions from df_construct
    plt.subplot(1, 2, 2)
    sns.barplot(x='Temperature', y='Valid_Proportion', hue='Model', data=df_construct)
    plt.title('Proportion of Valid Constructs')
    plt.ylabel('Proportion Valid')
    plt.xlabel('Temperature')

    plt.tight_layout()
    plt.show()

def main():
    # Example data loading
    data_validity = pd.read_csv("./lcl_experiments/df_validity.csv")
    data_construct =  pd.read_csv("./lcl_experiments/df_construct.csv")

    df_validity = pd.DataFrame(data_validity)
    df_construct = pd.DataFrame(data_construct)

    plot_proportions(df_validity, df_construct)

if __name__ == "__main__":
    main()