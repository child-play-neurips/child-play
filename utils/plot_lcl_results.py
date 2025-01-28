import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.weight'] = 'bold'

def plot_proportions(df_validity, df_construct, plot_type='bar'):
    # Aggregating statistics for validity data
    stats_validity = df_validity.groupby(['Temperature', 'Model']).agg(
        Correct_Mean=('Correct', 'mean'),
        Correct_Std=('Correct', 'std'),
        N=('Correct', 'count')
    ).reset_index()
    stats_validity['Correct_Proportion'] = stats_validity['Correct_Mean'] * 100  # Convert to percentage
    stats_validity['Correct_SE'] = (stats_validity['Correct_Std'] / np.sqrt(stats_validity['N'])) * 100  # Convert to percentage

    # Aggregating statistics for construct data
    stats_construct = df_construct.groupby(['Temperature', 'Model']).agg(
        Valid_Mean=('Valid', 'mean'),
        Valid_Std=('Valid', 'std'),
        N=('Valid', 'count')
    ).reset_index() 
    stats_construct['Valid_Proportion'] = stats_construct['Valid_Mean'] * 100  # Convert to percentage
    stats_construct['Valid_SE'] = (stats_construct['Valid_Std'] / np.sqrt(stats_construct['N'])) * 100  # Convert to percentage

    # Print the aggregated statistics
    print("=== Correct Responses ===")
    for temp in stats_validity['Temperature'].unique():
        print(f"\nTemperature: {temp}")
        temp_data = stats_validity[stats_validity['Temperature'] == temp]
        for _, row in temp_data.iterrows():
            print(f"  Model: {row['Model']}")
            print(f"    Mean Correct: {row['Correct_Mean']:.2f}")
            print(f"    Correct Proportion (%): {row['Correct_Proportion']:.2f}")
            print(f"    Standard Error (%): {row['Correct_SE']:.2f}")
            print(f"    Sample Size: {row['N']}")
    
    print("\n=== Valid Constructs ===")
    for temp in stats_construct['Temperature'].unique():
        print(f"\nTemperature: {temp}")
        temp_data = stats_construct[stats_construct['Temperature'] == temp]
        for _, row in temp_data.iterrows():
            print(f"  Model: {row['Model']}")
            print(f"    Mean Valid: {row['Valid_Mean']:.2f}")
            print(f"    Valid Proportion (%): {row['Valid_Proportion']:.2f}")
            print(f"    Standard Error (%): {row['Valid_SE']:.2f}")
            print(f"    Sample Size: {row['N']}")

    if plot_type == 'bar':
        plt.figure(figsize=(12, 6))

        # Preparing for plotting with updated model set
        models = stats_validity['Model'].unique()
        temperatures = stats_validity['Temperature'].unique()
        model_index = {model: i for i, model in enumerate(models)}

        bar_width = 0.2
        temp_position = np.arange(len(temperatures))

        # Plot for Percentage of Correct Responses
        plt.subplot(1, 2, 1)
        for model in models:
            model_data = stats_validity[stats_validity['Model'] == model]
            positions = temp_position + bar_width * model_index[model]
            plt.bar(positions, model_data['Correct_Proportion'], width=bar_width, label=model,
                    yerr=model_data['Correct_SE'], capsize=5, error_kw={'capthick': 2})

        plt.xticks(temp_position + bar_width * (len(models) - 1) / 2, temperatures, weight='bold')
        plt.title('Percentage of Correct Responses', fontsize=20, weight='bold')
        plt.ylabel('Percentage Correct (%)', fontsize=20, weight='bold')
        plt.xlabel('Temperature', fontsize=20, weight='bold')
        plt.ylim(0, 100)

        # Plot for Percentage of Valid Constructs
        plt.subplot(1, 2, 2)
        for model in models:
            model_data = stats_construct[stats_construct['Model'] == model]
            positions = temp_position + bar_width * model_index[model]
            plt.bar(positions, model_data['Valid_Proportion'], width=bar_width, label=model,
                    yerr=model_data['Valid_SE'], capsize=5, error_kw={'capthick': 2})

        plt.xticks(temp_position + bar_width * (len(models) - 1) / 2, temperatures, weight='bold')
        plt.title('Percentage of Valid Constructs', fontsize=20, weight='bold')
        plt.ylabel('Percentage Valid (%)', fontsize=20, weight='bold')
        plt.xlabel('Temperature', fontsize=20, weight='bold')
        plt.ylim(0, 100)

        plt.legend()
        plt.tight_layout()
        plt.show()

    elif plot_type == 'scatter':
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        handles, labels = None, None

        # Scatter plot for Percentage of Correct Responses
        ax1 = axes[0]
        for model in stats_validity['Model'].unique():
            model_data = stats_validity[stats_validity['Model'] == model]
            ax1.errorbar(
                model_data['Temperature'],
                model_data['Correct_Proportion'],
                yerr=model_data['Correct_SE'],
                label=model,
                fmt='o-',
                capsize=5
            )
        ax1.set_title('Percentage of Correct Responses', fontsize=20, weight='bold')
        ax1.set_ylabel('Percentage Correct (%)', fontsize=16, weight='bold')
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('Temperature', fontsize=16, weight='bold')

        # Scatter plot for Percentage of Valid Constructs
        ax2 = axes[1]
        for model in stats_construct['Model'].unique():
            model_data = stats_construct[stats_construct['Model'] == model]
            ax2.errorbar(
                model_data['Temperature'],
                model_data['Valid_Proportion'],
                yerr=model_data['Valid_SE'],
                label=model,
                fmt='o-',
                capsize=5
            )
        ax2.set_title('Percentage of Valid Constructs', fontsize=20, weight='bold')
        ax2.set_ylabel('Percentage Valid (%)', fontsize=16, weight='bold')
        ax2.set_ylim(0, 100)
        ax2.set_xlabel('Temperature', fontsize=16, weight='bold')

        # Collect handles and labels from the second subplot for legend placement
        handles, labels = ax2.get_legend_handles_labels()

        # Add the legend outside the top-most plot
        if handles and labels:  # Ensure handles and labels are not None
            ax1.legend(
                handles,
                labels,
                title='Player and Model',
                loc='upper left', 
                bbox_to_anchor=(1, 1),
                # ncol=3,
                # fancybox=True, 
                # shadow=True
            )
        plt.subplots_adjust(right=0.85, hspace=0.3)  # Adjust right to make space for legend
        plt.show()

def main():
    # Loading both the original and new experiment datasets
    data_validity = pd.concat([
        pd.read_csv("../lcl_experiments/df_validity.csv"),
        pd.read_csv("../lcl_experiments/df_validity_4o_experiments.csv")
    ])
    data_construct = pd.concat([
        pd.read_csv("../lcl_experiments/df_construct.csv"),
        pd.read_csv("../lcl_experiments/df_construct_4o_experiments.csv")
    ])

    df_validity = pd.DataFrame(data_validity)
    df_construct = pd.DataFrame(data_construct)

    plot_type = 'scatter'  # Change to 'bar' for bar plot
    plot_proportions(df_validity, df_construct, plot_type=plot_type)

if __name__ == "__main__":
    main()
