import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
try:
    dataset_path = '../data/iris.csv'  # Relative path to the data folder
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Please ensure the file exists in the data folder.")
    exit()

# Display basic info
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean
grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)

# Save grouped data as a CSV
output_table_path = '../outputs/tables/grouped_means.csv'
grouped.to_csv(output_table_path)
print(f"\nGrouped means saved to '{output_table_path}'.")

# Create and save visualizations
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='sepal length (cm)', data=df, palette='viridis')
plt.title('Bar Chart: Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
output_figure_path = '../outputs/figures/sepal_length_bar_chart.png'
plt.savefig(output_figure_path)
print(f"Bar chart saved to '{output_figure_path}'.")
plt.show()
