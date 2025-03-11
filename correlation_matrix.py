import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#accesses file
file_path = 'data.csv'  
df = pd.read_csv(file_path)

# select columns
variable_columns = [col for col in df.columns if col.endswith('_Exam') or col.endswith('_Q1_Grade')]

# replace zeros with the mean of nonzero values
for col in variable_columns:
    mean_value = df[col][df[col] != 0].mean()
    df[col] = df[col].replace(0, mean_value)

# generates correlation matrix
df_scores = df[variable_columns]
corr_matrix = df_scores.corr()

# fixes missing values
corr_matrix = corr_matrix.apply(lambda x: x.fillna(x.mean()))
corr_matrix = corr_matrix.apply(lambda x: x.fillna(x.mean()), axis=1)

# Displays correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# plots heatmap
sns.set_context("notebook", font_scale=0.5)
plt.figure(figsize=(12, 8))
ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 5}, linewidths=0.5)
plt.subplots_adjust(left=0.3)
plt.xticks(rotation=45, ha="right", fontsize=6)
plt.yticks(fontsize=6)
plt.gcf().subplots_adjust(bottom=0.25)
plt.title("Correlation Matrix for AP Exams")

# saves heatmap
plt.savefig("heatmap.png", dpi=300)
plt.show()