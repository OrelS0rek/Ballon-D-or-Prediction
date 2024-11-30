import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random 

#saving the dataset Scraped in 'data_scraping.ipynb' as df.
df = pd.read_csv(r"D:\Machine Learning Projects\Ballon Dor Rankings Prediction\cleaned_full_stats.csv")


#checking for useless columns and dropping them
print("\n\ncolumns:\n", df.columns)
df.drop('Unnamed: 0',axis=1,inplace=True)

#saving the data we want to predict (2024 stats) as 'current_df'
current_df = df[df['Year']==2024]
print("\n\ndata we want to predict (2024):\n", current_df)

#finding the right features for the model to use.
features = ['Player', 'Pos', 'Squad', 'Comp', 'Age',
        'MP', 'Starts', 'Min_x', '90s', 'Gls', 'Ast', 'G+A', 'G-PK',
       'PK', 'PKatt', 'CrdY', 'CrdR', 'Year','CL_Gls',
       'CL_Ast', 'CL_G+A', 'CL_G-PK', 'CL_PK', 'CL_MP', 'CL_Starts',
       'CL_PKatt', 'CL_CrdR', 'CL_CrdY', 'cl_finalist', 'cl_winner', 'LgRk',
       'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts', 'Pts/MP', 'Sqd_MP', 'Sqd_Gls',
       'Sqd_Ast', 'Sqd_G+A', 'wc_finalist', 'wc_winner',
       'Nation', 'WC_Starts', 'WC_Gls', 'WC_Ast', 'WC_G+A', 'WC_G-PK', 'WC_PK',
       'WC_PKatt', 'Top-Scorer']

#analyzing the data

print("data info:\n")
print(df[features].info())
print("\n\nstatics for data:\n",df[features].describe)

winners =  pd.read_csv(r"D:\Machine Learning Projects\Ballon Dor Rankings Prediction\winners.csv")
winners = winners[['Player','Year','Percent']]

#adding a collumn nominee for all ballon d'or nominees
winners['Nominees'] = 1
#adding the ballon dor nominees to the dataset.
full_stats = df.merge(
    winners,
    on= ['Player','Year'],
    how='left'
    )
full_stats['Percent'] = full_stats['Percent'].str.rstrip('%').astype(float) / 100

#creating a plot to see the correlation of 'Gls' and the target value 'Percent'.
plt.figure(figsize = (10,6))
plt.xticks(rotation=90, ha='right')
sns.barplot(x='Gls', y='Percent', data=full_stats,palette='tab10');


target_param = 'Percent'
selected_features = ['MP', 'Starts', 'Min_x', '90s', 'Gls',
      'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR', 'CL_Gls', 'CL_Ast',
      'CL_G+A', 'CL_G-PK', 'CL_PK', 'CL_MP', 'CL_Starts', 'CL_PKatt',
      'CL_CrdR', 'CL_CrdY', 'cl_finalist', 'cl_winner', 'W', 'D', 'L', 'GF',
      'GA', 'GD', 'Pts', 'Pts/MP', 'Sqd_MP', 'Sqd_Gls', 'Sqd_Ast', 'Sqd_G+A',
      'wc_finalist', 'wc_winner', 'WC_Starts', 'WC_Gls', 'WC_Ast',
      'WC_G+A', 'WC_G-PK', 'WC_PK', 'WC_PKatt', 'Top-Scorer', 'LgRk', 'Age']  # List of features to include

# Compute correlations with the target parameter for selected features
subset_df = full_stats[selected_features + [target_param]]

# Compute the correlation matrix for the subset
correlation_matrix = subset_df.corr()

# Select only correlations involving the target parameter
correlation_with_target = correlation_matrix[[target_param]].drop(target_param)
correlation_with_target_sorted = correlation_with_target.sort_values(by=target_param, ascending=False)
# Plot the heatmap
plt.figure(figsize=(6, 6))
sns.heatmap(correlation_with_target_sorted, annot=True, cmap='coolwarm', cbar=True, fmt='.2f', linewidths=0.5)
plt.title(f'Correlation of Selected Features with {target_param}', fontsize=14)
plt.ylabel('Features', fontsize=12)
plt.xlabel(f'Correlation with {target_param}', fontsize=12)
plt.tight_layout()
plt.show()


#scatter plot 'G+A' to 'Percent'
plt.scatter(full_stats['G+A'], full_stats['Percent'], alpha=0.5)
plt.title('G+A vs Percent')
plt.xlabel('G+A')
plt.ylabel('Percent')
plt.show()



#pariplots
sns.pairplot(df[selected_features], diag_kind='kde')
plt.suptitle('Pairplot of Selected Features')
plt.show()



#implementing linear regression
def mean_squared_error(chosen_features, weights, bias, points):
    #calculating the sum of the squared differences between the points and the slope
    total_error = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i][chosen_features]
        y = points.iloc[i]['Percent']
        total_error += (y-(weights*x+bias))**2
    #returning the mean
    return total_error/float(len(points))

def gradient_descent(chosen_features, current_weights, current_bias,points, learning_rate):
    n = len(points)
    weights_gradient = 0
    bias_gradient = 0
    
    for i in range(n):
        x = points.iloc[i][chosen_features]
        y = points.iloc[i]['Percent']
        #adding for each the gradient we calculated with partial derivatives, math would be joined in readMe file
        weights_gradient = -(2/n)*x*(y-(x*current_weights+current_bias))
        bias_gradient = -(2/n)*(y-(x*current_weights+current_bias))
    
    #applying gradient descent
    new_weights = current_weights-learning_rate*weights_gradient
    new_bias = current_bias-learning_rate*bias_gradient
    
    #returning the new weights and biases for the model
    return new_weights,new_bias

def train_model(chosen_features, points, learning_rate, epochs):
    weight = random.uniform(-1, 1)
    bias = random.uniform(-1,1)
    
    for epoch in range(epochs):
        weight,bias = gradient_descent(chosen_features, weight, bias, points, learning_rate)
        if (epoch%50==0):
            mse = mean_squared_error(chosen_features, weight, bias, points)
            print(f'iteration: {epoch}\n\nError: {mse}')
            
    return weight,bias
