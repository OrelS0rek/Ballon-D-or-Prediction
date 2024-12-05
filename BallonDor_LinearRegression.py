import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
print("\n\nstatistics for data:\n",df[features].describe)

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

def gls_barplot():
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
def corr_plot():
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
    
    return correlation_with_target_sorted


def pairplots():
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
        weights_gradient += -(2/n)*x*(y-(x*current_weights+current_bias))
        bias_gradient += -(2/n)*(y-(x*current_weights+current_bias))
    
    #applying gradient descent
    new_weights = current_weights-learning_rate*weights_gradient
    new_bias = current_bias-learning_rate*bias_gradient

    #returning the new weights and biases for the model
    return new_weights,new_bias


def gradient_descent_vectorized(X, y, weights, bias, learning_rate):
    n = len(y)

    # Calculate predictions
    predictions = X.dot(weights) + bias
    
    # Calculate residuals
    residuals = y - predictions
    
    # Compute gradients
    weights_gradient = -(2/n) * X.T.dot(residuals)
    bias_gradient = -(2/n) * residuals.sum()
    
    # Update weights and bias
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient
    
    return weights, bias



def train_model(chosen_features, points, learning_rate, epochs):
    weight = random.uniform(-1, 1)
    bias = random.uniform(-1,1)
    for epoch in range(epochs):
        weight,bias = gradient_descent(chosen_features, weight, bias, points, learning_rate)
        mse = mean_squared_error(chosen_features, weight, bias, points)
        print(f'iteration: {epoch}\n\nError: {mse}')
            
    return weight,bias

def fit(X, y, learning_rate, epochs):
    import numpy as np
    #i know i said it does not use libraries. i used numpy because not using numpy 
    #means doing all vector operations using for loops. with each loop iterating through
    #the whole dataframe which is sized (125687, 20), so it took hours to run.
    #to speed unefficiency i used numpy but the model works without numpy by using the
    #gradient_descent function and train_model() instead of fit()
    weight = np.random.uniform(-1, 1, size=X.shape[1])
    bias = random.uniform(-1,1)
    for epoch in range(epochs):
        weight, bias = gradient_descent_vectorized(X, y, weight, bias, learning_rate)
        print(f'iteration: {epoch}')
    return weight, bias

def find_k_nearest_neighbors(data, sample, k):
    distances = [(x, abs(x - sample)) for x in data if isinstance(x, (int, float))]
    distances.sort(key=lambda x: x[1])
    return distances[:k]  # Return top 5 nearest neighbors

def generate_synthetic_sample(sample, neighbors): 
    synthetic_values = []
    for neighbor in neighbors:
        diff = neighbor[0] - sample  # Find the difference between neighbor and sample
        random_factor = random.uniform(0, 1)  # Random factor between 0 and 1
        synthetic_value = sample + diff * random_factor  # Generate synthetic value
        if synthetic_value > 0:  # Ensure it stays within the minority class range
            synthetic_values.append(synthetic_value)
    return synthetic_values

def apply_smote(data, target_minority_size):
    # Select the minority class (entries with 'Percent' > 0)
    minority_class = data[data['Percent'] > 0]
    
    current_minority_size = len(minority_class)
    if current_minority_size >= target_minority_size:
        return minority_class  # No need to generate more samples
    
    synthetic_samples = []
    
    while len(synthetic_samples) + current_minority_size < target_minority_size:
        for _, sample_row in minority_class.iterrows():
            sample = sample_row['Percent']  # Extract Percent value
            neighbors = find_k_nearest_neighbors(minority_class['Percent'], sample, 5)  # Use only Percent column
            synthetic_values = generate_synthetic_sample(sample, neighbors)  # Generate synthetic values
            
            synthetic_samples.extend(synthetic_values)
            if len(synthetic_samples) + current_minority_size >= target_minority_size:
                break  # Stop when the target size is reached
    
    # Limit the synthetic samples to exactly match the target size
    synthetic_samples = synthetic_samples[:target_minority_size - current_minority_size]
    
    # Create a DataFrame with synthetic samples
    synthetic_df = minority_class.sample(n=len(synthetic_samples), replace=True).copy()
    synthetic_df['Percent'] = synthetic_samples  # Replace Percent values with synthetic values
    
    # Combine the original minority class with synthetic samples
    return pd.concat([minority_class, synthetic_df], ignore_index=True)

def predict(player_name, test_data, weights, bias):
    player_stats = test_data[test_data['Player']==player_name]
    if player_stats.empty:
        return f"Player {player_name} not found in test Data."
    features = player_stats[numeric_stats].iloc[0]
    prediction = (features*weights).sum() + bias
    return prediction

def predict_all(test_data, weights, bias):
    predictions = []
    for _, row in test_data.iterrows():
        player_name = row['Player']
        prediction = predict(player_name, test_data, weights, bias)
        predictions.append(prediction)

    sorted_data = test_data.copy()
    sorted_data['Predicted Percent'] = predictions
    if sorted_data['Predicted Percent'].isnull().any():
        print("Warning: Found NaN values in 'Predicted Percent'. filling invalid rows with 0.")
        sorted_data = sorted_data.fillna(0)

    # Ensure the column is numeric
    sorted_data['Predicted Percent'] = pd.to_numeric(sorted_data['Predicted Percent'], errors='coerce')

    # fill invalid zero values
    sorted_data = sorted_data.fillna(0)

    # Sort by 'Predicted Percent'
    return sorted_data[['Player','Predicted Percent']]

def predict_player_by_name():
    player_name = input("enter player to predict: ")
    predict(player_name)
    
    
    
def min_max_scaler(data, columns):
     scaled_data = data.copy()
     for column in columns:
         minvalue = data[column].min()
         maxvalue = data[column].max()
         scaled_data[column] = (data[column]-minvalue)/(maxvalue-minvalue)
     return scaled_data



statistics =  full_stats.describe()


correlation = corr_plot()



numeric_stats = ['CL_Gls','CL_G+A','CL_G-PK','CL_PKatt','CL_PK',
                 'CL_Ast','Gls','G+A','G-PK','CL_Starts','PKatt',
                 'PK','CL_MP','Ast','cl_winner','WC_Gls','WC_G-PK',
                 'cl_finalist','WC_G+A','Top-Scorer','Percent']
#applying min-max scaling to the data
full_stats = min_max_scaler(full_stats, numeric_stats)
full_stats = full_stats.fillna(0)
numeric_data = full_stats[numeric_stats]
numeric_data = numeric_data.apply(pd.to_numeric, errors='coerce')
numeric_data = numeric_data.dropna()
numeric_data = min_max_scaler(numeric_data, numeric_stats)

target_minority_size = 64000

balanced_minority_data = apply_smote(numeric_data, target_minority_size)

# Inspect the resulting dataset
print("Size of balanced minority class dataset:", len(balanced_minority_data))


#spliting training vs testing data so that:
    #stats before 2024 are the training data
    #stats after 2024 are the testing data, meaning we would try to predict the 2024
train_data = full_stats[full_stats['Year']<2024]
test_data = full_stats[full_stats['Year']==2024]
train_data_num = train_data[numeric_stats]
full_training_data = pd.concat([train_data_num, balanced_minority_data])
y = full_training_data['Percent']
X = full_training_data[['CL_Gls','CL_G+A','CL_G-PK','CL_PKatt','CL_PK',
                 'CL_Ast','Gls','G+A','G-PK','CL_Starts','PKatt',
                 'PK','CL_MP','Ast','cl_winner','WC_Gls','WC_G-PK',
                 'cl_finalist','WC_G+A','Top-Scorer']]

learning_rate = 0.01
epochs = 10
#w,b = train_model(numeric_stats, full_training_data, learning_rate, epochs)
w, b = fit(X, y, learning_rate, epochs)
predictions = predict_all(test_data, w, b)
nominees = predictions[predictions['Predicted Percent']>0]
predictions.to_csv(r"D:\Machine Learning Projects\Ballon Dor Rankings Prediction\results\predictions.csv")
nominees.to_csv(r"D:\Machine Learning Projects\Ballon Dor Rankings Prediction\results\nominnes.csv")
