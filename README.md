# Ballon-D-or-Prediction

*Objective*

trying to Predict the Winner of the Ballon D'Or with a machine learning model using NO libraries, and purely math, numpy and pandas.
my main goals with this project are predicting the ballon d'or winners. and learning about how linear regression works without using  a black box like sklearn that uses the algorithms without teaching me.

*Data Selection*


the full data will be available on kaggle soon.
i scraped the data from multiple sites containing data for :
* regurlar player stats, champions league player stats and world cup player stats
* ballon d'or winners and nominees and percent of votes each year.
here are the links of all websites i used:



*Exploratory Data Analysis*

for this section i had to use one library - matplotlib, because i cant compute plots without using libraries.
i plotted some scatter plots to help me visualize the data and some correlation plots.
all plots are available in the 'Plots' directory of the repository.

*The Project*

i started by scraping the statistics from the websites. id say that was most of the proccess of the project because it took time
to find good data and scrape it and handle the data such that i would have a compatible dataset which i can work with.
after changing the data a bit, i started to build the model for the machine learning and learned how linear regression works and the
math behind it. 

for the function gradient_descent i used the partial derivative with respect to b and w of the MSE (Mean Squared Error) function:

### Mean Squared Error
The formula for Mean Squared Error (MSE) is:

$$\
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\$$

Expanding $\(\hat{y}_i\)$ as $\(w \cdot x_i + b\)$, we get:

$$\
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b))^2
\$$

### Gradients for Gradient Descent
#### Gradient with respect to \(w\):
$$\
\frac{\partial MSE}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b)) \cdot x_i
\$$

#### Gradient with respect to \(b\):
$$\
\frac{\partial MSE}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b))
\$$

### Gradient Descent Update Rules
Update the parameters \(w\) and \(b\) using the following rules:

$$\
w \leftarrow w - \alpha \cdot \frac{\partial MSE}{\partial w}
\$$

$$\
b \leftarrow b - \alpha \cdot \frac{\partial MSE}{\partial b}
\$$

where $\(\alpha\)$ is the learning rate.

all of those mathematical equations are written in the gradient_descent functions and the mean_squared_error function.
after understanding the concepts and implementing them into the code. i wrote some functions to visualize the prediction and to actually use the weights and biases that are returned from gradient_descent, and got a prediction dataset in which the predicted Share for the players was 0 for everyone - i tried to debug the issue and found that out of my database of 64000 players, only 600 had a share of over 0 and 63000 player had a share of 0. so my database was VERY heavily imbalanced. 

i searched for techniques to balance a dataset without deleting majority class data because it is valueable data .
i found a technique called SMOTE - syntethic minority oversampling technique.
the algorithm takes a sample from the minority class and finds its k-nearest-neighbors using distance, 
then multiplying the difference between the neighbor and the sample by a random factor between 0,1 to 
generate a new 'syntethic' minority class row - this way i was able to generate more data for the minority class.

then i had a new dataset with 125000 rows, and 20 features, i finally ran the code with the new dataset and got in return a prediction still 0 for all players . i searched about what could be the possible issue and found it to be scaling - i had multiple features that range in thousands and other that are binary, and some even between 0,1 - to fix this i used min-max scaling so all stats would be distributed between 0-1 , i had one problem left which was the fact my computer was not capable of running this code,because i didnt use libraries - each vector operation had to iterate through the whole dataset, because there are many operations used in the algorithms , it took HOURS to run the code, the only time i managed to fully run the code and get results was when i left it open the whole night. so to make it simpler i only used numpy for the dot product to be more efficient and not itterate each time.
i then ran the code and finally got a full answer , with a pretty reasonable top 30 result:






still ,there were many problems with the results:
1. the algorithm predicted that ALL players in the top 5 leagues in 2024 would have a share of over 0, meaning they are all nominated - obviously that isnt possible, so i ran it again and to my suprise, each time i ran the code i got different results - some results had 2000 nominees and some 30, some had kane winning ,some bellingham and some vinicius - none had Rodri which was the actual winner, which is understandable cause my data doesnt contain much defending statistics and still he was way above other defenders, being top 20 in most runs and being the only non forward player in some of them - so the model did pick up on him being different which is good, but obviously still very unstablized.
2. while checking the weights for each features - i saw that the CL stats had by FAR the highest weights and the other features had not so high features - it is understanable because obviously the champions league has a huge impact on the winner of the ballon d'or - but the model depended to much on it and in some cases just predicted the whole top 30 to be real madrid players.
to handle this i found manual weights management and other techniques which i will later implement, and also some standardization models.

overall i will add feature engineering to try to overecome these problems and standardization and even i will try to add defence stats to try to optimize the model. i will update the results on here.

