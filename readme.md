## Stock MARket Trade Recommender:

In the basis of Modern Portfolio Theory lies the assertion that by diversifying our assets (investing in many different instruments) we can minimize the risk. ( Rememeber variance of a linear combination of random variables can b
e computed by sum of combination of variances and pairwise covariances).

Smartr consists of two general steps. In the first step it predicts future return rate and return risk (variance). In the second step, for a given *desired return rate* it computes a combination of stock shares that can will minimize the portfolio risk.
 
Price prediction model consists of a shallow neural network that uses an observed window of opening prices (per days) to predict a *future* price. *Future* can  be defined by the user by setting the *look_ahead_window* variable. The user also must specify a number for splitting the observed historic prices into training and validation sets. The use of validation set here is two-fold. First, as is customary, validation set is used for hyper-parameter setting and selecting the *best* possible model. Second, Smartr uses the validation set to estimate the standard deviation for each stock instrument. 

Once the returns and risks are computed from the first step, the second step uses scipy optimization for finding the portfolio. Generally methods used for this step output the weight of investment in each instrument. Since in reality we can not buy stocks by weights, rather in integer number of shares, (integer optimization is much more complex) I have added some constraints to the optimization so that it tries to make the output values close to integers and also given a budget it limits the sum of expenditures to the budget. Both weights and number of shares can be outputted since the optimization for number of shares is more complex and does not always terminate successfully.

**Note:** if you are using the parallelized version, remember that tensorflow spawns multiple forks, and you should either limit this number through tensorflow onfigurations (I haven't checked that out) or you should be aware of the number of nodes and tasks per node you are submitting. Most system's don't allow more than 4096 processes for a user at a time. You can change this number using *ulimit* if you have root permissions. 
