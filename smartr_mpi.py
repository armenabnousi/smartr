import importlib
from mpi4py import MPI
import single_stock_predictor
#importlib.reload(single_stock_predictor)
import pickle
import yfinance as yf
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
#import mdn
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import minimize
import os
import fnmatch
import sys

#get_ipython().run_line_magic('matplotlib', 'notebook')


get_tickers = False 
read_tickers = False
get_histories = False
get_updated_data = False
read_data = False
read_updated_data = True
get_business_info = False #you can go through business info field to exlude companies you don't want to invest in
if get_updated_data or get_business_info:
    read_tickers = True
temp_rates_dir="temp_rates_risks"
models_dir="models"
candidate_companies_filename="candidate_companies_test.txt"

weekly = False
window_size = 30
batch_size = 32
shuffle_buffer = None
distributional = False
epochs = 40
training_points = 1500
sd_estimate_required = True
training_verbosity = 0
look_ahead_window = 5


def download_tickers():
    get_ipython().system('curl -o nasdaqtraded_companylist.txt ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqtraded.txt')
    symbols = pd.read_csv("nasdaqtraded_companylist.txt", sep = "|")
    symbols = symbols.iloc[0:(symbols.shape[0] - 1),:] #last row is time
    tickers = {}
    failed = []
    for i in symbols.index:
        sym = symbols.iloc[i]['Symbol']
        ticker = yf.Ticker(sym)
        try:
            check = ticker.calendar
        except Exception as e:
            print(' '.join(["disregarding", sym, type(e).__name__]))
            failed.append(sym)
            continue
        print(' '.join([sym, 'added']))
        name = symbols.iloc[i]['Security Name']
        tickers[sym] = {'name': name, 'ticker': ticker}
    sym_data = {'tickers':tickers, 'failed':failed}
    with open('sym_data.pkl', 'wb') as symfile:
        pickle.dump(sym_data, symfile)
    return(sym_data)
if get_tickers:
    sym_data = download_tickers()


if read_tickers:
    with open('sym_data.pkl', 'rb') as symfile:
        sym_data = pickle.load(symfile)
        tickers, failed = sym_data['tickers'], sym_data['failed']


def download_business_info(tickers):
    for ticker in tickers.keys():
        #print(ticker)
        tickers[ticker]['business_summary'] = tickers[ticker]['ticker'].info.get('longBusinessSummary', None)
    with open('sym_data.pkl', 'wb') as symfile:
        pickle.dump(sym_data, symfile)
if get_business_info:
    download_business_info(tickers)


def download_histories(tickers):
    period = "9y"
    d = download_ticker_histories(tickers, period = period, interval = "1d", columns = ['Open'])
    with open("daily_history_9y.pkl", 'wb') as histfile:
        pickle.dump(d, histfile)
    return d
if get_histories:
    d = download_histories(sym_data['tickers'])


def download_ticker_histories(tickers, start = None, period = None, end = None, interval = "1d", columns = ['Open']):
    msft = yf.Ticker("MSFT")
    if start is None:
        temp_hist = msft.history(period="9y", interval="1d")
        end = list(temp_hist.index)[-1]
        start = list(temp_hist.index)[0]
    else:
        temp_hist = msft.history(start = start, interval = "1d")
        end = list(temp_hist.index)[-1]
    d = pd.DataFrame(data = 0, columns = list(tickers.keys()), index = temp_hist.index)
    for counter, i in enumerate(d.columns):
        if counter % 100 == 0:
            print(i)
        hist = (tickers[i]['ticker'].history(start = start, end = end, interval = interval)[columns]).drop_duplicates(keep = 'last')
        d[i] = hist
    return d


if read_data:
    with open("daily_history_9y.pkl", 'rb') as infile:
        d = pickle.load(infile)


def update_data(d, tickers, columns = ['Open']):
    last_date = (pd.to_datetime(d.index.values[-1])).strftime("%Y-%m-%d")
    d_update = download_ticker_histories(tickers, start = last_date, interval = "1d", columns = columns)
    d_update = d_update.iloc[1:,:]
    d_update.head()
    d_merged = pd.concat([d, d_update], axis = 0)
    with open("daily_history_updated.pkl", 'wb') as histfile:
        pickle.dump(d_merged, histfile)
    return d_merged
if get_updated_data:
    d = update_data(d, tickers, columns = ['Open'])


if read_updated_data:
    with open("daily_history_updated.pkl", 'rb') as infile:
        d = pickle.load(infile)

def remove_na_stocks(d):
    #backward-filling na's 
    d = d.iloc[:, list(np.where(d.isna().sum() < 20)[0])]
    d = d.fillna(method = "bfill")
    d.shape

    #removing rows where the latest price is na
    d = d.loc[:, ~(d.iloc[d.shape[0] - 1,:].isna())]
    d.shape
    return d

d = remove_na_stocks(d)
if candidate_companies_filename is not None:
    with open(candidate_companies_filename, 'r') as infile:
        candidate_companies = infile.read().split()
    candidate_companies = [i for i in candidate_companies if i in list(d.columns.values)]
else:
    candidate_companies = list(d.columns.values)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
if rank == 0:
    print("running with ", size, "processors, for ", len(candidate_companies), "companies")

rank_share = int(np.ceil(len(candidate_companies)/size))
rank_indices = candidate_companies[(rank * rank_share):min(len(candidate_companies),((1+rank) * rank_share))]
if rank == 0:
    print("each rank will process", rank_share, "companies out of", len(candidate_companies))
sys.stdout.flush()


#stock_name = "MSFT"
try:
    os.mkdir(models_dir)
except FileExistsError:
    pass

rate_and_risks = pd.DataFrame({'price':[0]*rank_share, 'train_sd':[0]*rank_share, 'name':['stock']*rank_share,
				'valid_sd':[0]*rank_share, 'prediction':[0]*rank_share})
for index_number, stock_name in enumerate(rank_indices):
    pred, train_sd, train_forecasts, valid_sd, valid_forecasts = single_stock_predictor.predict_tomorrow(stock_name, d, model_outdir = models_dir, weekly = weekly, training_points = training_points, window_size = window_size, batch_size = batch_size, distributional = distributional, epochs = epochs, sd_estimate_required = sd_estimate_required, shuffle_buffer = shuffle_buffer, training_verbosity = training_verbosity, look_ahead_window = look_ahead_window)
    rate_and_risks.iloc[index_number, :] = [d.iloc[d.shape[0]-1][stock_name], train_sd, stock_name, valid_sd, pred]
try:
    os.mkdir(temp_rates_dir)
except FileExistsError:
    pass

rate_and_risks.to_csv(os.path.join(temp_rates_dir, "rates_" + str(rank) + ".csv"), index = False)

comm.Barrier()

if rank == 0:
    filenames = os.listdir(temp_rates_dir)
    filenames = [name for name in filenames if fnmatch.fnmatch(name, "rates_*.csv")]
    nrows = len(candidate_companies)
    final_rates = pd.DataFrame()
    for filename in filenames:
        final_rates = pd.concat([final_rates, pd.read_csv(os.path.join(temp_rates_dir, filename))], axis = 0)
    final_rates.to_csv("rates_and_risks.csv", index = False)
'''
# In[ ]:


plt.figure()
single_stock_predictor.plot_predictions(d[stock_name], train_forecasts, train_sd, window_size, limit_begin = 0, limit_end = 200, look_ahead_window = look_ahead_window)


# In[ ]:


plt.figure()
single_stock_predictor.plot_predictions(d[stock_name], train_forecasts, valid_sd, window_size, limit_begin = 0, limit_end = 200, look_ahead_window = look_ahead_window)


# In[ ]:


plt.figure()
single_stock_predictor.plot_predictions(d[stock_name][training_points:], valid_forecasts, train_sd, window_size, limit_begin = 0, limit_end = 200, look_ahead_window = look_ahead_window)


# In[ ]:


plt.figure()
single_stock_predictor.plot_predictions(d[stock_name][training_points:], valid_forecasts, valid_sd, window_size, limit_begin = 0, limit_end = 200, look_ahead_window = look_ahead_window)


# In[ ]:


print(train_sd, valid_sd)


# okay, so far we have showed that for the specified stock (here, MSFT aka microsoft) our model generates rather dependable predictions of price and our estimated standard deviation seems to be fitting at least visually. Of course, one can argue that we have picked an easy ticker, you'd expect microsoft to have a stable price. Well, I can't argue against that. But now, I'm going to randomly pick 10 tickers and perform the same operation on each of them. Before that I
# m going to use the *cov()* function from pandas to compute pairwise correlation between the selected tickers.

# In[ ]:


import random
ticker_set = random.sample(list(d.columns), 10)
ticker_set


# In[416]:


ticker_set = [name for name in candidates if name in list(d.columns.values)]


# In[417]:


d_select = d[ticker_set]
print(d_select.shape)
corels = d_select.corr()


# In[418]:


covariances = d_select.cov()
covariances.head()


# In[419]:


corels.head()


# In[420]:


corels_matrix = np.array(corels)
#heatmap(corels_matrix, corels.columns.values, corels.columns.values, cbarlabel = "correlation")


# In[422]:


weekly = False
window_size = 30
batch_size = 32
shuffle_buffer = None
distributional = False
epochs = 40
training_points = 1500
sd_estimate_required = True
model_outdir = "models"
look_ahead_window = 5
training_verbosity = 0


# In[423]:


prediction_dict = {}
for stock_name in ticker_set:
    print(" ".join(["processing", stock_name]))
    pred, train_sd, train_forecasts, valid_sd, valid_forecasts = single_stock_predictor.predict_tomorrow(stock_name, d, model_outdir = model_outdir, weekly = weekly, training_points = training_points, window_size = window_size, batch_size = batch_size, distributional = distributional, epochs = epochs, sd_estimate_required = sd_estimate_required, shuffle_buffer = shuffle_buffer, training_verbosity = training_verbosity, look_ahead_window = look_ahead_window)
    prediction_dict[stock_name] = {'expected_tomorrow': pred,
                                   'train_sd': train_sd,
                                   'train_forecasts': train_forecasts,
                                   'valid_sd': valid_sd,
                                   'valid_forecasts': valid_forecasts}
    current_price = d.iloc[d.shape[0]-1][stock_name]
    return_rate = pred / current_price - 1
    prediction_dict[stock_name]['return_rate'] = return_rate


# It looks like for a few of the stocks, validation error was smaller than the training error. I'm curious why.

# So here are the plots from validation data:

# In[ ]:


plt.figure()
plt.subplot(5,2,1)
for plt_index, stock_name in enumerate(ticker_set):
    ax = plt.subplot(5,2,plt_index + 1)
    single_stock_predictor.plot_predictions(d[stock_name][training_points:], prediction_dict[stock_name]['valid_forecasts'], prediction_dict[stock_name]['valid_sd'], window_size, limit_begin = 0, limit_end = 200, ax = ax, legend = False, look_ahead_window = look_ahead_window)
    
            


# and here are plots for training data:

# In[ ]:


plt.figure()
plt.subplot(5,2,1)
for plt_index, stock_name in enumerate(ticker_set):
    ax = plt.subplot(5,2,plt_index + 1)
    single_stock_predictor.plot_predictions(d[stock_name], prediction_dict[stock_name]['train_forecasts'], prediction_dict[stock_name]['valid_sd'], window_size, limit_begin = 0, limit_end = 200, ax = ax, legend = False, look_ahead_window = look_ahead_window)
    


# In[ ]:


d_select.head()


# In[ ]:


for stock_name in ticker_set:
    pred = prediction_dict[stock_name]['expected_tomorrow']
    current_price = d.iloc[d.shape[0] - 1, :][stock_name]
    return_rate = pred / current_price - 1
    prediction_dict[stock_name]['return_rate'] = return_rate


# In[ ]:


rat_risks = pd.DataFrame({'return_rate':[prediction_dict[stock_name]['return_rate'] for stock_name in ticker_set],
                        'risk' : [prediction_dict[stock_name]['valid_sd'] for stock_name in ticker_set],
                         'price': d.iloc[d.shape[0] - 1][ticker_set]})


# In what follows I'll be minimzing the mixed variance by selecting the "best" portfolio for a given return rate and budjet. For the optimization part I have heavily been dependent on the codes in [this tutorial](https://towardsdatascience.com/efficient-frontier-optimize-portfolio-with-scipy-57456428323e). Kudos to **J Li**. I have modified the code to compute the portfolio return rate and risk from mixture of normals distribution. Later I will also modify the optimization function by adding new constraints to compute the number of shares to buy with a given budget rather than the weight of instruments in the portfolio.

# In[ ]:


fig, ax = plt.subplots()
ax.scatter(rat_risks['risk'], rat_risks['return_rate'])
for stock_name in ticker_set:
    ax.annotate(stock_name, (prediction_dict[stock_name]['valid_sd'], prediction_dict[stock_name]['return_rate']))
plt.xlabel("risk")
plt.ylabel("return rate")
plt.show()


# variance of a linear combination of random variable can be computed by the formula below: [(source)](https://en.wikipedia.org/wiki/Variance)
# <img src="ext/lincomb_variance.png" style="height:200px">

# In[ ]:





# In[ ]:


def get_portfolio_risk(weights, rat_risks, covariances):
    weight_matrix = np.outer(weights , weights)
    weight_cov_combined = covariances * weight_matrix
    mixed_var = np.sum(np.sum(weight_cov_combined))
    return mixed_var

def get_portfolio_return(weights, rat_risks):
    total_return_rate = np.sum(rat_risks['return_rate'] * weights)
    return total_return_rate


# In[ ]:


def optimize_weights(rat_risks, target_return=0.1):
    instruments_count = rat_risks.shape[0]
    init_guess = np.ones(instruments_count) * (1.0 / instruments_count)
    bounds = ((0.0, 1.0),) * instruments_count
    weights = minimize(get_portfolio_risk, init_guess,
                       args=(rat_risks, covariances), method='SLSQP',
                       options={'disp': False},
                       constraints=({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)},
                                    {'type': 'eq', 'args': (rat_risks,),
                                     'fun': lambda inputs, rat_risks:
                                     target_return - get_portfolio_return(weights=inputs,
                                                                          rat_risks=rat_risks)}
                                   ),
                       bounds=bounds)
    return weights.x, weights.success, weights.status


# In[ ]:


weights, success, status = optimize_weights(rat_risks)
print(get_portfolio_risk(results, rat_risks, covariances))
print(success)
print(status)


# In[ ]:


weights


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(rat_risks['risk'], rat_risks['return_rate'])
for i, stock_name in enumerate(ticker_set):
    ax.annotate(' '.join([stock_name, str(round(weights[i], 5))]), (prediction_dict[stock_name]['valid_sd'], prediction_dict[stock_name]['return_rate']))
plt.title("weigth of each instrument for minimized risk for 0.1 return rate")
plt.xlabel("risk")
plt.ylabel("rate of return")
plt.show()


# The weights seem to make sense, with the instruments with higher risk getting a weight of zero and the ones with higher return rate and small risk getting the largest of weights. 
# Now let's modify the optimization function to accept and budget constraint as well as output number of shares per instrument (integer) rather than weights.

# In[ ]:


def get_portfolio_risk_by_shares(shares, rat_risks, covariances):
    weights = shares / np.sum(shares)
    weight_matrix = np.outer(weights , weights)
    weight_cov_combined = covariances * weight_matrix
    mixed_var = np.sum(np.sum(weight_cov_combined))
    return mixed_var

def get_portfolio_return_by_shares(shares, rat_risks, budget):
    spent = np.sum(rat_risks['price'] * shares)
    unspent = budget - spent
    returns = spent * np.sum(rat_risks['return_rate'] * shares) + unspent
    total_return_rate = (returns/budget) - 1
    return total_return_rate

def budget_constraint(shares, rat_risks, budget):
    prices = np.array(rat_risks['price'])
    unspent = budget - np.sum(shares * prices)
    return(unspent)


# In[ ]:


def optimize_shares(rat_risks, target_return=0.1, budget = 2000):
    #normalized_prices = prices / prices.ix[0, :]
    instruments_count = rat_risks.shape[0]
    init_guess = np.ones(instruments_count) * 2 #(1.0 / instruments_count)
    bounds = ((0.0, np.inf),) * instruments_count
    shares = minimize(get_portfolio_risk_by_shares, init_guess,
                       args=(rat_risks, covariances), method='SLSQP',
                       options={'disp': False},
                       constraints=({'type': 'ineq', 'fun': lambda x: budget_constraint(x, rat_risks, budget)}, #make sure total is less than budget
                                    {'type': 'eq', 'args': (rat_risks, budget), #make the return rate equal to the expected rate
                                     'fun': lambda inputs, rat_risks, budget:
                                     target_return - get_portfolio_return_by_shares(inputs, rat_risks, budget)},
                                    {'type':'eq','fun': lambda x : max([0] + [x[i]-int(x[i]) for i in range(len(x)) if x[i]-int(x[i]) > 0])}, #try to make them as close to integer as possible
                                   ),
                       bounds=bounds
                     )
    return shares.x, shares.success, shares.status, shares.message


# In[ ]:


budget = 2000
share_distribution, success, status, message = optimize_shares(rat_risks, budget = budget)
share_distribution
output = rat_risks
output['shares'] = np.floor(share_distribution)


# In[ ]:


message


# In[ ]:


print(' '.join(['retrun rate', str(get_portfolio_return_by_shares(rat_risks['shares'], rat_risks, budget))]))
print(' '.join(['risk', str(get_portfolio_risk_by_shares(rat_risks['shares'], rat_risks, covariances))]))
print(' '.join(['invested:', str(np.sum(rat_risks['shares'] * rat_risks['price']))]))


# In[ ]:


share_distribution


# In[ ]:


rat_risks


# In[ ]:


#t = pd.DataFrame({'weights':weights})
weights = np.random.randint(1, 10, size = 4)
print(weights)
weight_mat = np.outer(weights , weights)
#pd.pivot_table(t, values = ['weights'], index = 'weights', columns = ['weights'])
#t.info()
print(weight_mat)
#np.stack(weights, 


# In[ ]:


weight_mat = np.outer(weights , weights)
mv = covariances.iloc[-4:,-4:] * weight_mat
np.sum(np.sum(mv))
print(mv)


# In[ ]:


np.sum(np.sum(mv))


# In[ ]:


print(covariances.iloc[-4:,-4:])
print(weight_mat)
print(covariances.iloc[-4:,-4:] * weight_mat)


# In[ ]:


-11.118789 * 16


# In[ ]:

'''

#funnction copied from matplotlib gallery
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

