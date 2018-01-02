# This script is supposed to graph the efficient frontier
# based on the historical prices of member securities of
# a given portfolio.
# 
# A qualified portfolio should contain only securities 
# that can be tracked using tickers on Yahoo! Finance.
#
# Original Version (V1.0) by Johnny MOON


import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
import datetime


database = pd.DataFrame()
tickers = ['AAPL', 'FB', 'GOOG', 'MSFT']
q = len(tickers)
records = 0
for i in tickers:
    tmp = web.DataReader(i, 'yahoo', '1/1/2010', datetime.date.today())
    database[i] = tmp['Adj Close']
    if records == 0:
        records = len(tmp)
    else:
        records = min(records, len(tmp))
returns = np.log(database / database.shift(1))
returns = returns.tail(records - 1)
returns.fillna(value=0, inplace=True)
cov = returns.cov() * 252
mean = returns.mean() * 252

sds = []
rtn = []

for _ in range(100000):
    w = np.random.rand(q)
    w /= sum(w)
    rtn.append(sum(mean * w))
    sds.append(np.sqrt(reduce(np.dot, [w, cov, w.T])))

plt.plot(sds, rtn, 'ro')


def sd(w):
    return np.sqrt(reduce(np.dot, [w, cov, w.T]))

x0 = np.array([1.0 / q for x in range(q)])
bounds = tuple((0, 1) for x in range(q))

given_r = np.arange(.18, .26, .001)
risk = []


def sharpe(w):
    return sum(w * mean) / sd(w)

for i in given_r:
    constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1},
                   {'type': 'eq', 'fun': lambda x: sum(x * mean) - i}]
    rst = solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds)
    risk.append(rst.fun)

plt.plot(risk, given_r, 'x')

constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
minv = solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds).fun
minvr = sum(solver.minimize(sd, x0=x0, constraints=constraints, bounds=bounds).x * mean
plt.plot(minv, minvr, 'y*')

constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1}
maxsv = solver.minimize(-sharpe, x0=x0, constraints=constraints, bounds=bounds).fun
maxsr = sum(solver.minimize(-sharpe, x0=x0, constraints=constraints, bounds=bounds).x * mean
plt.plot(maxsv, maxsr, 'y*')

plt.grid(True)
plt.title('Efficient Frontier: AAPL, FB, GOOG, MSFT')
plt.xlabel('portfolio volatility')
plt.ylabel('portfolio return')


