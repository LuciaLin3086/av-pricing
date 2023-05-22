import numpy as np
import pandas as pd
from math import exp, log, sqrt

# input variables
St = 50
Save_t = 50
K = 50
r = 0.1
q = 0.05
sigma = 0.8
t = 0.25
left_time = 0.25
n = 100

# hyperparameter
N = 10000
rep = 20


class PriceSimulator:
    def __init__(self, St, r, q, sigma, left_time, t, n, N):
        self.St = St
        self.r = r
        self.q = q
        self.sigma = sigma
        self.left_time = left_time
        self.t = t
        self.n = n
        self.N = N

        self.delta_T = self.left_time / self.n

    def price_simulate(self):
        std_normal_sample = pd.DataFrame(np.random.randn(int(self.n), int(self.N))) # 一個column代表一個path
        normal_sample = std_normal_sample * self.sigma * sqrt(self.delta_T) + \
               (self.r - self.q - self.sigma ** 2 / 2) * self.delta_T
        path_price = normal_sample.cumsum(axis = 0) # 沿著row走，代表分別對每個column運算
        lnSt = path_price + log(self.St)
        St = np.exp(lnSt) # St為 n * N 的dataframe，每個column為第1期到第n期的模擬股價路徑

        return St


class Payoff:
    def __init__(self, price, Save_t, left_time, t, n):
        self.price = price # dataframe
        self.Save_t = Save_t
        self.left_time = left_time
        self.t = t
        self.n = n

    def get_payoff(self, K):
        # 找出每個path中的平均股價，Save為一個有N個元素的row
        Save = (self.Save_t * (self.t / self.left_time * self.n + 1) + self.price.mean(axis = 0) * self.n) /  (self.t / self.left_time * self.n + 1 + self.n)
        payoff = np.where((Save - K) > 0, (Save - K), 0)

        return payoff


# repeat for 20 times
values_ls = []
for i in range(rep):
    price_simulate = PriceSimulator(St, r, q, sigma, left_time, t, n, N)
    price = price_simulate.price_simulate()

    lookback_payoff = Payoff(price, Save_t, left_time, t, n)
    payoff = lookback_payoff.get_payoff(K)

    value = exp(-r * left_time) * payoff.mean()
    values_ls.append(value)

values_arr = np.array(values_ls)
CI1 = values_arr.mean() - 2 * values_arr.std()
CI2 = values_arr.mean() + 2 * values_arr.std()


print(f"95% CI: [{CI1:.6f}, {CI2:.6f}]")
print(f"European option value = {values_arr.mean():.6f}")
