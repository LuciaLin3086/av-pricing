import numpy as np
from math import exp, sqrt


class MaxMinAvePrice:

    def __init__(self, St, Save_t, r, q, sigma, t, left_time, n):
        self.St = St
        self.Save_t = Save_t
        self.r = r
        self.q = q
        self.sigma = sigma
        self.t = t
        self.left_time = left_time
        self.n = n

        self.delta_T = self.left_time / self.n
        self.u = exp(self.sigma * sqrt(self.delta_T))
        self.d = exp(- self.sigma * sqrt(self.delta_T))

        self.Amax = np.zeros((self.n + 1, self.n + 1)) # max average price
        self.Amin = np.zeros((self.n + 1, self.n + 1)) # min average price
        self.node_list_tree = np.zeros((self.n + 1, self.n + 1)).tolist()

        self.get_max_min_average_price() # get max & min average price for each node

    def get_max_min_average_price(self):
        if self.t == 0: # today = issue day
            for time in range(self.n + 1):
                for down_steps in range(time + 1):
                    self.Amax[down_steps, time] = (self.St
                                                   + self.St * self.u * (1 - self.u ** (time - down_steps)) / (1 - self.u)
                                                   + self.St * self.u ** (time - down_steps) * self.d * (1 - self.d ** down_steps) / (1 - self.d)
                                                   ) / (time + 1)
                    self.Amin[down_steps, time] = (self.St
                                                   + self.St * self.d * (1 - self.d ** down_steps) / (1 - self.d)
                                                   + self.St * self.d ** down_steps * self.u * (1 - self.u ** (time - down_steps)) / (1 - self.u)
                                                   ) / (time + 1)

        else:
            for time in range(self.n + 1):
                for down_steps in range(time + 1):
                    self.Amax[down_steps, time] = (self.Save_t * (self.t / self.left_time* self.n + 1)
                                                   + self.St * self.u * (1 - self.u ** (time - down_steps)) / (1 - self.u)
                                                   + self.St * self.u ** (time - down_steps) * self.d * (1 - self.d ** down_steps) / (1 - self.d)
                                                   ) / (self.t / self.left_time* self.n + 1 + time)
                    self.Amin[down_steps, time] = (self.Save_t * (self.t / self.left_time* self.n + 1)
                                                   + self.St * self.d * (1 - self.d ** down_steps) / (1 - self.d)
                                                   + self.St * self.d ** down_steps * self.u * (
                                                               1 - self.u ** (time - down_steps)) / (1 - self.u)
                                                   ) / (self.t / self.left_time* self.n + 1 + time)

## for testing
if __name__ == "__main__":
    St = 100
    Save_t = 110
    r = 0.05
    q = 0.02
    sigma = 0.5
    t = 0.1
    left_time = 0.5
    n = 3

    max_min_ave = MaxMinAvePrice(St, Save_t, r, q, sigma, t, left_time, n)
    Amax = max_min_ave.Amax
    print(Amax)

    # delta_T = left_time / n
    # u = exp(sigma * sqrt(delta_T))
    # d = exp(- sigma * sqrt(delta_T))
    #
    # print(St * u ** 2 * d ** 1)
    # print(St * d ** 3)
    #
    # A_2_1 = (St+ St * u * (1 - u ** (2 - 1)) / (1 - u)
    # + St * u ** (2-1) * d * (1 - d ** 1) / (1 - d) ) / (2 + 1)
    #
    # A_1_0 =( St + St * u ) / 2
    #
    # print(A_1_0)
    # print(A_2_1)