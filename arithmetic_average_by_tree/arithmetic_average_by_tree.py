import numpy as np
from math import exp, sqrt, log


# input variables
St = 50
Save_t = 50
K = 50
r = 0.1
q = 0.05
sigma = 0.8
t = 0
left_time = 0.25 # T - t
n = 100
M = 100

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
        self.p = (exp((self.r - self.q) * self.delta_T) - self.d) / (self.u - self.d)

        self.Amax = np.zeros((self.n + 1, self.n + 1)) # max average price
        self.Amin = np.zeros((self.n + 1, self.n + 1)) # min average price
        self.node_list_tree = np.zeros((self.n + 1, self.n + 1)).tolist()

        self.get_max_min_average_price() # get max & min average price for each node

    def get_max_min_average_price(self):
        for time in range(self.n + 1):
            for down_steps in range(time + 1):
                self.Amax[down_steps, time] = (self.Save_t * (self.t / self.left_time * self.n + 1)
                                               + self.St * self.u * (1 - self.u ** (time - down_steps)) / (1 - self.u)
                                               + self.St * self.u ** (time - down_steps) * self.d * (1 - self.d ** down_steps) / (1 - self.d)
                                               ) / (self.t / self.left_time* self.n + 1 + time)
                self.Amin[down_steps, time] = (self.Save_t * (self.t / self.left_time * self.n + 1)
                                               + self.St * self.d * (1 - self.d ** down_steps) / (1 - self.d)
                                               + self.St * self.d ** down_steps * self.u * (
                                                           1 - self.u ** (time - down_steps)) / (1 - self.u)
                                               ) / (self.t / self.left_time* self.n + 1 + time)

class StoreNodeList:

    def __init__(self, Amax, Amin, M):
        self.Amax = Amax
        self.Amin = Amin
        self.M = M

        self.representative = np.zeros(self.M + 1)
        self.option_value = []

    def get_representative_linearly(self):
        for k in range(self.M + 1):
            self.representative[k] = (self.M - k) / self.M * self.Amax + k / self.M * self.Amin
    # Bonus 1
    def get_representative_logarithmically(self):
        for k in range(self.M + 1):
            self.representative[k] = exp((self.M - k) / self.M * log(self.Amax) + k / self.M * log(self.Amin))


    def get_maturity_payoff(self, K): # average call
        for ave in self.representative:
            self.option_value.append(max(ave - K, 0))



max_min_ave_price = MaxMinAvePrice(St, Save_t, r, q, sigma, t, left_time, n)
Amax = max_min_ave_price.Amax
Amin = max_min_ave_price.Amin

for computation_way in ["Linearly", "Logarithmically"]:
    if computation_way == "Linearly":
        print("Linearly equally-spaced placement method")
    elif computation_way == "Logarithmically":
        print("\nLogarithmically equally-spaced placement method")

    for E_A in ["E", "A"]: # European or American

        node_list_tree = np.zeros((n + 1, n + 1)).tolist()

        for time in range(n + 1):
            for down_steps in range(time + 1):
                node = StoreNodeList(Amax[down_steps, time], Amin[down_steps, time], M)
                node_list_tree[down_steps][time] = node

                if computation_way == "Linearly":
                    node_list_tree[down_steps][time].get_representative_linearly()
                elif computation_way == "Logarithmically":
                    node_list_tree[down_steps][time].get_representative_logarithmically()


        # payoff for every representative ave price of each terminal node
        for down_steps in range(n + 1):
            node_list_tree[down_steps][n].get_maturity_payoff(K)

        # backward induction for option value
        for time in range(n - 1, -1, -1):
            for down_steps in range(0, time + 1):
                node = node_list_tree[down_steps][time]
                up_node = node_list_tree[down_steps][time + 1]
                down_node = node_list_tree[down_steps + 1][time + 1]

                u = max_min_ave_price.u
                d = max_min_ave_price.d

                for val in node.representative:
                    # 先算出每個representative average price 的Au、Ad
                    Au = ((t / left_time * n + 1 + time) * val
                          + St * u ** (time + 1 - down_steps) * d ** down_steps) / (t / left_time * n + 1 + time + 1)
                    Ad = ((t / left_time * n + 1 + time) * val
                          + St * u ** (time - down_steps) * d ** (down_steps + 1)) / (t / left_time * n + 1 + time + 1)

                    ### sequential search for ku ###
                    index_ku = 0
                    for up_val in up_node.representative:
                        if Au >= up_val:
                            break
                        elif Au < up_val:
                            index_ku += 1

                    # 可能發生 Au 比最小的 up_val 還小的情況，此時使 Cu 對應到最小的 option value
                    if index_ku - 1 == M: # index_ku = M + 1
                        up_option = up_node.option_value[M]
                    # 可能發生全部的 up_val 都一樣的情況，代表 Cu 也都一樣，那就直接對應 Cu
                    elif up_node.representative[index_ku - 1] == up_node.representative[index_ku]:
                        up_option = up_node.option_value[index_ku]
                    else: # linear interpolation
                        wu = (up_node.representative[index_ku - 1] - Au) \
                             / (up_node.representative[index_ku - 1] - up_node.representative[index_ku])
                        up_option = wu * up_node.option_value[index_ku] + (1 - wu) * up_node.option_value[index_ku - 1]

                    ### sequential search for kd ###
                    index_kd = 0
                    for down_val in down_node.representative:
                        if Ad >= down_val:
                            break
                        elif Ad < down_val:
                            index_kd += 1

                    if index_kd - 1 == M:
                        down_option = down_node.option_value[index_kd - 1]
                    elif down_node.representative[index_kd - 1] == down_node.representative[index_kd]:
                        down_option = down_node.option_value[index_kd]
                    else: # linear interpolation
                        wd = (down_node.representative[index_kd - 1] - Ad) \
                             / (down_node.representative[index_kd - 1] - down_node.representative[index_kd])
                        down_option = wd * down_node.option_value[index_kd] + (1 - wd) * down_node.option_value[index_kd - 1]


                    p = max_min_ave_price.p
                    delta_T = max_min_ave_price.delta_T

                    if E_A == "A": # American option
                        exercise_value = val - K
                        intrinsic_value = exp(-r * delta_T) * (p * up_option + (1 - p) * down_option)
                        option_val = max(exercise_value, intrinsic_value)
                    else: # European option
                        option_val = exp(-r * delta_T) * (p * up_option + (1 - p) * down_option)


                    node.option_value.append(option_val)



        if E_A == "A":
            print(f"American option value = {node_list_tree[0][0].option_value[0]:.4f}")
        else:
            print(f"European option value = {node_list_tree[0][0].option_value[0]:.4f}")



