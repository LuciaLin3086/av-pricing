import numpy as np
from math import exp, sqrt, log
import time as TIME

start = TIME.time()

# input variables
St = 100
Save_t = 110
K = 90
r = 0.05
q = 0.02
sigma = 0.5
t = 0.1
left_time = 0.5
n = 100
M = 50
E_A = "E"

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
                                               + self.St * self.u ** (time - down_steps) * self.d * (
                                                           1 - self.d ** down_steps) / (1 - self.d)
                                               ) / (self.t / self.left_time * self.n + 1 + time)
                self.Amin[down_steps, time] = (self.Save_t * (self.t / self.left_time * self.n + 1)
                                               + self.St * self.d * (1 - self.d ** down_steps) / (1 - self.d)
                                               + self.St * self.d ** down_steps * self.u * (
                                                       1 - self.u ** (time - down_steps)) / (1 - self.u)
                                               ) / (self.t / self.left_time * self.n + 1 + time)

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


    def get_maturity_payoff(self, K):
        for ave in self.representative:
            self.option_value.append(max(ave - K, 0))



max_min_ave_price = MaxMinAvePrice(St, Save_t, r, q, sigma, t, left_time, n)
Amax = max_min_ave_price.Amax
Amin = max_min_ave_price.Amin
# print(Amax)
# print(Amin)
# print(Amax[0, 2] - Amin[0, 2])


node_list_tree = np.zeros((n + 1, n + 1)).tolist()

for time in range(n + 1):
    for down_steps in range(time + 1):
        node = StoreNodeList(Amax[down_steps, time], Amin[down_steps, time], M)
        node_list_tree[down_steps][time] = node

        node_list_tree[down_steps][time].get_representative_linearly()
        # print(node.representative)
# print(node_list_tree[0][2].representative[0] - node_list_tree[0][2].representative[1])


# payoff for every representative ave price of each terminal node
for down_steps in range(n + 1):
    node_list_tree[down_steps][n].get_maturity_payoff(K)


# backward induction for option value
for time in list(reversed(range(0, n))):
    for down_steps in range(0, time + 1):
        node = node_list_tree[down_steps][time]
        up_node = node_list_tree[down_steps][time + 1]
        down_node = node_list_tree[down_steps + 1][time + 1]

        u = max_min_ave_price.u
        d = max_min_ave_price.d

        for val in node.representative:
            # 先算出每個representative average price 的Au、Ad
            # 先算出每個representative average price 的Au、Ad
            Au = ((t / left_time * n + 1 + time) * val
                  + St * u ** (time + 1 - down_steps) * d ** down_steps) / (t / left_time * n + 1 + time + 1)
            Ad = ((t / left_time * n + 1 + time) * val
                  + St * u ** (time - down_steps) * d ** (down_steps + 1)) / (t / left_time * n + 1 + time + 1)
            # print(Au)

            ### binary search ###
            a1 = 0
            b1 = M
            while True:
                med1 = int((a1 + b1) / 2)
                if b1 - a1 == 1:
                    break
                else:
                    if Au > up_node.representative[med1]:
                        a1 = a1
                        b1 = med1
                    elif Au < up_node.representative[med1]:
                        a1 = med1
                        b1 = b1
                    else:
                        b1 = med1
                        break

            index_ku = b1
            # print(index_ku)
            # 可能發生全部的 up_val 都一樣的情況，那就直接對應 Cu
            # print(up_node.representative[index_ku - 1] - up_node.representative[index_ku])
            if up_node.representative[index_ku - 1] == up_node.representative[index_ku]:
                up_option = up_node.option_value[index_ku]
                # print(up_option)
            else: # linear interpolation
                wu = (up_node.representative[index_ku - 1] - Au) \
                     / (up_node.representative[index_ku - 1] - up_node.representative[index_ku])
                up_option = wu * up_node.option_value[index_ku] + (1 - wu) * up_node.option_value[index_ku - 1]
            # print(up_option)

            a2 = 0
            b2 = M
            while True:
                med2 = int((a2 + b2) / 2)
                if b2 - a2 == 1:
                    break
                else:
                    if Ad > down_node.representative[med2]:
                        a2 = a2
                        b2 = med2
                    elif Ad < down_node.representative[med2]:
                        a2 = med2
                        b2 = b2
                    else:
                        b2 = med2
                        break

            index_kd = b2
            # print(index_kd)
            # 可能發生全部的 down_val 都一樣的情況，那就直接對應 Cd
            if down_node.representative[index_kd - 1] == down_node.representative[index_kd]:
                down_option = down_node.option_value[index_kd]
                # print(down_option)
            else: # linear interpolation
                wd = (down_node.representative[index_kd - 1] - Ad) \
                     / (down_node.representative[index_kd - 1] - down_node.representative[index_kd])
                down_option = wd * down_node.option_value[index_kd] + (1 - wd) * down_node.option_value[index_kd - 1]

            # print(down_option)



            p = max_min_ave_price.p
            delta_T = max_min_ave_price.delta_T

            if E_A == "A": # American option
                exercise_value = val - K
                intrinsic_value = exp(-r * delta_T) * (p * up_option + (1 - p) * down_option)
                option_val = max(exercise_value, intrinsic_value)
            else: # European option
                option_val = exp(-r * delta_T) * (p * up_option + (1 - p) * down_option)

            # print(option_val)
            node.option_value.append(option_val)

end = TIME.time()

print(f"option value = {node_list_tree[0][0].option_value[0]:.6f}")
print(f"computation time by binary search: {end - start}")







#%%
# start1 = time.time()
# Au = 54
# up_rep = [80, 78, 75, 74, 72, 70, 68, 67, 65, 60, 55, 53, 52, 49, 45, 44]
# up_option_value = [25, 24, 23, 21, 18]
#
# a1 = 0
# b1 = 15
# while True:
#     med = int((a1 + b1) / 2)
#     if b1 - a1 == 1:
#         break
#     else:
#         if Au > up_rep[med]:
#             a1 = a1
#             b1 = med
#         else:
#             a1 = med
#             b1 = b1
#
# end1 = time.time()
# print(a1)
# print(b1)
# print(end1 - start1)
