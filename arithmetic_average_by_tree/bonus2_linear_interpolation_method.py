import numpy as np
from math import exp, sqrt, log


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
M = 500
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


node_list_tree = np.zeros((n + 1, n + 1)).tolist()

for time in range(n + 1):
    for down_steps in range(time + 1):
        node = StoreNodeList(Amax[down_steps, time], Amin[down_steps, time], M)
        node_list_tree[down_steps][time] = node

        node_list_tree[down_steps][time].get_representative_linearly()
        # print(node.representative)
# print(node_list_tree[0][2].representative[1] - node_list_tree[0][2].representative[2])


# payoff for every representative ave price of each terminal node
for down_steps in range(n + 1):
    node_list_tree[down_steps][n].get_maturity_payoff(K)

    # print(node_list_tree[down_steps][n].option_value)

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
            Au = ((t / left_time * n + 1 + time) * val
                  + St * u ** (time + 1 - down_steps) * d ** down_steps) / (t / left_time * n + 1 + time + 1)
            Ad = ((t / left_time * n + 1 + time) * val
                  + St * u ** (time - down_steps) * d ** (down_steps + 1)) / (t / left_time * n + 1 + time + 1)

            # print(Au)


            ### linear interpolation method ###
            # 可能發生全部的 up_val 都一樣的情況，代表 Cu 也都一樣，那就直接對應 Cu
            # print(up_node.representative[M] - up_node.representative[0])
            if up_node.representative[0] == up_node.representative[M]:
                up_option = up_node.option_value[M]
                # up_option = up_node.option_value[M]
                # wu = (up_node.representative[0] - Au) \
                #      / (up_node.representative[0] - up_node.representative[1])
                # up_option = wu * up_node.option_value[1] + (1 - wu) * up_node.option_value[0]
                # print(up_option)
            else:
                ku = M * (Au - up_node.representative[0]) / (up_node.representative[M] - up_node.representative[0])
                # 可能發生 Au 比最小的 up_val 還小的情況，此時使 Cu 對應到最小的 option value
                if ku >= M:
                    up_option = up_node.option_value[M]
                # elif ku < 0:
                #     up_option = up_node.option_value[0]
                else:
                    index_ku = int(ku) + 1
                    wu = (up_node.representative[index_ku - 1] - Au) \
                         / (up_node.representative[index_ku - 1] - up_node.representative[index_ku])
                    up_option = wu * up_node.option_value[index_ku] + (1 - wu) * up_node.option_value[index_ku - 1]
                # print(up_option)
            # print(up_option)

            # 可能發生全部的 down_val 都一樣的情況，代表 Cd 也都一樣，那就直接對應 Cd
            if down_node.representative[0] == down_node.representative[M]:
                down_option = down_node.option_value[M]
                # wd = (down_node.representative[0] - Ad) \
                #      / (down_node.representative[0] - down_node.representative[1])
                # down_option = wd * down_node.option_value[1] + (1 - wd) * down_node.option_value[0]
                # print(down_option)
            else:
                kd = M * (Ad - down_node.representative[0]) / (
                            down_node.representative[M] - down_node.representative[0])
                # if kd >= M:
                #     down_option = down_node.option_value[M]
                # 可能發生 Ad 比最大的 down_val 還大的情況，此時使 Cd 對應到最大的 option value
                if kd < 0:
                    down_option = down_node.option_value[0]
                else:
                    index_kd = int(kd) + 1
                    wd = (down_node.representative[index_kd - 1] - Ad) \
                         / (down_node.representative[index_kd - 1] - down_node.representative[index_kd])
                    down_option = wd * down_node.option_value[index_kd] + (1 - wd) * down_node.option_value[index_kd - 1]
                # print(down_option)

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



print(f"option value = {node_list_tree[0][0].option_value[0]:.6f}")

# if up_node.representative[0] == up_node.representative[M]:
#     up_option = up_node.option_value[0]
#     print(up_option)
# else:
#     ku = M * (Au - up_node.representative[0]) / (up_node.representative[M] - up_node.representative[0])
#     # 可能發生 Au 比最小的 up_val 還小的情況，此時使 Cu 對應到最小的 option value
#     # print(ku)
#     if ku >= M:
#         up_option = up_node.option_value[M]
#     elif ku < 0:
#         up_option = up_node.option_value[0]
#     else:
#         index_ku = int(ku) + 1
#         wu = (up_node.representative[index_ku - 1] - Au) \
#              / (up_node.representative[index_ku - 1] - up_node.representative[index_ku])
#         up_option = wu * up_node.option_value[index_ku] + (1 - wu) * up_node.option_value[index_ku - 1]
#
# # print(up_option)











#%%
# Au = 54
# up_rep = [80, 78, 75, 74, 72, 70, 68, 67, 65, 60, 55, 53, 52, 49, 45, 44]
# up_option_value = [30, 29, 28, 26, 25, 24, 23, 21, 20, 19, 18, 17, 15, 14, 12, 10]
# M = 15
#
# if up_rep[0] == up_rep[M]:
#     up_option = up_option_value[0]
# else:
#     ku = M * (Au - up_rep[0]) / (up_rep[M] - up_rep[0])
#     # 可能發生 Au 比最小的 up_val 還小的情況，此時使 Cu 對應到最小的 option value
#     print(ku)
#     if ku >= M:
#         up_option = up_option_value[M]
#     elif ku < 0:
#         up_option = up_option_value[0]
#     else:
#         index_ku = int(ku) + 1
#         wu = (up_rep[index_ku - 1] - Au) \
#              / (up_rep[index_ku - 1] - up_rep[index_ku])
#         up_option = wu * up_option_value[index_ku] + (1 - wu) * up_option_value[index_ku - 1]
#
# print(wu)
# print(up_option)