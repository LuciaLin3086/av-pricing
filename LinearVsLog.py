import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # 統計圖型庫底層由matplot lib 建構, 可以搭配使用
from ArithAvgByTree import AvgOption
sns.set() # 用seaborn設定圖片會比較漂亮
sns.set_style('white') # set_stytle可以用來設定背景與筆觸, 我習慣用white

# 設定調色盤, 可以在https://seaborn.pydata.org/tutorial/color_palettes.html 選顏色
sns.set_palette("rocket")

plt.figure(figsize = (12, 8)) # 設定畫版大小(12, 8) 表示x長12 y長8 常見的設定有(8, 6), (12, 8)
interpolation_ls = ['linear', 'log']
for interpolation in interpolation_ls:
    price_ls = []
    for M in np.arange(50, 450, 50):

        European_call = AvgOption(50, 50, 0.1, 0.05, 0.8, 0, 0.25, M, 100, 50, 'C', 'E', interpolation, 'bin')
        price_ls.append(European_call.get_value())

    plt.plot(np.arange(50, 450, 50), price_ls, label=interpolation,
             marker='D')  # label: 想要在legend顯示的文字 
                          # marker = "D" 表示dimond, 可以換成'o'表示dot, 也可以不加, 手動調整color可以加上color參數, 但我用set_pallete 調色盤所以他會自動從裡面選


plt.title("Log vs Linear", fontsize = 14, weight = "bold") # fontsize: 文字大小, weight文字粗度
plt.xlabel("M") # x 軸文字
plt.ylabel("price")  # ｙ 軸文字
plt.legend(loc = 'best') # 標籤 標籤文字在plot函數中的label裡面設定
sns.despine() # 原本圖形四周會有外匡, 用這個dispine 就會去除掉上面和右邊的外匡
plt.savefig("LovLinear.pdf", dpi = 1000, bbox_inches = 'tight') # 儲存圖片, 檔名.pdf, .png都可以, dpi 是相數越高越清晰但會跑很久, bbox_inches是指圖片四周留白大小,一般來說都選最小
plt.show()