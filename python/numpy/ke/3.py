import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
# 读取excel中的数据
df = pd.read_excel(r'C:\Users\75421\Desktop\数据.xlsx', names=['time', 'SSTA'])
# print(df)
X = df['time']
Y = df['SSTA']
# print(X, Y)
c = 0.5
y_above = np.zeros(Y.shape[0])
y_below = np.zeros(Y.shape[0])

for i in range(Y.shape[0]):
    if abs(Y[i]) >= c:
        y_above[i] = Y[i]
    else:
        y_below[i] = Y[i]
# print(y_above, y_below)
# plt.tight_layout
plt.title("气温数据图")
plt.xlabel("Time")
plt.xticks(rotation=60, fontsize=6)
plt.ylabel("SSTA")
plt.bar(X, y_above, width=5.0, color='red', label="Above average")
plt.bar(X, y_below, width=5.0, color='grey', label="below average")

plt.axhline(y=c, color='black', linestyle=":") 
plt.savefig(r"C:\Users\75421\Desktop\shares_bar.png")
print("柱状图生成成功！请在桌面查看")