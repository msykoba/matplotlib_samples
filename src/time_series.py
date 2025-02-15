# pandasのdate_range使用
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# サンプルデータを作成
np.random.seed(42)  # 乱数の再現性を確保
# dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
dates = pd.date_range(start="2024-01-01", end="2024-01-10", freq="1440min")  # 分刻み指定
print(dates)
values = np.random.randint(50, 100, size=10)  # 50〜100の範囲でランダムな値

# データをDataFrameに格納
df = pd.DataFrame({"Date": dates, "Value": values})

# グラフを描画
plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Value"], marker="o", linestyle="-", color="b", label="Random Value")

# グラフの装飾
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Sample Time Series Data")
plt.xticks(rotation=45)  # 日付を見やすくする
plt.legend()
plt.grid()

# subplot
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(111)
ax1.plot(df["Date"], df["Value"], marker="o", linestyle="-", color="b", label="Random Value")
for xticks in ax1.get_xticklabels():
    xticks.set_rotation(45)

# 表示
plt.show()