# datetime使用
import datetime
import numpy as np
import matplotlib.pyplot as plt

# サンプルデータを作成
np.random.seed(42)  # 乱数の再現性を確保
dates = [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=i) for i in range(10)]
values = np.random.randint(50, 100, size=10)  # 50〜100の範囲でランダムな値

# グラフを描画
plt.figure(figsize=(10, 5))
plt.plot(dates, values, marker="o", linestyle="-", color="b", label="Random Value")

# グラフの装飾
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Sample Time Series Data")
plt.xticks(rotation=45)  # 日付を見やすくする
plt.legend()
plt.grid()

# 表示
plt.show()
