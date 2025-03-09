import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def random_walk(n_steps):
    # 平均, 標準偏差
    mu = 0
    sigma = 0.1
    eps = np.random.normal(mu, sigma, n_steps-1)

    # # 初期値
    # y0 = 0
    # eps = np.insert(eps, 0, y0)
    # y = np.cumsum(eps)

    y = np.zeros(n_steps)
    y[0] = 0
    # a = 1.0  # 時系列の根(-1<a<1のとき定常、1以上のとき非定常)
    # c = 0.0  # ドリフト率
    a = 1.0
    c = 0.001
    for i in range(n_steps-1):
        y[i+1] = a * y[i] + eps[i] + c
    return y


def autocovariance(series, lag):
    mean = np.mean(series)
    n = len(series)
    return np.sum((series[:n-lag] - mean) * (series[lag:] - mean)) / n


def autocorrelation(series, lag):
    return autocovariance(series, lag) / autocovariance(series, 0)


def plot_and_correlogram(series, max_lag=300):
    dates = pd.date_range(start="2020-01-01", periods=len(series), freq="D")
    fig, axes = plt.subplots(2, 2 ,figsize=(20, 10))
    axes[0,0].plot(dates, series, label='Random Walk')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].set_ylabel('Value')
    axes[0,0].set_title('1D Random Walk')
    axes[0,0].legend()
    axes[0,0].grid()

    lags = np.arange(1, max_lag + 1)
    acf_values = [autocorrelation(series, lag) for lag in lags]
    axes[1,0].bar(lags, acf_values, width=0.5, color='b', alpha=0.7)
    axes[1,0].set_xlabel('Lag')
    axes[1,0].set_ylabel('Autocorrelation')
    axes[1,0].set_title('Correlogram')
    axes[1,0].grid()

    diff_series = np.diff(series, n=1)
    axes[0,1].plot(dates[1:], diff_series, label='Random Walk')
    axes[0,1].set_xlabel('Steps')
    axes[0,1].set_ylabel('Value')
    axes[0,1].set_title('Differenced 1D Random Walk')
    axes[0,1].grid()

    acf_values_diff = [autocorrelation(diff_series, lag) for lag in lags]
    axes[1,1].bar(lags, acf_values_diff, width=0.5, color='b', alpha=0.7)
    axes[1,1].set_xlabel('Lag')
    axes[1,1].set_ylabel('Autocorrelation')
    axes[1,1].set_title('Correlogram')
    axes[1,1].grid()
    plt.show()


def plot_random_walk(n_steps):
    np.random.seed(1)
    walk = random_walk(n_steps)
    # ランダムウォークの場合、期待値は0
    # ランダムウォークの場合、分散はt*sigma*2

    # 自己共分散の計算と表示 lag=0とすると分散
    # ランダムウォークの場合、(t-k)*sigma**2
    lag = 120
    acv = autocovariance(walk, lag)
    print(f'Lag-{lag} Autocovariance : {acv}')

    # 自己相関の計算と表示
    # ランダムウォークの場合、sqrt(t-k/t)
    acf = autocorrelation(walk, lag)
    print(f'Lag-{lag} Autocorrelation: {acf}')

    # 描画
    plot_and_correlogram(walk)


# 実行
plot_random_walk(1000)
