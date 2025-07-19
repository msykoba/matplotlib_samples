import numpy as np
import matplotlib.pyplot as plt

# 方位角 (Azimuth)：0～360度
az_list = [0, 45, 90, 135, 180, 225, 270, 315]

# 仰角 (Elevation)：0～90度
el_list = [10, 30, 45, 60, 70, 50, 20, 15]

# 仰角を天頂中心の半径に変換（天頂＝中心＝0）
r = 90 - np.array(el_list)

# 方位角をラジアンに変換（matplotlibのpolarは0が東で反時計回りなので調整）
theta = np.radians(np.array(az_list))

# プロット
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_zero_location('N')  # 0°を北に
ax.set_theta_direction(-1)       # 時計回りにする

# プロット
ax.scatter(theta, r, c='blue', s=50)

# グリッドとラベル
ax.set_rlim(0, 90)
ax.set_rgrids([30, 60, 90], labels=['60°', '30°', '0°'])  # 中心が90→天頂、外周が0→地平線

# PRN
prn_list = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08']
for t, r_val, prn in zip(theta, r, prn_list):
    ax.text(t, r_val, prn, fontsize=8, ha='center', va='center', color='black')

ax.set_title('Skyplot (Azimuth-Elevation)', va='bottom')
plt.show()
