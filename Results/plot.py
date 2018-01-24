import numpy as np
import matplotlib.pyplot as plt

# x = np.array([0.3427, 0.3441, 0.3484, 0.4767, 0.5684, 0.6857, 0.7000]);
# y = np.array([0.6439, 0.6439,0.6439, 0.6212,0.4091,0.1818,0.0530])
# plt.plot(x, y, label="Selective Search 0")
# x1 = np.array([0.2888, 0.2907, 0.2951,0.3849,0.5404,0.6857,0.8667]);
# y1 = np.array([0.8182, 0.8258,0.8182,0.8106,0.6591,0.3636,0.0985])
# plt.plot(x1, y1, label="Selective Search 1")
# x2 = np.array([0.2296, 0.2301,0.2321,0.2923,0.4356,0.6344,0.8889]);
# y2 = np.array([0.7879, 0.7879,0.7879,0.7727,0.6667,0.4470,0.1818])
# plt.plot(x2, y2, label="Selective Search 2")
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Iou0.25_64win_srFeat PR-curve')
# plt.legend()
# plt.ylim(0,1)



x = np.array([0.3465, 0.3465, 0.3520, 0.4570, 0.5948, 0.6613, 0.6842]);
y = np.array([0.6667, 0.6667,0.6667,0.6439,0.5227,0.3106,0.0985])
plt.plot(x, y, label="Selective Search 0")
x1 = np.array([0.2921, 0.2929, 0.2973, 0.3682,0.5323,0.6698,0.8077,0.8667]);
y1 = np.array([0.8409, 0.8409, 0.8333,0.8258,0.7500,0.5379,0.3182,0.0985])
plt.plot(x1, y1, label="Selective Search 1")
x2 = np.array([0.2284, 0.2294, 0.2340,0.2811,0.4353,0.5827,0.7714,0.9130,1.0000]);
y2 = np.array([0.8030, 0.8030,0.8030,0.7879,0.7652,0.6136,0.4091,0.1591,0.0076])
plt.plot(x2, y2, label="Selective Search 2")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Iou0.25_64win_allFeat PR-curve')
plt.legend()
plt.ylim(0,1)

plt.show()