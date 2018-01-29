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



# x = np.array([0.3465, 0.3465, 0.3520, 0.4570, 0.5948, 0.6613, 0.6842]);
# y = np.array([0.6667, 0.6667,0.6667,0.6439,0.5227,0.3106,0.0985])
# plt.plot(x, y, label="Selective Search 0")
# x1 = np.array([0.2921, 0.2929, 0.2973, 0.3682,0.5323,0.6698,0.8077,0.8667]);
# y1 = np.array([0.8409, 0.8409, 0.8333,0.8258,0.7500,0.5379,0.3182,0.0985])
# plt.plot(x1, y1, label="Selective Search 1")
# x2 = np.array([0.2284, 0.2294, 0.2340,0.2811,0.4353,0.5827,0.7714,0.9130,1.0000]);
# y2 = np.array([0.8030, 0.8030,0.8030,0.7879,0.7652,0.6136,0.4091,0.1591,0.0076])
# plt.plot(x2, y2, label="Selective Search 2")
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Iou0.25_64win_allFeat PR-curve')
# plt.legend()
# plt.ylim(0,1)


# y = np.array([0.6753, 0.6753, 0.6711, 0.8372, 0.8571, 1, 1]);
# x = np.array([0.3939, 0.3939, 0.3864, 0.2727,0.1364,0.0227,0.0076])
# plt.plot(x, y, label="Selective Search 0")
# y1 = np.array([0.5592, 0.5592, 0.5822, 0.6990,0.8958,1.000,1.0000,1]);
# x1 = np.array([0.6439, 0.6439, 0.6439,0.5455,0.3258,0.1212,0.0227,0.0152])
# plt.plot(x1, y1, label="Selective Search 1")
# y2 = np.array([0.5028, 0.5028, 0.5141,0.6241,0.8571,0.9630,1.000,1.0000]);
# x2 = np.array([0.6894, 0.6894,0.6894,0.6288,0.4545,0.1970,0.0530,0.0227])
# plt.plot(x2, y2, label="Selective Search 2")
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Iou0.25_64win_srFeat_origData(150+obj_imgs,300-bg_imgs) PR-curve')
# plt.legend()
# plt.ylim(0.4,1.1)
# plt.xlim(0,1)

y = np.array([0.7922, 0.7922, 0.8043, 1.0000, 1.0000, 1.0000]);
x = np.array([0.4621, 0.4621, 0.2803, 0.0682,0.0227,0.0076])
plt.plot(x, y, label="Selective Search 0")
y1 = np.array([0.7519, 0.8235, 1.0000, 1.0000,1.0000,1.000]);
x1 = np.array([0.7348, 0.6364, 0.2803, 0.1061, 0.0227,0.0076])
plt.plot(x1, y1, label="Selective Search 1")
y2 = np.array([0.7143, 0.7429, 0.8167,0.8660, 0.9091, 0.9661,1.0000,1,1.000,1.0000]);
x2 = np.array([0.7955, 0.7879, 0.7424,0.6364, 0.5303, 0.4318,0.2424,0.0909,0.0303,0.0227])
plt.plot(x2, y2, label="Selective Search 2")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Iou0.25_64win_srFeat_(1000+ objects imgs, 6000+ backgroud imgs) PR-curve')
plt.legend()
plt.ylim(0.4,1.1)
plt.xlim(0,1)


# y = np.array([0.6233, 0.6233, 0.6454, 0.6757, 0.7447]);
# x = np.array([0.6894, 0.6894, 0.6894, 0.5682, 0.2652])
# plt.plot(x, y, label="Selective Search 0")
# y1 = np.array([0.4810, 0.4739, 0.4927, 0.5973, 0.7400,0.8800, 1.0000]);
# x1 = np.array([0.7652, 0.7576, 0.7652, 0.6742, 0.5606,0.1667, 0.0303])
# plt.plot(x1, y1, label="Selective Search 1")
# y2 = np.array([0.4280, 0.4336, 0.4400,0.6190, 0.7478, 0.8718,1.0000]);
# x2 = np.array([0.8333, 0.8409, 0.8333,0.7879, 0.6515, 0.2576,0.0606])
# plt.plot(x2, y2, label="Selective Search 2")
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Iou0.25_64win_srFeat_(3000+ objects imgs, 6000+ backgroud imgs) PR-curve')
# plt.legend()
# plt.ylim(0.4,1.1)
# plt.xlim(0,1)

plt.show()