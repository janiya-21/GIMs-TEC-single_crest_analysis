import numpy as np
import seaborn as sns
import datetime
import matplotlib
from matplotlib import pyplot as plt
from single_crest_locations import data_single_crest
matplotlib.use('TkAgg')
'''
    更新data_single-crest的最后一列数据为Dst index
    统计单峰出现概率和地磁活动水平的关系
    地磁暴情况下单峰发生概率
'''

data_2022 = np.loadtxt('Dst_2003-01-01_2022-12-31_D.dat', dtype=str)
data_2023 = np.loadtxt('Dst_2023-01-01_2023-12-31_P.dat', dtype=str)
data = np.vstack([data_2022, data_2023])
result = []
for line in data:
    date, time, doy, dst = line[:4]
    year, month, day = date.split('-')
    hour = time.split(':')[0]
    result.append([int(year), int(month), int(day), int(hour), int(doy), int(dst)])

data_dst = np.array(result)
# data_dst = data_dst[:, -1]
# updated_data_single_crest = []   # 更新数据最后一列为Dst
# i, j = 0, 0
# while i < len(data_single_crest) and j < len(data_dst):
#     if np.array_equal(data_single_crest[i][:4], data_dst[j][:4]):
#         updated_row = np.append(data_single_crest[i], data_dst[j][-1])
#         updated_data_single_crest.append(updated_row)
#         i += 1
#     else:
#         j += 1
#
# updated_data_single_crest = np.array(updated_data_single_crest)


# dst = updated_data_single_crest[:, -1]
# little, weak, intense, storm = 0, 0, 0, 0
# for i in range(len(dst)):
#     if dst[i] > -50:
#         storm += 1
#     if dst[i] >= -25:
#         little += 1
#     elif dst[i] > -100:
#         weak += 1
#     else:
#         intense += 1


# dst_level = [little/300165, weak/300165, intense/300165]
# dst_storm = [storm/300165, 1 - storm/300165]
# print(dst_level)
# print(dst_storm)

# storm_dst, storm_date = [], []
# for i in range(len(updated_data_single_crest)):
#     if updated_data_single_crest[i][-1] <= -50:
#         year, month, day, hour, minute = map(int, updated_data_single_crest[i][:5])
#         storm_date.append(datetime.datetime(year, month, day, hour, minute))
#         storm_dst.append(updated_data_single_crest[i][-1])

# plt.figure(1)
# sns.histplot(dst, stat='density', color='cornflowerblue')
# plt.xlabel('Geomagnetic activity Dst(nT)')
# plt.ylabel('Probability Density')
#
# plt.figure(2)
# levels = ['Dst>=-25nT', '-100nT<Dst<-25nT', 'Dst<=-100nT']
# plt.bar(levels, dst_level, width=0.5, color=['lightsteelblue', 'cornflowerblue', 'midnightblue'])
# plt.xlabel('Geomagnetic activity')
# plt.ylabel('Normalized Number of Single Crest Events')
#
# plt.figure(3)
# plt.plot(dst, color='cornflowerblue')
#
# plt.figure(4)
# level_storm = ['Dst>-50nT', 'Dst<=-50nT']
# plt.bar(level_storm, dst_storm, width=0.4, color=['cornflowerblue', 'midnightblue'])
# plt.xlabel('Geomagnetic activity')
# plt.ylabel('Normalized Number of Single Crest Events')


# rec_peace, rec_m, rec_storm = [], [], []
# for i in range(len(updated_data_single_crest)):
#     if updated_data_single_crest[i][-1] > -50:
#         rec_peace.append(updated_data_single_crest[i][-4])
#     else:
#         rec_storm.append(updated_data_single_crest[i][-4])
#
#
# sns.histplot(rec_peace, stat='probability', fill=False, binwidth=0.0025, element='step', color='lightsteelblue')
# # sns.histplot(rec_m, stat='probability', fill=False, binwidth=0.005, element='step', color='cornflowerblue')
# sns.histplot(rec_storm, stat='probability', fill=False, binwidth=0.0025, element='step', color='midnightblue')
# plt.xlim(min(rec_storm), max(rec_storm))
# plt.xlabel('REC(GECU)')
# plt.ylabel('Probability Density')
# plt.legend(labels=['Dst>-50nT', 'Dst<=-50nT'])

# Dst指数极大年、极小年中的季节分布有没有不对称性质

plt.ion()
plt.show(block=True)




