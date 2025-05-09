import numpy as np
import os
import pyIGRF
import math
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
matplotlib.use('TkAgg')
'''
    筛选双增强区中的双峰：经度差值<=45°，地磁纬度分布在磁赤道异侧，特殊情况分布在磁赤道同侧，后续可以重点关注这部分双峰的是时间分布
    对于经度差值>45°的双增强区，从dataset无法判断两个增强区是否存在重叠，不能简单判断为单峰
    双峰REC之和、平均值的概率分布、地磁纬度差概率分布，磁赤道同侧双峰的经度差分布
    two-intensification 经度差概率分布，地方时分布（<=45°一个峰值，>45°两个波峰）
'''


def lt(hour, minute, lon1, lon2):

    def local_t(hour, minute, lon):
    #     total_minutes = hour * 60 + minute + lon / 15 * 60
    #     local_hour = int(total_minutes // 60) % 24
    #     local_minute = int(total_minutes % 60) / 60
    #     return local_hour, local_minute
        if -7.5 <= lon < 7.5:
            return hour + minute / 60
        elif 7.5 <= lon < 172.5:
            for i in range(11):
                if (7.5 + 15 * i) <= lon < 7.5 + 15 * (i + 1):
                    return (hour + i + 1) % 24 + minute / 60
        elif lon >= 172.5 or lon < -172.5:
            return (hour + 12) % 24 + minute / 60
        else:
            for i in range(11):
                if (-7.5 - 15 * (i + 1)) <= lon < (-7.5 - 15 * i):
                    return (hour - i - 1) % 24 + minute / 60
    local_time1 = local_t(hour, minute, lon1)
    local_time2 = local_t(hour, minute, lon2)
    return local_time1, local_time2


def calculate_mcl(lat1, lat2):
    return (max(lat1, lat2) - min(lat1, lat2)) / 2


# mag_lat1 * mag_lat2 > 0 分布在地磁赤道同侧
# def sort_double_crest(lon1, lon2, lat1, lat2, year):
#     def calculate_mag_lat(lat, lon):
#         value = pyIGRF.igrf_value(lat, lon, 300, year)
#         mag_lat = math.degrees(math.atan(0.5 * math.tan(math.radians(value[1]))))
#         return mag_lat
#
#     mag_lat1 = calculate_mag_lat(lat1, lon1)
#     mag_lat2 = calculate_mag_lat(lat2, lon2)
#     offset = np.abs(lon1 - lon2) if lon1 * lon2 >= 0 else min(np.abs(lon1 - lon2), 360-np.abs(lon1 - lon2))
#     is_double_crest = (offset <= 45)
#     return is_double_crest, mag_lat1, mag_lat2, offset


def sort_double_crest1(lon1, lon2):
    offset = min(np.abs(lon1 - lon2), 360-np.abs(lon1 - lon2))
    is_double_crest = offset <= 45
    return is_double_crest


def read_files():
    crest1, crest2 = [], []
    count_double = 0
    local_time = []
    offset_lon, offset_lat = [], []
    lon, mcl = [], []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        data = np.genfromtxt(file_path)
        for i in range(len(data)-2):
            if data[i][6] == 1 and data[i+1][6] == 2 and data[i+2][6] == 1:  # 两个增强区
                year = int(data[i][0])
                mon = int(data[i][1])
                # if (2008 <= year <= 2009 or 2018 <= year <= 2020) and (1 <= mon <= 2 or 11 <= mon <= 12):
                lat1, lon1 = data[i][8:10]
                lat2, lon2 = data[i + 1][8:10]
                hour, minute = data[i][3:5]
                # is_double_crest, mag_lat1, mag_lat2, offset = sort_double_crest(lon1, lon2, lat1, lat2, year)
                is_double_crest = sort_double_crest1(lon1, lon2)
                if is_double_crest:
                    # count_double += 1
                    local_time1, local_time2 = lt(hour, minute, lon1, lon2)
                    # crest1.append(data[i])
                    # crest2.append(data[i+1])
                    local_time.append(local_time1)
                    local_time.append(local_time2)
                    # local_time.append(local_hour2 + local_minute2)
                    # offset_lon.append(offset)
                    # offset_lat.append(np.abs(mag_lat1 - mag_lat2))
                    # print(f"{mag_lat1}, {mag_lat2}")
                    # lon.append(lon1)
                    # lon.append(lon2)
                    # mcl.append(calculate_mcl(mag_lat1, mag_lat2))

        if data[-2][6] == 1 and data[-1][6] == 2:
            year = int(data[-1][0])
            mon = int(data[-1][1])
            # if (2008 <= year <= 2009 or 2018 <= year <= 2020) and (1 <= mon <= 2 or 11 <= mon <= 12):
            lat1, lon1 = data[-2][8:10]
            lat2, lon2 = data[-1][8:10]
            hour, minute = data[-2][3:5]
            # is_double_crest, mag_lat1, mag_lat2, offset = sort_double_crest(lon1, lon2, lat1, lat2, year)
            is_double_crest = sort_double_crest1(lon1, lon2)
            if is_double_crest:
                # count_double += 1
                local_time1, local_time2 = lt(hour, minute, lon1, lon2)
                # crest1.append(data[-2])
                # crest2.append(data[-1])
                local_time.append(local_time1)
                local_time.append(local_time2)
                # local_time.append(local_hour2 + local_minute2)
                # offset_lon.append(offset)
                # offset_lat.append(np.abs(mag_lat1 - mag_lat2))
                # print(f"{mag_lat1}, {mag_lat2}")
                # lon.append(lon1)
                # mcl.append(calculate_mcl(mag_lat1, mag_lat2))

    print("double crest:", count_double)
    return np.array(crest1), np.array(crest2), np.array(local_time), offset_lon, mcl, lon


path = "D:/pythonProject/GIMs/intensifications"
crest1, crest2, local_time_double, offset_lon, mcl, lon = read_files()
# rec_sum = []
# rec_max, rec_min = [], []


# for i in range(len(crest1)):
#     year = crest1[i][0]
#     rec_sum.append((crest1[i][-3]+crest2[i][-3]) / 2.0)
#     if year == 2003 or year == 2013 or year == 2014 or year == 2022:
#         rec_max.append((crest1[i][-3]+crest2[i][-3]) / 2.0)
#     elif year == 2008 or year == 2009 or year == 2018 or year == 2019:
#         rec_min.append((crest1[i][-3]+crest2[i][-3]) / 2.0)

# sns.histplot(rec_sum, stat='probability', fill=False, binwidth=0.002, element='step', color='black')
# sns.histplot(rec_max, stat='probability', fill=False, binwidth=0.002, element='step', color='orange')
# sns.histplot(rec_min, stat='probability', fill=False, binwidth=0.002, element='step', color='cornflowerblue')
# plt.legend(['All Data', 'Solar Maximum years', 'Solar Minimum years'])
# plt.xlim(min(rec_sum), max(rec_sum))
# plt.xlabel('REC(GECU)')
# plt.ylabel('Probability Density')

# sns.histplot(local_time, binwidth=0.25, fill=False, stat='density', element='step')
# sns.histplot(offset_lon, bins=15, fill=False, stat='density', element='step')
# sns.histplot(offset_lat, fill=False, stat='density', element='step')
# plt.xlabel('Absolute Magnetic Latitude Difference')
# plt.xlabel('Local Time')
# plt.ylabel('Probability Density')
# plt.xlim(0, max(offset_lat))
# plt.xlim([0, 24])
# plt.xlim(0, 45)

# plt.figure()
# plt.scatter(offset_lon, offset_lat, s=3)
# plt.xlabel('Absolute Longitude Difference')
# plt.ylabel('Absolute Magnetic Latitude Difference')
# plt.xlim(0, 45)
# plt.ylim(0, 50)

# plt.figure()
# plt.title('longitudinal distribution of two-intensifications 3h apart')
# sns.histplot(lon, stat='density', element='step', fill=False)
# plt.xlim(-180, 180)
# plt.xlabel('Geographic Longitude')
# plt.ylabel('Probability Density')


# local_time_bins = (local_time * 60 // 15) * 15 / 60
# data = pd.DataFrame({'longitude': lon, 'local time': local_time_bins, 'Value': mcl})
# data['longitude_bin'] = (data['longitude'] // 3) * 3
# data['local_time_bin'] = data['local time'].round(2)
# heatmap_data = data.groupby(['longitude_bin', 'local_time_bin'])['Value'].mean().reset_index()
# plt.figure()
# plt.hexbin(heatmap_data['longitude_bin'], heatmap_data['local_time_bin'],
#            C=heatmap_data['Value'], cmap='coolwarm', zorder=2)
# cb = plt.colorbar()
# cb.set_label('MCL')
# plt.xlabel('Geographic Longitude')
# plt.ylabel('Local Time')
# # plt.title("Jun Solstice")
# plt.xlim(-180, 180)
# plt.ylim(8, 22)

# data = pd.DataFrame({'longitude': lon, 'local time': local_time, 'mcl': mcl})
# data['local_time_bin'] = (data['local time'] * 60 // 15) * 15 / 60
# data['longitude_bin'] = (data['longitude'] // 3) * 3
# heatmap_data = data.groupby(['longitude_bin', 'local_time_bin'])['mcl'].mean().reset_index()
#
# plt.figure()
# heatmap_pivot = heatmap_data.pivot_table(index='local_time_bin', columns='longitude_bin', values='mcl')
# sns.heatmap(heatmap_pivot, cmap='coolwarm', cbar_kws={'label': 'MCL'}, zorder=2)
# plt.xlabel('Geographic Longitude (°)')
# plt.ylabel('Local Time')
# # plt.xlim(-180, 180)
# plt.grid(True, linestyle='--', alpha=0.75, zorder=0)
# plt.gca().spines['top'].set_visible(True)
# plt.gca().spines['right'].set_visible(True)
# plt.gca().spines['left'].set_visible(True)
# plt.gca().spines['bottom'].set_visible(True)
# plt.ion()
# plt.show(block=True)

