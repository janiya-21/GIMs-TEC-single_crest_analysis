import math
import os
import numpy as np
import pyIGRF
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import matplotlib
from double_crest_analysis import lasting_time_double, record_double
matplotlib.use('TkAgg')


def local_time_timezone(ut_hour, longitude):
    if -7.5 <= longitude < 7.5:
        lt_hour = ut_hour
    elif 7.5 <= longitude < 172.5:
        for i in range(11):
            if (7.5 + 15 * i) <= longitude < 7.5 + 15 * (i + 1):
                lt_hour = (ut_hour + i + 1) % 24
    elif longitude >= 172.5 or longitude < -172.5:
        lt_hour = (ut_hour + 12) % 24
    else:
        for i in range(11):
            if (-7.5 - 15 * (i + 1)) <= longitude < (-7.5 - 15 * i):
                lt_hour = (ut_hour - i - 1) % 24
    return lt_hour


def is_evolution(year, mlat_single, lon_single, lat1, lon1, lat2, lon2, station_single, station1, station2):
    delta_lon = lambda a, b: min(abs(a - b), 360 - abs(a - b))
    if not (delta_lon(lon1, lon2) <= 45 and min(station_single, station1, station2) >= 1):
        return False, 0, 0, 0, 0

    calc_mlat = lambda lat, lon: math.degrees(
        math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(lat, lon, 300, year)[1])))
    )
    mlat1, mlat2 = calc_mlat(lat1, lon1), calc_mlat(lat2, lon2)
    diff1, diff2 = abs(mlat_single - mlat1), abs(mlat_single - mlat2)
    lon_min = delta_lon(lon_single, lon1 if diff1 <= diff2 else lon2)  # lon_min表示地磁纬度和单峰接近的双峰之一和单峰的经度距离
    lon_max = delta_lon(lon_single, lon2 if diff1 <= diff2 else lon1)
    lat_min, lat_max = sorted([diff1, diff2])
    return (
        lat_min <= 5 and lon_min <= 15 and lon_max <= 45,
        lon_min, lon_max, lat_min, lat_max
    )


def find_single_to_double():
    data = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        data_year = np.genfromtxt(file_path)
        data.extend(data_year.tolist())
    data = np.array(data)
    single_crest = []
    double_crest = []
    lon_min, lon_max = [], []
    lat_min, lat_max = [], []
    lasting_time_single = []
    start_record_single = datetime.datetime(2003, 1, 1, 2, 30)
    longitude = 158.5  # 第一条单峰的地磁纬度和经度
    magnetic_latitude = math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(-9.5, 158.5, 300, 2003)[1])))
    last_time = 15
    for i in range(len(data) - 3):
        if data[i][6] == 1 and data[i + 1][6] == 1:  # 当前是单峰
            year, mon, day, hour, minute = map(int, data[i][:5])
            now_record_single = datetime.datetime(year, mon, day, hour, minute)
            lat, lon = data[i][8:10]
            mlat = math.degrees(math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(lat, lon, 300, year)[1]))))

            mlat_diff = np.abs(magnetic_latitude - mlat)
            long_diff = min(np.abs(longitude - lon), 360 - np.abs(longitude - lon))
            if (now_record_single - start_record_single).total_seconds() <= 960 and (
                    mlat_diff <= 5 and long_diff <= 15):  # 单峰持续
                last_time += 15
            else:  # 当前是一个新的单峰
                last_time = 15

            if data[i + 2][6] == 2 and data[i + 3][6] == 1:  # 当前是单峰且下一个是双增强区
                lat1, lon1 = data[i + 1][8:10]
                lat2, lon2 = data[i + 2][8:10]
                flag, dlon_min, dlon_max, dlat_min, dlat_max = is_evolution(year, mlat, lon, lat1, lon1, lat2, lon2,
                                                                            data[i][-1], data[i + 1][-1],
                                                                            data[i + 2][-1])
                if flag:  # 可以视为演化
                    single_crest.append(data[i])
                    lasting_time_single.append(last_time)
                    double_crest.append(data[i + 1])
                    lon_min.append(dlon_min)
                    lon_max.append(dlon_max)
                    lat_min.append(dlat_min)
                    lat_max.append(dlat_max)

            start_record_single = now_record_single
            magnetic_latitude = mlat
            longitude = lon

        else:
            last_time = 15

    return single_crest, lasting_time_single, lon_min, lon_max, lat_min, lat_max, double_crest


path = "D:/pythonProject/GIMs/intensifications"
single_crest, lasting_time_single, lon_min, lon_max, lat_min, lat_max, double_crest = find_single_to_double()
print(f"single to double: {len(single_crest)}")
count = 0
for line in single_crest:
    if line[1] >= 22 or line[1] < 2:
        count += 1
print(count)

# 查找演化前单峰持续时间>=45min，演化前后三个增强区距离都很接近的记录，后续对应的双峰演化时间
# count = []
# for i in range(len(single_crest)):
#     if lasting_time_single[i] >= 45 and lon_max[i] <= 45 and lat_max[i] <= 10:
#         year, mon, day, hour, minute = map(int, double_crest[i][:5])
#         count.append((datetime.datetime(year, mon, day, hour, minute)))
# print(len(count))
# time_duration_map = dict(zip(record_double, lasting_time_double))
# lasting_time_double_after_evolution = []
# for entry in count:
#     if entry in time_duration_map:
#         print(time_duration_map[entry])


# 查找演化后的双峰对应的持续时间
time_duration_map = dict(zip(record_double, lasting_time_double))
lasting_time_double_after_evolution = []
for entry in double_crest:
    year, mon, day, hour, minute = map(int, entry[:5])
    now_record = datetime.datetime(year, mon, day, hour, minute)
    if now_record in time_duration_map:
        lasting_time_double_after_evolution.append(time_duration_map[now_record])
# print(f"matched double_crest: {len(lasting_time_double_after_evolution)} ")
sns.histplot(lasting_time_double_after_evolution, element="step", binwidth=15, fill=False, stat='count')
plt.xlim(15, max(lasting_time_double_after_evolution))


# heatmap
single = np.clip(lasting_time_single, 15, max(lasting_time_single))
double = np.clip(lasting_time_double_after_evolution, 15, max(lasting_time_double_after_evolution))
bins_x = np.unique(np.concatenate([np.array([15, 45]), np.arange(60, 300, 60), np.array([300, max(lasting_time_single)])]))
bins_y = np.unique(np.concatenate([np.array([15, 45]), np.arange(60, 300, 60), np.array([300, max(lasting_time_double_after_evolution)])]))
hist, xedges, yedges = np.histogram2d(
    single,
    double,
    bins=[bins_x, bins_y]
)
mask = hist.T == 0
plt.figure()
ax = sns.heatmap(
    hist.T,
    mask=mask,
    annot=True,
    fmt=".0f",
    cmap="GnBu",
    cbar_kws={'label': 'Number of Events'},
    xticklabels=[f"{int(x)}" for x in xedges[:-1]],
    yticklabels=[f"{int(y)}" for y in yedges[:-1]],
    linewidths=0.5
)
ax.invert_yaxis()
ax.set_xticks(np.arange(len(xedges[:-1])) + 0.5)
ax.set_yticks(np.arange(len(yedges[:-1])) + 0.5)
ax.set_xlabel('Single Crest Duration (minutes)', fontsize=14)
ax.set_ylabel('Double Crest Duration (minutes)', fontsize=14)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.grid(which='major', axis='both', linestyle='--', alpha=0.75)


plt.ion()
plt.tight_layout()
plt.show(block=True)
