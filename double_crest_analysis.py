import os
import math
import datetime
import numpy as np
import pyIGRF

'''
    统计所有可视为双峰的双增强区（南北峰都至少一个台站）持续时间
'''


def double_crest_lasting_time():
    data = []
    double_crest = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        data_year = np.genfromtxt(file_path)
        data.extend(data_year.tolist())
    data = np.array(data)
    for i in range(len(data) - 2):
        if data[i][6] == 1 and data[i + 1][6] == 2 and data[i + 2][6] == 1:
            # lon1, lon2 = data[i][9], data[i + 1][9]
            # if min(np.abs(lon1 - lon2), 360 - np.abs(lon1 - lon2)) <= 45 and data[i][-1] >= 1 and data[i + 1][-1] >= 1:
            # 双峰持续过程中不考虑台站，如果开始持续的第一个双峰台站是0会在演化筛选掉
            # 不考虑双峰之间的距离，避免在双峰演化过程中会因为中间有TECmax间隔>45导致计时隔断
            double_crest.append(data[i])
            double_crest.append(data[i + 1])
            i += 1
    double_crest.append(data[-2])
    double_crest.append(data[-1])
    lasting_time, record_double = [], []
    last_time = 15
    start_record = datetime.datetime(2003, 1, 1, 2, 0)
    record_double.append(start_record)
    long_north = 173.5
    long_south = -179.5
    mlat_north = math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(14.5, 173.5, 300, 2003)[1])))
    mlat_south = math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(-12.5, -179.5, 300, 2003)[1])))
    for i in range(2, len(double_crest), 2):
        year, mon, day, hour, minute = map(int, double_crest[i][:5])
        now_record = datetime.datetime(year, mon, day, hour, minute)
        lat1, lon1 = double_crest[i][8:10]
        lat2, lon2 = double_crest[i + 1][8:10]
        mlat1 = math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(lat1, lon1, 300, year)[1])))
        mlat2 = math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(lat2, lon2, 300, year)[1])))
        lon_north = lon1 if mlat1 >= mlat2 else lon2
        lon_south = lon1 if mlat1 < mlat2 else lon2
        offset_lon_north = min(np.abs(lon_north - long_north), 360 - np.abs(lon_north - long_north))
        offset_lon_south = min(np.abs(lon_south - long_south), 360 - np.abs(lon_south - long_south))
        offset_lat_north = np.abs(max(mlat1, mlat2) - mlat_north)
        offset_lat_south = np.abs(min(mlat1, mlat2) - mlat_south)
        if (now_record - start_record).total_seconds() <= 960:
            if (offset_lon_north <= 15 and offset_lat_north <= 5) or (offset_lon_south <= 15 and offset_lat_south <= 5):  # 是否是上一个双峰的持续，加上地理位置的限制 两个峰都限制还是其中之一
                last_time += 15
        else:
            lasting_time.append(last_time)
            record_double.append(now_record)
            last_time = 15

        start_record = now_record
        long_north = lon_north
        long_south = lon_south
        mlat_north = max(mlat1, mlat2)
        mlat_south = min(mlat1, mlat2)

    lasting_time.append(last_time)
    return lasting_time, record_double


path = "D:/pythonProject/GIMs/intensifications"
lasting_time_double, record_double = double_crest_lasting_time()
# print(len(lasting_time_double), len(record_double))
# print(len([x for x in lasting_time_double if x == 15]))

