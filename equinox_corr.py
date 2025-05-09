import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import spearmanr
from dst_index import data_dst
from collections import defaultdict
matplotlib.use('TkAgg')


'''
    天平均地方时和太阳活动的关系
    计算天平均或者七天平均的函数
    之前的year数据有误  修正过
'''


def find_f107(year, doy):
    for row in solar_index:
        if year == row[0] and doy == row[1]:
            return row[-1]


def process_avg(interval=1):
    num_avg = []
    num_avg_min, num_avg_max = [], []
    lt_year, lt_doy, lt_month = [], [], []
    count = 0
    start_day = datetime.datetime(2003, 1, 1)
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        data = np.genfromtxt(file_path)
        for i in range(len(data) - 1):
            year, month, day, hour, minute = map(int, data[i][:5])
            end_day = datetime.datetime(year, month, day)
            if (end_day - start_day).days >= interval:
                num_avg.append(count)
                lt_year.append(start_day.year)
                lt_month.append(start_day.month)
                lt_doy.append(start_day.timetuple().tm_yday)
                start_day = end_day
                count = 1 if data[i][6] == 1 and data[i + 1][6] == 1 else 0
            else:
                if data[i][6] == 1 and data[i + 1][6] == 1:
                    count += 1
        if data[-1][6] == 1:
            count += 1

    num_avg.append(count)
    lt_year.append(start_day.year)
    lt_month.append(start_day.month)
    lt_doy.append(start_day.timetuple().tm_yday)
    return num_avg, lt_year, lt_month, lt_doy


path = "D:/pythonProject/GIMs/intensifications"
solar_file_path = "D:/pythonProject/GIMs/omni2_daily_JbsxhV8O1Q.lst"  # updated f107 2003-2023
solar_index = np.loadtxt(solar_file_path)

for i in range(len(solar_index)):
    if solar_index[i][3] == 999.9:
        solar_index[i][3] = solar_index[i-1][3]

if __name__ == '__main__':
    num_avg, lt_year, lt_month, lt_doy = process_avg()
    f107 = []
    for i in range(len(num_avg)):
        f107.append(find_f107(lt_year[i], lt_doy[i]))

    the_num_avg, the_f107 = [], []
    for i in range(len(num_avg)):
        if 1 <= lt_month[i] <= 2 or lt_month[i] == 12:
            the_num_avg.append(num_avg[i])
            the_f107.append(f107[i])
    print(len(the_num_avg), len(the_f107))
    corr, p_value = spearmanr(the_num_avg, the_f107)
    print(corr, p_value)