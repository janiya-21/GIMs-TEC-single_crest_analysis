import math
import os
import numpy as np
import pandas as pd
import pyIGRF
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import copy
from dst_index import data_dst
from scipy.stats import pearsonr, spearmanr
# from double_crest import local_time_double
matplotlib.use('TkAgg')


def local_time_timezone(ut_hour, long):
    if -7.5 <= long < 7.5:
        return ut_hour
    elif 7.5 <= long < 172.5:
        for i in range(11):
            if (7.5 + 15 * i) <= long < 7.5 + 15 * (i + 1):
                return (ut_hour + i + 1) % 24
    elif long >= 172.5 or long < -172.5:
        return (ut_hour + 12) % 24
    else:
        for i in range(11):
            if (-7.5 - 15 * (i + 1)) <= long < (-7.5 - 15 * i):
                return (ut_hour - i - 1) % 24


def parse_group_data(group):
    """
        长持续时间单峰信息
        return mlat、longitude sequence;local_time;track boundary (easternmost,westernmost,northernmost,southernmost)
               avg_dst within duration;duration;lat_avg_speed;lon_avg_speed
        同一个增强区，单峰的经度位置可能大幅度波动，计算漂移速度困难

    """
    def plot_mlat_lt(duration_min):
        """增强区地磁纬度和地方时序列 地方时梯度 不考虑世界时情况下mlat和LT分布情况"""
        # matplotlib.use('Agg')
        if duration >= duration_min:
            d_lt = np.gradient(lt)
            dd_lt = np.gradient(d_lt)
            label = end_time.strftime('%Y-%m-%d')
            fig, (ax1, ax3, ax4) = plt.subplots(1, 3, figsize=(12, 5))
            ax1.plot(times_sequence, mlats, marker='o', color='#1f77b4')
            ax1.tick_params(axis='y', labelcolor='#1f77b4')
            ax2 = ax1.twinx()
            ax2.plot(times_sequence, lt, marker='o', color='#ff7f0e')
            ax2.tick_params(axis='y', labelcolor='#ff7f0e')
            ax2.set_ylim(10, 19)
            ax3.plot(times_sequence, d_lt, marker='o', color='#1f77b4')
            ax3.tick_params(axis='y', labelcolor='#1f77b4')
            ax3.axhline(y=0.25, color='#1f77b4', linestyle='--')
            ax3.axhline(y=0, color='#1f77b4', linestyle='--')
            ax5 = ax3.twinx()
            ax5.plot(times_sequence, dd_lt, marker='o', color='#ff7f0e')
            ax5.tick_params(axis='y', labelcolor="#ff7f0e")
            ax5.axhline(y=0, color='#ff7f0e', linestyle='--')
            ax3.plot(times_sequence, d_lt, marker='o')
            ax4.scatter(lt, mlats, s=6)
            for ax in [ax1, ax3, ax4]:
                ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
                if not ax == ax4:
                    ax.set_xlabel("Time Sequence (15min)")
            ax1.set_ylabel("Magnetic Latitude (°)")
            ax2.set_ylabel("Local Time")
            ax3.set_ylabel("LT Gradient")
            ax5.set_ylabel("LT Second Order Gradient")
            ax4.set_xlabel("Local Time")
            ax4.set_ylabel("Magnetic Latitude (°)")
            plt.suptitle(f"{label}  {duration / 60:.2f}h  ΔLT/ΔUT:{((lt[-1] - lt[0]) / (duration / 60)):.2f}")
            plt.tight_layout()
            save_dir = "C:/Users/zp129/Desktop/毕设/数据分析结果/单峰漂移/MLAT_LT/avg_mlat_speed_15"
            filename = f"{label}_{int(duration / 60)}h.png".replace(" ", "_")
            plt.savefig(os.path.join(save_dir, filename), dpi=300, facecolor='white')
            plt.close()
    lt, dst = [], []
    mlats, longs = [], []
    times_sequence = []
    start_time = end_time = None
    for i, line in enumerate(group):
        year, month, day, hour, minute = map(int, line[:5])
        mlat, lon = line[8:10]
        mlats.append(mlat)
        longs.append(lon)

        total_minutes = hour * 60 + minute + lon / 15 * 60
        lt.append(int(total_minutes // 60) % 24 + int(total_minutes % 60) / 60)
        # lts.append(local_time_timezone(hour, lon) + minute / 60)

        dst.append(dst_dict.get((year, month, day, hour), np.nan))
        if i == 0:
            start_time = datetime.datetime(year, month, day, hour, minute)
        times_sequence.append((datetime.datetime(year, month, day, hour, minute) - start_time).total_seconds() / 900)
        end_time = datetime.datetime(year, month, day, hour, minute)

    duration = (end_time - start_time).total_seconds() / 60 + 15
    delta_mlat = mlats[-1] - mlats[0]
    delta_long = min(abs(longs[-1] - longs[0]), 360 - abs(longs[-1] - longs[0]))
    lon_east, lon_west = max(longs), min(longs)
    mlat_north, mlat_south = max(mlats), min(mlats)
    lat_avg_speed = delta_mlat / (duration / 60)
    lon_avg_speed = delta_long / (duration / 60)
    # plot_mlat_lt(240)
    # far corresponding_lt
    max_abs_index = np.argmax(np.abs(mlats))
    min_abs_index = np.argmin(np.abs(mlats))
    return np.array(longs), np.array(mlats), np.array(lt), min(dst), lt[max_abs_index], lt[min_abs_index]


def heatmap_lat_long_offset(min_duration):
    mlat_offset, long_offset = [], []
    for i in range(len(duration)):
        if duration[i] >= min_duration:
            mlat_offset.append(lat_offset[i])
            long_offset.append(lon_offset[i])
    lat_bin_size = 2
    lon_bin_size = 5
    lat_min, lat_max = np.floor(min(mlat_offset)), np.ceil(max(mlat_offset))
    lon_min, lon_max = np.floor(min(long_offset)), np.ceil(max(long_offset))
    lat_bins = np.arange(lat_min, lat_max + lat_bin_size, lat_bin_size)
    lon_bins = np.arange(lon_min, lon_max + lon_bin_size, lon_bin_size)
    hist, xedges, yedges = np.histogram2d(long_offset, mlat_offset, bins=[lon_bins, lat_bins])
    mask = hist.T == 0
    plt.figure(figsize=(12, 5))
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
    ax.set_xlabel('Longitude Offset (°)', fontsize=12)
    ax.set_ylabel('Magnetic Latitude Offset (°)', fontsize=12)
    plt.title(f'min_duration={min_duration}min (Lon bin={lon_bin_size}°, MLAT bin={lat_bin_size}°)', fontsize=14)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.grid(which='major', axis='both', linestyle='--', alpha=0.75)
    plt.tight_layout()
    plt.show(block=True)


def heatmap_duration_Dst():
    # distribution Duration vs Dst_avg
    # pearson_corr, p_pearson = pearsonr(dst_group, duration_group)
    # spearman_corr, p_spearman = spearmanr(dst_group, duration_group)
    # print(f"Pearson相关系数: {pearson_corr:.3f} (p={p_pearson:.3e})")
    # print(f"Spearman相关系数: {spearman_corr:.3f} (p={p_spearman:.3e})")
    # bins [ )
    print(f"Dst_avg_min:{min(dst_group)}, Dst_avg_max:{max(dst_group)}")
    duration_bins = np.unique(np.concatenate([np.array([15, 60]), np.arange(60, 240, 60), np.array([240, 600])]))
    dst_bins = np.unique(np.concatenate([np.array([-350, -200]), np.arange(-200, -100, 50), np.arange(-100, 70, 20)]))
    counts, xedges, yedges = np.histogram2d(dst_group, duration, bins=[dst_bins, duration_bins])
    mask = counts.T == 0
    row_sums = counts.T.sum(axis=1, keepdims=True)
    percentage_matrix = (counts.T / row_sums * 100).round(2)
    annot_matrix = np.where(counts.T == 0, "", np.char.add(percentage_matrix.astype(str), ""))
    plt.figure()
    sns.heatmap(
        percentage_matrix,
        mask=mask,
        annot=annot_matrix,
        fmt="",
        cmap="GnBu",
        cbar=True,
        cbar_kws={'label': 'Percentage (%)'},
        xticklabels=[f"{int(x)}" for x in xedges[:-1]],
        yticklabels=[f"{int(y)}" for y in yedges[:-1]],
    )
    plt.xticks(np.arange(len(xedges[:-1])) + 0.5)
    plt.yticks(np.arange(len(yedges[:-1])) + 0.5)
    plt.xlabel("Minimum Dst(nT)", fontsize=12)
    plt.ylabel("Duration(min)", fontsize=12)
    plt.gca().invert_yaxis()
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)
    plt.grid(which='major', axis='both', linestyle='--', alpha=0.75)


def year_category(y):
    solar_max = {2003, 2013, 2014}
    solar_min = {2008, 2009, 2018, 2019}
    if y in solar_max:
        return "solar_max"
    elif y in solar_min:
        return "solar_min"
    else:
        return "solar_mid"


def month_category(m):
    if m in [5,6,7,8]:
        return "Jun.Solstice"
    elif m in [1,2,11,12]:
        return "Dec.Solstice"
    else:
        return "Equinoxes"


def parse_avg_mlat(mlat_lt_datalist):
    result = defaultdict(dict)
    for yc in ["solar_max", "solar_min", "solar_mid"]:
        for season in ["Jun.Solstice", "Dec.Solstice", "Equinoxes"]:
            lt_means = []
            for lt in sorted(mlat_lt_datalist[yc][season].keys()):
                values = mlat_lt_datalist[yc][season][lt]
                lt_means.append((lt, np.mean(values) if values else None, len(values)))
            result[f"{yc}_{season}"] = {
                "lt": [x[0] for x in lt_means],
                "mlat_avg": [x[1] for x in lt_means],
                "event counts": [x[2] for x in lt_means]
            }
    print(pd.DataFrame.from_dict(result, orient='index'))
    return result


def parse_avg_mla_allyears(mlat_lt_datalist):
    result = defaultdict(dict)
    for season in ["Jun.Solstice", "Dec.Solstice", "Equinoxes"]:
        lt_means = []
        for lt in sorted(mlat_lt_datalist[season].keys()):
            values = mlat_lt_datalist[season][lt]
            lt_means.append((lt, np.mean(values) if values else None, len(values)))
        result[f"{season}"] = {
            "lt": [x[0] for x in lt_means],
            "mlat_avg": [x[1] for x in lt_means],
            "event counts": [x[2] for x in lt_means]
        }
    print(pd.DataFrame.from_dict(result, orient='index'))
    return result


def plot_mlat_lt_variation(result, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    for ax, yc in zip(axes, ["solar_max", "solar_mid", "solar_min"]):
        ax.set_title(year_labels[yc], fontsize=12)
        for season in ["Jun.Solstice", "Dec.Solstice", "Equinoxes"]:
            key = f"{yc}_{season}"
            data = result[key]
            x = np.array(data['lt'])
            y = np.array(data['mlat_avg'])
            valid = ~np.isnan(y)
            ax.plot(x[valid], y[valid], **season_styles[season], label=season.replace('.', ' '))
        ax.set_xticks(np.arange(8, 23, 2))
        ax.set_xlim(8, 22)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, alpha=0.5, which='both', linestyle='--')
        if ax == axes[0]:
            ax.set_ylabel('Average Magnetic Latitude (°)', fontsize=12)
        else:
            ax.tick_params(labelleft=True)
        if ax == axes[1]:
            ax.set_xlabel('Local Time (h)', fontsize=12)
    plt.suptitle(title, fontsize=14, y=0.9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.05))
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show(block=True)


def plot_mlat_lt_variation_allyears(result, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    for season in ["Jun.Solstice", "Dec.Solstice", "Equinoxes"]:
        key = f"{season}"
        data = result[key]
        x = np.array(data['lt'])
        y = np.array(data['mlat_avg'])
        valid = ~np.isnan(y)
        ax.plot(x[valid], y[valid], **season_styles[season], label=season.replace('.', ' '))
    ax.set_xticks(np.arange(8, 23, 2))
    ax.set_xlim(8, 22)
    if title == 'Southern Crest':
        ax.set_ylim(-25, 0)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.5, which='both', linestyle='--')
    ax.set_ylabel('Average Magnetic Latitude (°)', fontsize=12)
    ax.set_xlabel('Local Time (h)', fontsize=12)
    plt.title(title, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.05))
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show(block=True)


path = "D:/pythonProject/GIMs/intensifications_single"
single_crest_simple = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    data = np.genfromtxt(file_path)
    single_crest_simple.extend(data.tolist())
single_crest_simple = np.array(single_crest_simple)
print(f"number of single crest from 2003 to 2023: {len(single_crest_simple)}")
longitude = single_crest_simple[:, 9]
last_time = 15
duration, time_record = [], []
ratios = []
lat_offset, lon_offset = [], []
start_lat_set, end_lat_set = [], []
start_lon_set, end_lon_set = [], []
groups, current_group = [], []

former_time = datetime.datetime(2003, 1,  1, 8, 45)
long = 75.5
mag_lat = math.degrees(math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(-7.5, 75.5, 300, 2003)[1]))))
time_record.append(former_time)
start_long = long
start_lat = mag_lat
start_lat_set.append(mag_lat)
start_lon_set.append(long)
current_single_record = copy.deepcopy(single_crest_simple[0])
current_single_record[8] = mag_lat
current_group.append(current_single_record)

avg_mlat_vs_lt = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
avg_mlat_vs_lt_n = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
avg_mlat_vs_lt_s = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
avg_mlat_vs_lt_allyear = defaultdict(lambda: defaultdict(list))
avg_mlat_vs_lt_allyear_n = defaultdict(lambda: defaultdict(list))
avg_mlat_vs_lt_allyear_s = defaultdict(lambda: defaultdict(list))
for row in single_crest_simple:
    year, mon, day, hour, minute = map(int, row[:5])
    lat, lon = row[8:10]
    now_time = datetime.datetime(year, mon, day, hour, minute)

    total_minutes = hour * 60 + minute + lon / 15 * 60
    lt = int(total_minutes // 60) % 24 + int(total_minutes % 60) / 60
    # lt = local_time_timezone(hour, lon) + int(total_minutes % 60) / 60

    # calculate magnetic latitude
    value = pyIGRF.igrf_value(lat, lon, 300, int(year))
    m_lat = math.degrees(math.atan(0.5 * math.tan(math.radians(value[1]))))
    current_single_record = copy.deepcopy(row)
    current_single_record[8] = m_lat

    # avg_mlat_vs_lt[year_category(year)][month_category(mon)][math.floor(lt)].append(m_lat)
    # avg_mlat_vs_lt_allyear[month_category(mon)][math.floor(lt)].append(m_lat)
    # if m_lat >= 0:
    #     avg_mlat_vs_lt_n[year_category(year)][month_category(mon)][math.floor(lt)].append(m_lat)
    #     avg_mlat_vs_lt_allyear_n[month_category(mon)][math.floor(lt)].append(m_lat)
    # else:
    #     avg_mlat_vs_lt_s[year_category(year)][month_category(mon)][math.floor(lt)].append(m_lat)
    #     avg_mlat_vs_lt_allyear_s[month_category(mon)][math.floor(lt)].append(m_lat)

    # mag_lat,long 前个单峰的地磁纬度、经度
    offset_long = min(abs(lon - long), 360 - abs(lon - long))
    offset_mlat = abs(m_lat - mag_lat)
    if now_time - former_time <= datetime.timedelta(minutes=30) and offset_long <= 45 and offset_mlat <= 5:  # intensification duration criteria 45; single_crest migration 15
        last_time += (now_time - former_time).total_seconds() / 60
        current_group.append(current_single_record)
    else:
        # former single-crest
        duration.append(last_time)
        lat_offset.append(abs(mag_lat - start_lat))
        lon_offset.append(min(abs(long - start_long), 360 - abs(long - start_long)))
        end_lat_set.append(mag_lat)
        end_lon_set.append(long)
        groups.append(list(current_group))
        current_group = [current_single_record]

        # a now one
        start_lat = m_lat  # 当前单峰是一个新持续单峰的起始记录
        start_long = lon
        start_lat_set.append(m_lat)
        start_lon_set.append(lon)
        last_time = 15
        time_record.append(now_time)

    former_time = now_time
    mag_lat = m_lat
    long = lon

duration.append(last_time)
lat_offset.append(abs(mag_lat - start_lat))
lon_offset.append(min(abs(long - start_long), 360 - abs(long - start_long)))
end_lat_set.append(mag_lat)
end_lon_set.append(long)
groups.append(list(current_group))
assert len(groups) == len(duration)
assert len(lat_offset) == len(duration)
assert len(lon_offset) == len(duration)
assert len(time_record) == len(duration)
# heatmap_lat_long_offset(45)
print(f"single groups: {len(duration)}, duration avg: {np.mean(duration)},"
      f" std: {np.std(duration, ddof=1)}min, max: {max(duration)}min")
print(f"duration>=45min:{len([x for x in duration if x >= 45])},"
      f"duration>=60min:{len([x for x in duration if x >= 60])},"
      f"duration>=120min:{len([x for x in duration if x >= 120])},"
      f"duration>=240min:{len([x for x in duration if x >= 240])},")


# season/solar year difference
# month_short, month_long = [], []
# for i in range(len(duration)):
#     if duration[i] >= 240:
#         month_long.append(time_record[i].month)
#     elif duration[i] <= 45:
#         month_short.append(time_record[i].month)
# sns.histplot(month_short, binwidth=1, label='<=45min', stat='probability', fill=False, element='step')
# sns.histplot(month_long, binwidth=1, label='>=240min', stat='probability', element='step', fill=False)
# plt.legend(frameon=False)
# plt.xlabel('Month')
# plt.ylabel('Probability Density')
# plt.xlim(1, 12)
# plt.xlim(2003, 2023)
# plt.xticks(np.arange(2003, 2024, 2))

# Dst_avg vs Duration
dst_dict = {}
for d in data_dst:
    key = tuple(map(int, d[:4]))
    dst_dict[key] = d[-1]
dst_group, lat_speed_group = [], []
derivative_lt_ut, LT_max_mlat, LT_min_mlat = [], [], []
for i, group in enumerate(groups):
    longs, mlats, lt, dst_avg, lt_max_mlat, lt_min_mlat = parse_group_data(group)
    dst_group.append(dst_avg)
    if dst_avg <= -100 and duration[i] >= 240:
        print(time_record[i], duration[i])
    # if duration[i] >= 60:
    #     dlt_dut = (lt[-1] - lt[0]) / (duration[i] / 60)
    #     derivative_lt_ut.append(dlt_dut)
    #     if dlt_dut >= 0.7:
    #         lat_speed_group.append((mlats[-1] - mlats[0]) / (duration[i] / 60))
    #         LT_max_mlat.append(lt_max_mlat)
    #         LT_min_mlat.append(lt_min_mlat)

heatmap_duration_Dst()
# assert len(lat_speed_group) == len([x for x in derivative_lt_ut if 0.7 <= x])
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# sns.histplot(derivative_lt_ut, stat='probability', element='step', fill=False, ax=ax1, zorder=2)
# ax1.set_title(f"Duration>=1h  N={len(derivative_lt_ut)}", fontsize=14)
# print(f"0.75-1.25:{len([x for x in derivative_lt_ut if 0.75 <= x <= 1.25])},"
#       f"{len([x for x in derivative_lt_ut if 0.75 <= x <= 1.25]) / len(derivative_lt_ut):.2f}")
# print(f"0.7-1.3:{len([x for x in derivative_lt_ut if 0.7 <= x <= 1.3])},"
#       f"{len([x for x in derivative_lt_ut if 0.7 <= x <= 1.3]) / len(derivative_lt_ut):.2f}")
# print(f"0.7-:{len([x for x in derivative_lt_ut if 0.7 <= x])},"
#       f"{len([x for x in derivative_lt_ut if 0.7 <= x]) / len(derivative_lt_ut):.2f}")
# print(f"0.5-:{len([x for x in derivative_lt_ut if 0.5 <= x])},"
#       f"{len([x for x in derivative_lt_ut if 0.5 <= x]) / len(derivative_lt_ut):.2f}")
# ax1.set_xlabel("ΔLT/ΔUT", fontsize=12)

# sns.histplot(lat_speed_group, stat='probability', binwidth=0.5, element='step', fill=False, ax=ax2, zorder=2)
# ax2.set_title(f"ΔLT/ΔUT >= 0.7  N={len(lat_speed_group)}", fontsize=14)
# ax2.set_xlabel("Average MLAT Speed (°/h)", fontsize=12)

# sns.histplot(LT_max_mlat, stat='probability', binwidth=0.25, element='step', fill=False, ax=ax2, zorder=2)
# sns.histplot(LT_min_mlat, stat='probability', binwidth=0.25, element='step', fill=False, ax=ax2, zorder=2)
# ax2.legend(['max|MLAT|', 'min|MLAT|'], frameon=False)
# ax2.set_title(f"ΔLT/ΔUT >= 0.7  N={len(lat_speed_group)}", fontsize=14)
# ax2.set_xlabel("Local Time (h)", fontsize=12)

# for ax in [ax1, ax2]:
#     ax.set_ylabel("Probability Density", fontsize=12)
#     ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

# MLat vs LT  MLAT motion
# season_styles = {
#     "Jun.Solstice": {'color': '#ff7f0e', 'marker': 'o'},
#     "Dec.Solstice": {'color': '#1f77b4', 'marker': 'o'},
#     "Equinoxes": {'color': '#2ca02c', 'marker': 'o'}
# }
# year_labels = {
#     "solar_max": "Solar Maximum",
#     "solar_mid": "Moderate Solar Activity Years",
#     "solar_min": "Solar Minimum"
# }
# 细分太阳活动
# result_all = parse_avg_mlat(avg_mlat_vs_lt)
# result_n = parse_avg_mlat(avg_mlat_vs_lt_n)
# result_s = parse_avg_mlat(avg_mlat_vs_lt_s)
# plot_mlat_lt_variation(result_all, "All Data")
# plot_mlat_lt_variation(result_n, "Northern Crest")
# plot_mlat_lt_variation(result_s, "Southern Crest")
# 所有年份平均
# result_all = parse_avg_mla_allyears(avg_mlat_vs_lt_allyear)
# result_n = parse_avg_mla_allyears(avg_mlat_vs_lt_allyear_n)
# result_s = parse_avg_mla_allyears(avg_mlat_vs_lt_allyear_s)
# plot_mlat_lt_variation_allyears(result_all, "All Data")
# plot_mlat_lt_variation_allyears(result_n, "Northern Crest")
# plot_mlat_lt_variation_allyears(result_s, "Southern Crest")



# sns.histplot(ratios, stat='probability', element='step', fill=False, binwidth=1)
# threshold_long_offset = 10
# threshold_lat_offset = 5
# threshold_ratios =
# threshold_curvature =

# for i in range(len(duration)):
#     if (lon_offset[i] >= threshold_long_offset or lat_offset[i] >= threshold_lat_offset) and duration[i] >= 45:
#         ratios.append(lon_offset[i] / lat_offset[i])
# plt.xlim(0, max(ratios))
# print(f"sum:{len(ratios)}, "
#       f"ratios>=5:{len([x for x in ratios if x >= 5])},{len([x for x in ratios if x >= 5]) / len(ratios) * 100:.2f},"
#       f"ratios>=10:{len([x for x in ratios if x >= 10])},{len([x for x in ratios if x >= 10]) / len(ratios) * 100:.2f},"
#       f"ratios<=2:{len([x for x in ratios if x <= 2])},{len([x for x in ratios if x <= 2]) / len(ratios) * 100:.2f},"
#       f"ratios<=1:{len([x for x in ratios if x <= 1])},{len([x for x in ratios if x <= 1]) / len(ratios) * 100:.2f}")

# classification
# categories = {
#     'stay': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': []},
#     'trans': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': []},
#     'e': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': []},
#     'p': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': []}
# }
# MIN_DURATION = 120
# MAX_DURATION = 540
# MAX_STAY_DRIFT = 2.0
# MAX_STAY_LAT = 2.5
# trans_doy = []
# for i in range(len(lasting_time)):
#     if lasting_time[i] < MIN_DURATION or lasting_time[i] > MAX_DURATION:
#         continue

    # start = start_lat_set[i]
    # end = end_lat_set[i]
    # lat_diff = abs(end - start)
    # max_abs = max(abs(start), abs(end))
    # if (lat_diff <= MAX_STAY_DRIFT) or (max_abs <= MAX_STAY_LAT):
    #     cat = 'stay'
    # elif (min(abs(start), abs(end)) > MAX_STAY_LAT) and (start * end < 0):
    #     cat = 'trans'
    # elif abs(end) < abs(start):
    #     cat = 'e'
    # else:
    #     cat = 'p'

    # categories[cat]['lat_offset'].append(lat_offset[i])
    # categories[cat]['start'].append(start)
    # categories[cat]['end'].append(end)
    # categories[cat]['start_long'].append(start_lon_set[i])
    # categories[cat]['end_long'].append(end_lon_set[i])
    # categories[cat]['lat_speed'].append(lat_avg_speed[i])

    # if cat == 'trans':   # case study
        # categories[cat]['month'].append(lasting_time_year[i])
        # lon_avg_speed = (end_lon_set[i] - start_lon_set[i]) / (lasting_time[i] / 60)
        # if lon_avg_speed <= -10:
        # if abs(start_lon_set[i] - end_lon_set[i]) >= lasting_time[i] / 4 or abs(start_lon_set[i] - end_lon_set[i]) >= 35:
        # if abs(start_lon_set[i] - end_lon_set[i]) > 15 or lat_diff < 7:
        #     continue
        # print(f"{lasting_time_year[i]}    {lat_offset[i]:.2f}    "
        #         f"{start_lat_set[i]:.2f}   {end_lat_set[i]:.2f}   {start_lon_set[i]}    {end_lon_set[i]}"
        #           f"    {int(lasting_time[i])}        doy:{lasting_time_doy[i]}     {lat_avg_speed[i]:.2f}      {lon_avg_speed:.2f}")


plt.ion()
# plt.tight_layout()
plt.show(block=True)

