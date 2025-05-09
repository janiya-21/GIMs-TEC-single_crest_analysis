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
from scipy.stats import pearsonr, spearmanr, linregress
from equinox_corr import solar_index, find_f107
from dst_index import data_dst
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
import statsmodels.api as sm
# from double_crest import local_time_double
matplotlib.use('TkAgg')


def pandas_monthly_avg():
    '''
        单峰持续时间和太阳活动周期是否相关？
        持续时间月平均值
    '''
    df = pd.DataFrame({
        'year': lasting_time_year,
        'month': lasting_time_doy,
        'time': lasting_time
    })
    full_idx = pd.MultiIndex.from_product(
        [df.year.unique(), range(1, 13)],
        names=['year', 'month']
    )
    result = (
        df.groupby(['year', 'month'])
        ['time'].mean()
        .reindex(full_idx)
        .reset_index()
    )
    dates = pd.to_datetime(result['year'].astype(str) + '-' + result['month'].astype(str), format='%Y-%m').to_numpy()
    averages = result['time'].to_numpy()
    plt.figure(figsize=(12, 5))
    plt.plot(dates, averages, marker='o', zorder=2)
    plt.xlim(min(dates), max(dates))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)


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
    """长持续时间单峰信息"""
    times, longs, mag_lats, lts = [], [], [], []
    for line in group:
        parts = line.strip().split()
        year, month, day, hour, minute = map(int, parts[:5])
        lat = float(parts[8])
        lon = float(parts[9])
        value = pyIGRF.igrf_value(lat, lon, 300, year)
        total_minutes = hour * 60 + minute + lon / 15 * 60
        local_hour = int(total_minutes // 60) % 24
        local_minute = int(total_minutes % 60) / 60

        longs.append(lon)
        times.append(datetime.datetime(year, month, day, hour, minute))
        mag_lats.append(math.degrees(math.atan(0.5 * math.tan(math.radians(value[1])))))
        # lts.append(local_time_timezone(hour, lon) + minute / 60)
        lts.append(local_hour + local_minute)
    return np.array(times), np.array(longs), np.array(mag_lats), np.array(lts), datetime.datetime(year, month, day)


def analyze_trend(times, lons, mag_lats):
    """计算时空变化趋势"""
    t_hours = [(t - times[0]).total_seconds() / 3600 for t in times]

    lon_slope, _, _, _, _ = linregress(t_hours, lons)
    lon_drift_rate = lon_slope

    lat_slope, _, _, _, _ = linregress(t_hours, mag_lats)
    lat_change_rate = lat_slope
    stats = {
        'lon_range': (lons.min(), lons.max()),
        'lat_range': (mag_lats.min(), mag_lats.max()),
        'lon_std': np.std(lons),
        'lat_std': np.std(mag_lats),
        'corr_coef': np.corrcoef(lons, mag_lats)[0, 1]
    }
    return lon_drift_rate, lat_change_rate, stats, t_hours


def plot_spatial_trend(group_id, t_hours, lons, mag_lats, lts, stats, times):
    """绘制时空演变图"""
    plt.figure(figsize=(12, 4))
    plt.suptitle(f"{times[0]}-{times[-1]}")
    plt.subplot(1, 3, 1)
    plt.plot(t_hours, lons, 'bo-', markersize=4)
    plt.xlim(0, 9)
    plt.xlabel('Time')
    plt.ylabel('Longitude')
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    # plt.title(f'Group {group_id}: Lon Drift (Rate={stats["lon_drift_rate"]:.2f} deg/h)')

    plt.subplot(1, 3, 2)
    plt.plot(t_hours, mag_lats, 'ro-', markersize=4)
    plt.xlim(0, 9)
    plt.xlabel('Time')
    plt.ylabel('Geomagnetic Latitude')
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    # plt.title(f'Lat Change (Rate={stats["lat_change_rate"]:.2f} deg/h)')

    plt.subplot(1, 3, 3)
    plt.plot(t_hours, lts, 'yo-', markersize=4)
    plt.xlim(0, 9)
    plt.ylim(10, 19)
    plt.xlabel('Time')
    plt.ylabel('Local Time')
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.show(block=True)


def process_avg(interval=27):
    num_avg = []
    lt_year, lt_doy, lt_month, lt_hour = [], [], [], []
    count = 0
    start_day = datetime.datetime(2003, 1, 6)
    for row in single_crest_simple:
        y, m, d, h = map(int, row[:4])
        end_day = datetime.datetime(y, m, d)
        if (end_day - start_day).days >= interval:  # interval=27 27-Day avg
        # if (end_day - start_day).total_seconds() >= interval * 3600:  # hourly num
            num_avg.append(count / 27)
            # num_avg.append(count)
            lt_year.append(start_day.year)
            lt_month.append(start_day.month)
            lt_doy.append(start_day.timetuple().tm_yday)
            lt_hour.append(start_day.hour)
            start_day += datetime.timedelta(days=27)
            count = 1
        else:
            count += 1

    end_day = datetime.datetime(2023, 12, 31)
    num_avg.append(count / (end_day - start_day).days)
    # num_avg.append(count)
    lt_year.append(start_day.year)
    lt_month.append(start_day.month)
    lt_doy.append(start_day.timetuple().tm_yday)
    lt_hour.append(start_day.hour)
    return num_avg, lt_year, lt_month, lt_doy, lt_hour


def plot_heatmap():
    data_num = pd.DataFrame({'Year': outliers_year, 'DayOfYear': outliers_doy, 'Value': outliers})
    heatmap_data = data_num.pivot(index='Year', columns='DayOfYear', values='Value')
    fig, ax = plt.subplots(figsize=(30, 5))
    sns.heatmap(heatmap_data, cmap='coolwarm', cbar_kws={'label': 'Count', 'orientation': 'vertical', 'fraction': 0.05},
                alpha=0.8, zorder=2, ax=ax)
    ax.set_xticks(np.arange(1, 381, 20))
    ax.set_xticklabels(np.arange(1, 381, 20), rotation=0)
    plt.xlabel('Day of Year')
    plt.ylabel('Year')
    plt.grid(True, zorder=0, alpha=0.75, linestyle='--')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, top=0.85)
    plt.ion()
    plt.show(block=True)


# def plot_number_of_events_seasonal(season_counter):
#     years = sorted(season_counter["June Solstice"].keys())
#     jun_sol = np.array([season_counter["June Solstice"].get(y, 0) for y in years])
#     equinox = np.array([season_counter["Equinoxes"].get(y, 0) for y in years])
#     dec_sol = np.array([season_counter["December Solstice"].get(y, 0) for y in years])
#     plt.figure(figsize=(10, 4))
#     x_year = np.arange(min(years), max(years) + 1)
#     plt.plot(x_year, jun_sol)
#     plt.plot(x_year, equinox)
#     plt.plot(x_year, dec_sol)
#     plt.legend(['June Solstice', 'Equinoxes', 'December Solstice'], frameon=False)
#     plt.xlabel('Year')
#     plt.ylabel('season count')
#     plt.xticks(np.arange(min(years), max(years) + 1, 1))
#     plt.xlim(min(years), max(years))
#     plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
#     plt.ion()
#     plt.show(block=True)


def plot_seasonal_counts():
    fig, axes = plt.subplots(4, 1, figsize=(10, 16))
    seasons = ['March Equinox', 'June Solstice', 'September Equinox', 'December Solstice']
    region_counters = {
        'North': season_counter_n,
        'South': season_counter_s,
        'Magnetic Equator': season_counter_e
    }
    for i, season in enumerate(seasons):
        ax = axes[i]
        all_years = set()
        for counter in region_counters.values():
            if season in counter:
                all_years.update(counter[season].keys())

        min_year, max_year = min(all_years), max(all_years)
        x_year = np.arange(min_year, max_year + 1)

        for region_name, counter in region_counters.items():
            season_data = counter.get(season, {})
            y_values = [season_data.get(year, 0) for year in x_year]
            ax.plot(x_year, y_values, label=region_name)

        ax.set_title(f'{season}', fontsize=10)
        ax.set_xticks(x_year[::1])
        ax.tick_params(labelbottom=(i == 3))
        ax.set_xlim(min_year, max_year)
        ax.set_ylabel('Count')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(frameon=False)
        if i == 3:
            ax.set_xlabel('Year')

    plt.show(block=True)


def plot_pd():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    sns.histplot(longitude, ax=axs[0, 0], stat='density', fill=False, bins=120, element='step', color='black')
    sns.histplot(long_min, ax=axs[0, 0], stat='density', fill=False, bins=120, element='step')
    sns.histplot(long_max, ax=axs[0, 0], stat='density', fill=False, bins=120, element='step')
    axs[0, 0].set_xlabel('Geographic Longitude(°)',fontsize=14)
    axs[0, 0].set_ylabel('Probability Density',fontsize=14)
    axs[0, 0].set_xlim(-180, 180)
    axs[0, 0].legend(['All Data', 'Solar min', 'Solar max'], frameon=False, fontsize=12)

    sns.histplot(magnetic_latitude, ax=axs[0, 1], stat='density', fill=False, bins=70, element='step', color='black')
    sns.histplot(lat_min, ax=axs[0, 1], stat='density', fill=False, bins=70, element='step')
    sns.histplot(lat_max, ax=axs[0, 1], stat='density', fill=False, bins=70, element='step')
    print("magnetic latitude:\t")
    print(f"avg: {np.mean(magnetic_latitude):.2f}, std: {np.std(magnetic_latitude, ddof=1):.2f}")
    print(f"north_crest: {len([x for x in magnetic_latitude if x > 2.5])}")
    print(f"south_crest: {len([x for x in magnetic_latitude if x < -2.5])}")
    print(f"equator_crest: {len([x for x in magnetic_latitude if -2.5 <= x <= 2.5])}")
    axs[0, 1].set_xlabel('Geomagnetic Latitude(°)',fontsize=14)
    axs[0, 1].set_ylabel('Probability Density',fontsize=14)
    axs[0, 1].set_xlim(-35, 35)

    sns.histplot(local_time, ax=axs[1, 0], stat='probability', fill=False, binwidth=0.25, element='step', color='black')
    sns.histplot(local_time_double, ax=axs[1, 0], stat='probability', fill=False, binwidth=0.25, element='step', color='red')
    print("local_time:\t")
    print(f"avg: {np.mean(local_time):.2f}, std: {np.std(local_time, ddof=1):.2f}")
    axs[1, 0].set_xlabel('Local Time',fontsize=14)
    axs[1, 0].set_ylabel('Probability Density',fontsize=14)
    axs[1, 0].set_xlim(0, 24)
    axs[1, 0].legend(['Single Crest', 'Double Crest'], frameon=False,fontsize=12)

    sns.histplot(month, ax=axs[1, 1], stat='probability', fill=False, bins=60, element='step', color='black')
    sns.histplot(month_min, ax=axs[1, 1], stat='probability', fill=False, bins=60, element='step')
    sns.histplot(month_max, ax=axs[1, 1], stat='probability', fill=False, bins=60, element='step')
    axs[1, 1].set_xlabel('Month',fontsize=14)
    axs[1, 1].set_ylabel('Probability Density',fontsize=14)
    axs[1, 1].set_xlim(1, 13)

    plt.ion()
    plt.tight_layout()
    plt.show(block=True)


def plot_lat_offset():
    '''持续单峰  起始位置和终止位置地磁纬度偏差'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # plt.suptitle(f'MIN_DURATION = {MIN_DURATION} min, MAX_DURATION = {MAX_DURATION} min', fontsize=14)
    plt.suptitle(f'{MIN_DURATION}-{MAX_DURATION} min', fontsize=14)
    color = ['#1f77b4','#ff7f0e', '#2ca02c', '#d62728']
    sns.histplot(categories['stay']['lat_offset'], element='step', fill=False, binwidth=1, ax=ax1, stat='probability', color=color[0])
    sns.histplot(categories['trans']['lat_offset'], element='step', fill=False, binwidth=1, ax=ax1, stat='probability', color=color[1])
    sns.histplot(categories['e']['lat_offset'], element='step', fill=False, binwidth=1, ax=ax1, stat='probability', color=color[2])
    sns.histplot(categories['p']['lat_offset'], element='step', fill=False, binwidth=1, ax=ax1, stat='probability', color=color[3])
    ax1.set_xlabel("Geomagnetic Latitude Offset (°)", fontsize=14)
    ax1.set_ylabel("Probability Density", fontsize=14)
    ax1.set_xlim(0, 25)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.scatter(categories['stay']['start'], categories['stay']['end'], s=2, zorder=2, alpha=0.5, color=color[0])
    ax2.scatter(categories['trans']['start'], categories['trans']['end'], s=2, zorder=2, alpha=0.5, color=color[1])
    ax2.scatter(categories['e']['start'], categories['e']['end'], s=2, zorder=2, alpha=0.5, color=color[2])
    ax2.scatter(categories['p']['start'], categories['p']['end'], s=2, zorder=2, alpha=0.5, color=color[3])
    ax2.legend(['Stationary', 'Trans-Equatorial', 'Equatorward', 'Poleward'], frameon=False, fontsize=12)
    x = np.linspace(-45, 30, 100)
    ax2.plot(x, x, linestyle='--', color='black', zorder=1)
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-30, 30)
    ax2.set_xlabel('Initial Geomagnetic Latitude (°)', fontsize=14)
    ax2.set_ylabel('Terminal Geomagnetic Latitude (°)', fontsize=14)
    ax2.grid(True, linestyle='--', zorder=0, alpha=0.5)
    plt.tight_layout()
    plt.show(block=True)


def plot_trans_track():
    '''case study: Trans-Equatorial tracking'''
    longitude_sequence = np.arange(-180, 190, 10)
    mag_lat_equator = []
    for lon in longitude_sequence:
        mag_lat = []
        for year in range(2003, 2025):
            value = pyIGRF.igrf_value(0, lon, 300, year)
            mag_lat.append(math.degrees(math.atan(0.5 * math.tan(math.radians(value[1])))))
        mag_lat_equator.append(np.average(mag_lat))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    plot_config = {
        'line': {'color': 'k', 'alpha': 0.3, 'linewidth': 0.8},
        'start': {'s': 5, 'color': 'k', 'label': 'Start'},
        'end_summer': {'s': 5, 'color': '#ff7f0e', 'label': 'End_Jun.solstice'},
        'end_winter': {'s': 5, 'color': '#1f77b4', 'label': 'End_Dec.solstice'},
        'end_equinox': {'s': 5, 'color': 'lightgreen', 'label': 'End_Equinoxes'},
    }
    seasons = {
        'winter': {'months': {1, 2, 11, 12}, 'config': 'end_winter'},
        'equinox': {'months': {3, 4, 9, 10}, 'config': 'end_equinox'},
        'summer': {'months': {5, 6, 7, 8}, 'config': 'end_summer'}
    }
    legend_config = {'loc': 'lower left', 'frameon': False, 'fontsize': 10}
    northward = {'start_lon': [], 'start_lat': [], 'end_lon': [], 'end_lat': [], 'month': []}
    southward = {'start_lon': [], 'start_lat': [], 'end_lon': [], 'end_lat': [], 'month': []}
    for slon, slat, elon, elat, month in zip(categories['trans']['start_long'],
                                      categories['trans']['start'],
                                      categories['trans']['end_long'],
                                      categories['trans']['end'],
                                      categories['trans']['month']):
        if slat < elat:
            northward['start_lon'].append(slon)
            northward['start_lat'].append(slat)
            northward['end_lon'].append(elon)
            northward['end_lat'].append(elat)
            northward['month'].append(month)
        else:
            southward['start_lon'].append(slon)
            southward['start_lat'].append(slat)
            southward['end_lon'].append(elon)
            southward['end_lat'].append(elat)
            southward['month'].append(month)

    for slon, slat, elon, elat in zip(northward['start_lon'], northward['start_lat'],
                                      northward['end_lon'], northward['end_lat']):
        ax1.plot([slon, elon], [slat, elat], **plot_config['line'])
    ax1.scatter(northward['start_lon'], northward['start_lat'], **plot_config['start'], zorder=2)
    # ax1.scatter(northward['end_lon'], northward['end_lat'], **plot_config['end'], zorder=2)
    for season in seasons:
        mask = [m in seasons[season]['months'] for m in northward['month']]
        lons = [lon for lon, m in zip(northward['end_lon'], mask) if m]
        lats = [lat for lat, m in zip(northward['end_lat'], mask) if m]
        ax1.scatter(lons, lats,  **plot_config[seasons[season]['config']], zorder=2)
    ax1.set_title("Northward", fontsize=14)

    for slon, slat, elon, elat in zip(southward['start_lon'], southward['start_lat'],
                                      southward['end_lon'], southward['end_lat']):
        ax2.plot([slon, elon], [slat, elat], **plot_config['line'])
    sc_start = ax2.scatter([], [], **plot_config['start'])
    sc_end_winter = ax2.scatter([], [], **plot_config['end_winter'])
    sc_end_summer = ax2.scatter([], [], **plot_config['end_summer'])
    sc_end_equinox = ax2.scatter([], [], **plot_config['end_equinox'])
    ax2.scatter(southward['start_lon'], southward['start_lat'], **plot_config['start'], zorder=2)
    # ax2.scatter(southward['end_lon'], southward['end_lat'], **plot_config['end'], zorder=2)
    for season in seasons:
        mask = [m in seasons[season]['months'] for m in southward['month']]
        lons = [lon for lon, m in zip(southward['end_lon'], mask) if m]
        lats = [lat for lat, m in zip(southward['end_lat'], mask) if m]
        ax2.scatter(lons, lats,  **plot_config[seasons[season]['config']], zorder=2)
    ax2.set_title("Southward", fontsize=14)

    ax1.annotate(f'N = {len(northward["start_lon"])}', xy=(0.05, 0.9),
                 xycoords='axes fraction', ha='left', fontsize=12)
    ax2.annotate(f'N = {len(southward["start_lon"])}', xy=(0.05, 0.9),
                 xycoords='axes fraction', ha='left', fontsize=14)

    for ax in [ax1, ax2]:
        ax.plot(longitude_sequence, mag_lat_equator, linestyle='--', alpha=0.5, color='k', zorder=1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-20, 15)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax2.legend(handles=[sc_start, sc_end_winter, sc_end_equinox, sc_end_summer], labels=['Start', 'End_Dec.solstice', 'End_Equinoxes', 'End_Jun.solstice'], **legend_config)
    ax2.set_xlabel("Longitude (°)", fontsize=14)
    fig.supylabel("Geomagnetic Latitude (°)", fontsize=14)


path = "D:/pythonProject/GIMs/intensifications_single"
f107_27_path = "D:/pythonProject/GIMs/omni2_27day_KJyS_3dXJo.lst"
Dst_27_path = "D:/pythonProject/GIMs/omni2_27day_Ih9CuO_93l.lst"
solar_index_27 = np.loadtxt(f107_27_path)
dst_index_27 = np.loadtxt(Dst_27_path)
solar_index_27 = solar_index_27[:, -1]
dst_index_27 = dst_index_27[:, -1]

single_crest_simple = []
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    data = np.genfromtxt(file_path)
    single_crest_simple.extend(data.tolist())
single_crest_simple = np.array(single_crest_simple)
print(f"number of single crest from 2003 to 2023: {len(single_crest_simple)}")

# basic characteristics
longitude = single_crest_simple[:, 9]
magnetic_latitude = []
local_time, month = [], []
lat_max, lat_min, long_min, long_max, month_min, month_max = [], [], [], [], [], []
last_time = 15
lasting_time, lasting_time_year, lasting_time_doy = [], [], []
start_record = datetime.datetime(2003, 1,  1, 8, 45)
long = 75.5
mag_lat = math.degrees(math.atan(0.5 * math.tan(math.radians(pyIGRF.igrf_value(-7.5, 75.5, 300, 2003)[1]))))
season_counter_n = defaultdict(lambda: defaultdict(int))
season_counter_s = defaultdict(lambda: defaultdict(int))
season_counter_e = defaultdict(lambda: defaultdict(int))
lt_summer, lt_e, lt_winter = [], [], []
time_record, mlat_avg = [], []
time_record.append(start_record)
lat_offset, lat_avg_speed = [], []  # 持续单峰起始位置和终止位置地磁纬度变化，地磁纬度平均速度
start_long, start_lat = long, mag_lat
start_lat_set, end_lat_set = [], []
start_lon_set, end_lon_set = [], []
start_lat_set.append(mag_lat)
start_lon_set.append(long)
for row in single_crest_simple:
    year, mon, day, hour, minute = map(int, row[:5])
    lat, lon = row[8:10]

    # month
    # start_time = datetime.datetime(year, 1, 1, 0, 0)
    end_time = datetime.datetime(year, mon, day, hour, minute)
    # end_year = datetime.datetime(year + 1, 1, 1, 0, 0)
    # seconds = (end_time - start_time).total_seconds()
    # total_seconds = (end_year - start_time).total_seconds()
    # month.append(seconds / (total_seconds / 12) + 1)

    # calculate magnetic latitude
    value = pyIGRF.igrf_value(lat, lon, 300, int(year))
    m_lat = math.degrees(math.atan(0.5 * math.tan(math.radians(value[1]))))
    magnetic_latitude.append(m_lat)

    # if m_lat < -2.5:
    #     if 1 <= mon <= 2 or 11 <= mon <= 12:
    #         lt_winter.append(lon)
    #     elif 5 <= mon <= 8:
    #         lt_summer.append(lon)
    #     else:
    #         lt_e.append(lon)

    # calculate local time by timezone
    # local_hour = local_time_timezone(hour, lon)
    # local_time.append(local_hour + minute / 60)

    # seasonal distribution of number of events
    # season = (
    #     "June Solstice" if 5 <= mon <= 8 else
    #     "March Equinox" if 3 <= mon <= 4 else
    #     "September Equinox" if 9 <= mon <= 10 else
    #     "December Solstice"
    # )
    # if m_lat > 2.5:
    #     season_counter_n[season][year] += 1
    # elif m_lat < -2.5:
    #     season_counter_s[season][year] += 1
    # else:
    #     season_counter_e[season][year] += 1

    # lasting time
    # mag_lat,long 上一条单峰的地磁纬度、经度;m_lat,lon 当前单峰的地磁纬度
    offset_long = min(abs(lon - long), 360 - abs(lon - long))
    offset_mlat = abs(m_lat - mag_lat)
    if end_time - start_record <= datetime.timedelta(minutes=30) and offset_long <= 45 and offset_mlat <= 5:
        last_time += (end_time - start_record).total_seconds() / 60
    else:
        lasting_time.append(last_time)  # 上一个单峰的持续时间
        lasting_time_year.append(start_record)  # 上一个持续单峰结束的时间记录
        lasting_time_doy.append(start_record.timetuple().tm_yday)

        lat_offset.append(abs(mag_lat - start_lat))
        lat_avg_speed.append((mag_lat - start_lat) / (last_time / 60))
        end_lat_set.append(mag_lat)
        end_lon_set.append(long)

        start_lat = m_lat  # 当前单峰是一个新持续单峰的起始记录
        start_lat_set.append(m_lat)
        start_lon_set.append(lon)
        last_time = 15
    start_record = end_time
    mag_lat = m_lat
    long = lon

    # difference between solar max and solar min
    # if year == 2003 or 2013 <= year <= 2014:
    #     lat_max.append(magnetic_latitude[-1])
    #     month_max.append(month[-1])
    #     long_max.append(lon)
    # if 2008 <= year <= 2009 or 2018 <= year <= 2019:
    #     lat_min.append(magnetic_latitude[-1])
    #     month_min.append(month[-1])
    #     long_min.append(lon)

    # hourly avg mlat lombscargle
    # if (end_time - start_record).total_seconds() >= 3600:
    #     mlat_avg.append(np.mean(magnetic_latitude))
    #     time_record.append(end_time)
    #     magnetic_latitude.clear()
    #     start_record = end_time
    # magnetic_latitude.append(m_lat)


# lomb-scargle magnetic latitude
# mlat_avg.append(np.mean(magnetic_latitude))
# time_hour = [(x - time_record[0]).total_seconds() / 3600 for x in time_record]  # 小时为单位
# frequency, power = LombScargle(time_hour, mlat_avg).autopower(maximum_frequency=0.25)
# peaks, _ = find_peaks(power)
# peak_powers = power[peaks]
# peak_frequencies = frequency[peaks]
# top_indices = np.argsort(peak_powers)[-50:][::-1]
# top_peak_powers = peak_powers[top_indices]
# top_peak_frequencies = peak_frequencies[top_indices]
# for i in range(len(top_peak_powers)):
#     print(f"Power:{top_peak_powers[i]:.4f}, Frequency:{1 / top_peak_frequencies[i] / 24.0:.4f}day")
# plt.figure(figsize=(10, 2))
# plt.plot(frequency, power)
# plt.ylim(0, 0.3)
# plt.xlim([1e-6, 0.1])
# plt.xlabel('Frequency (per hour)')
# plt.xscale('log')
# plt.ylabel('Power')
# plt.tight_layout()


# plot_pd()
# plot_seasonal_counts()

lasting_time.append(last_time)
lasting_time_year.append(start_record)
lasting_time_doy.append(start_record.timetuple().tm_yday)
lat_offset.append(abs(mag_lat - start_lat))
lat_avg_speed.append((mag_lat - start_lat) / (last_time / 60))
end_lat_set.append(mag_lat)
end_lon_set.append(long)
print(f"unique single: {len(lasting_time)}, lasting time avg: {np.mean(lasting_time)},"
      f" std: {np.std(lasting_time, ddof=1)}, max: {max(lasting_time)}")

# classification
categories = {
    'stay': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': []},
    'trans': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': [], 'month': []},
    'e': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': []},
    'p': {'lat_offset': [], 'start': [], 'end': [], 'start_long': [], 'end_long': [], 'lat_speed': []}
}
MIN_DURATION = 120
MAX_DURATION = 540
MAX_STAY_DRIFT = 2.0
MAX_STAY_LAT = 2.5
trans_doy = []
for i in range(len(lasting_time)):
    if lasting_time[i] < MIN_DURATION or lasting_time[i] > MAX_DURATION:
        continue

    start = start_lat_set[i]
    end = end_lat_set[i]
    lat_diff = abs(end - start)
    max_abs = max(abs(start), abs(end))
    if (lat_diff <= MAX_STAY_DRIFT) or (max_abs <= MAX_STAY_LAT):
        cat = 'stay'
    elif (min(abs(start), abs(end)) > MAX_STAY_LAT) and (start * end < 0):
        cat = 'trans'
    elif abs(end) < abs(start):
        cat = 'e'
    else:
        cat = 'p'

    categories[cat]['lat_offset'].append(lat_offset[i])
    categories[cat]['start'].append(start)
    categories[cat]['end'].append(end)
    categories[cat]['start_long'].append(start_lon_set[i])
    categories[cat]['end_long'].append(end_lon_set[i])
    categories[cat]['lat_speed'].append(lat_avg_speed[i])

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

# plot_lat_offset()
# plot_trans_track()
# pandas_monthly_avg()
# plt.figure()
# sns.histplot(lasting_time, binwidth=15, element='step', fill=False, color='black')
# plt.xlim(15, max(lasting_time) + 1)
# plt.xlabel('Duration (min)')
# f107_lasting, lasting_time_ = [], []
# for i in range(len(lasting_time)):
#     if lasting_time[i] > 210:
#         lasting_time_.append(lasting_time[i])
#         f107_lasting.append(find_f107(lasting_time_year[i], lasting_time_doy[i]))
# plt.scatter(lasting_time_, f107_lasting, s=2)
# plt.figure()
# sns.histplot(f107_lasting, fill=False, bins=120, element='step')
# plt.xlim(min(f107_lasting), max(f107_lasting))
# plt.xlabel("F107")
# corr, p_value = spearmanr(lasting_time_, f107_lasting)
# print(f"corr_lasting_time_vs_F10.7: {corr}")


# correlation: daily number of events vs F10.7
# daily_avg, the_year, the_month, doy, the_hour = process_avg()
# print(f"27 day avg: {daily_avg}")
# std = np.std(daily_avg, ddof=1)
# mean = np.mean(daily_avg)
# print(f"daily number max: {max(daily_avg)}, min: {min(daily_avg)} ")
# print(f"daily number avg: {mean}, std: {std} ")
# f107 = []
# outliers, outliers_year, outliers_doy = [], [], []
# for i in range(len(daily_avg)):
#     outliers_year.append(the_year[i])
#     outliers_doy.append(doy[i])
#     outliers.append(daily_avg[i]) if abs(daily_avg[i] - mean) > std else outliers.append(np.nan)
#     # outliers.append(daily_avg[i])
#     f107.append(find_f107(the_year[i], doy[i]))

# num_jun, solar_jun, dst_jun = [], [], []
# num_dec, solar_dec, dst_dec = [], [], []
# num_e, solar_e, dst_e = [], [], []
# for i in range(len(daily_avg)):
    # 60-120, 120-240, 240-300
    # if 120 <= doy[i] < 240:
    #     solar_jun.append(solar_index_27[i])
    #     dst_jun.append(dst_index_27[i])
    #     num_jun.append(daily_avg[i])
    # elif 60 <= doy[i] < 120 or 240 <= doy[i] < 300:
    #     solar_e.append(solar_index_27[i])
    #     dst_e.append(dst_index_27[i])
    #     num_e.append(daily_avg[i])
    # else:
    #     solar_dec.append(solar_index_27[i])
    #     dst_dec.append(dst_index_27[i])
    #     num_dec.append(daily_avg[i])

# 回归方程
# def solve_equation(x, y):
#     slope, intercept, r, *_ = linregress(x, y)
#     return f"y = {slope:.3f}x + {intercept:.1f} R²={r**2:.2f}"

# plt.figure()
# sns.regplot(x=solar_dec, y=num_dec, scatter_kws={'s': 5}, line_kws={'lw': 1.5})
# sns.regplot(x=solar_jun, y=num_jun, scatter_kws={'s': 5}, line_kws={'lw': 1.5})
# sns.regplot(x=solar_e, y=num_e, scatter_kws={'s': 5, 'color': 'k'}, line_kws={'lw': 1.5, 'color': 'k'})
# plt.text(max(solar_index_27)*0.55, max(daily_avg), s=f"Dec.Solstice {solve_equation(solar_dec, num_dec)}", color='#1f77b4', ha='left', va='top', fontsize=12)
# plt.text(max(solar_index_27)*0.55, max(daily_avg)*0.9, s=f"Jun.Solstice {solve_equation(solar_jun, num_jun)}", color='#ff7f0e', ha='left', va='top', fontsize=12)
# plt.text(max(solar_index_27)*0.55, max(daily_avg)*0.8, s=f"Equinoxes {solve_equation(solar_e, num_e)}", color='k', ha='left', va='top', fontsize=12)
# plt.xlim(min(solar_index_27)-5, max(solar_index_27)+5)
# plt.ylim(-1, max(daily_avg)+1)
# plt.xlabel("27-Day Averaged Solar Index F10.7 (SFU)", fontsize=12)
# plt.ylabel("27-Day Averaged Number of Event", fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
# plt.title("Solar Flux Correlation", fontsize=14)
# 多元线性回归建模
# df = pd.DataFrame({
#     'N_single': num_e,
#     'F10.7': solar_e,
#     'Dst': dst_e,
# })
# df["F10.7_Dst"] = df["F10.7"] * df["Dst"]  # 多重共线性
# 中心化处理
# df["F10.7_centered"] = df["F10.7"] - df["F10.7"].mean()
# df["Dst_centered"] = df["Dst"] - df["Dst"].mean()
# df["F10.7_Dst_centered"] = df["F10.7_centered"] * df["Dst_centered"]
# df["F10.7_centered_sq"] = df["F10.7_centered"] ** 2
# X_centered = sm.add_constant(df[["F10.7_centered", "Dst_centered", "F10.7_Dst_centered"]])
# X_centered = sm.add_constant(df[["F10.7"]])
# model_centered = sm.OLS(df["N_single"], X_centered).fit()
# print(model_centered.summary())
# df["F10.7_sq"] = df["F10.7"] ** 2  # 非线性项
# X = df[["F10.7", "Dst", "F10.7_Dst"]]  # 交互项
# X = df[["F10.7", "Dst"]]  # 无交互
# X = sm.add_constant(X)
# y = df["N_single"]
# model = sm.OLS(y, X).fit()
# print(model.summary())

# spearman_r, spearman_p = spearmanr(daily_avg, dst_index_27)
# pearson_r, pearson_p = pearsonr(daily_avg, dst_index_27)
# print("all data")
# print(f"spearman corr: {spearman_r}, spearman p: {spearman_p}")
# print(f"pearson corr: {pearson_r}, pearson p: {pearson_p}")
# spearman_r1, spearman_p1 = spearmanr(num_e, solar_e)
# pearson_r1, pearson_p1 = pearsonr(num_e, solar_e)
# print(f"spearman corr: {spearman_r1}, spearman p: {spearman_p1}")
# print(f"pearson corr: {pearson_r1}, pearson p: {pearson_p1}")
# plot_heatmap()


# correlation: hourly number of events vs Dst
# hour_avg, the_year, the_month, doy, the_hour = process_avg()
# dst = []
# i, j = 0, 0
# while i < len(hour_avg):
#     if the_year[i] == data_dst[j][0] and doy[i] == data_dst[j][4] and the_hour[i] == data_dst[j][3]:
#         dst.append(data_dst[j][-1])
#         i += 1
#     else:
#         j += 1
# corr, p_value = spearmanr(hour_avg, dst)
# print(f"corr_hourly_avg_vs_Dst: {corr}")


# sns.histplot(lt_winter, stat='probability', fill=False, bins=120, element='step')
# sns.histplot(lt_summer, stat='probability', fill=False, bins=120, element='step')
# sns.histplot(lt_e, stat='probability', fill=False, bins=120, element='step', color='black')
# plt.legend(['Southern', 'Northern', 'Equatorial'], frameon=False, fontsize=12)
# plt.legend(['Dec.Solstice', 'Jun.Solstice', 'Equinoxes'], frameon=False, fontsize=12)
# plt.xlim(-180, 180)
# plt.xlabel('Longitude(°)', fontsize=14)
# plt.ylabel('Probability Density', fontsize=14)
# plt.title('Southern Single Crest')
# plt.title('Southern Single Crest', fontsize=14)
# print(np.mean(lt_winter), np.mean(lt_e), np.mean(lt_summer))
# print(len([x for x in lt_winter if x >= 0]) / len(lt_winter))
# print(len([x for x in lt_summer if x >= 0]) / len(lt_summer))
# print(len([x for x in lt_e if x >= 0]) / len(lt_e))
# plt.xlim(1, 12)
# plt.xlabel('Month')

# top 10 the longest single crest
# combined = list(zip(lasting_time, lasting_time_year, lasting_time_doy))
# sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
# top_10 = sorted_combined[:30]
# for record in top_10:
#     last_time, year, doy = record
#     print(f"lasting time: {last_time} min| start: {year}")


# the longest single crest TOP10
# with open("D:/pythonProject/GIMs/longest_single_top10.txt") as f:  # parse by blank line
# current_group, groups = [], []
# with open("D:/pythonProject/GIMs/long_lasting_track.txt") as f:
#     for line in f:
#         stripped_line = line.strip()
#         if stripped_line:
#             current_group.append(stripped_line)
#         else:
#             if current_group:
#                 groups.append(current_group)
#                 current_group = []
#
#     if current_group:
#         groups.append(current_group)
# assert len(groups) == 10
# for i, group in enumerate(groups, 1):
#     times, longs, mag_lats, lts, label = parse_group_data(group)  # 世界时小时记录，经度，地磁纬度，TEC峰值地方时
#     plt.plot(longs, mag_lats, marker='o', markersize=4, label=f'Case{i}|{(times[-1]-times[0]).total_seconds() / 3600}h|Δmlat={(mag_lats[-1] - mag_lats[0]):.2f}°', zorder=2)
#     plt.plot(longs[0], mag_lats[0], marker='+', color='k', markersize=6)
#     lon_drift_rate, lat_change_rate, stats, t_hours = analyze_trend(times, longs, mag_lats)
#     stats.update({
#         'lon_drift_rate': lon_drift_rate,
#         'lat_change_rate': lat_change_rate
#     })
#
#     print(f"\nGroup {i} Analysis:")
#     print(
#         f"Longitude Drift: {lon_drift_rate:.2f} deg/h (Range: {stats['lon_range'][0]:.1f}-{stats['lon_range'][1]:.1f} deg)")
#     print(
#         f"Latitude Change: {lat_change_rate:.2f} deg/h (Range: {stats['lat_range'][0]:.1f}-{stats['lat_range'][1]:.1f} deg)")
#     print(f"Spatial Correlation: {stats['corr_coef']:.2f}")
#
    # plot_spatial_trend(i, t_hours, longs, mag_lats, lts, stats, times)
# long-lasting single long_vs_mlat track
# plt.xlim(-180, 180)
# plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
# plt.legend(loc='best', frameon=False, fontsize=10)
# plt.xlabel('Longitude(°)', fontsize=12)
# plt.ylabel('Magnetic Latitude(°)', fontsize=12)


# records = []
# parse single crest
# with open("D:/pythonProject/GIMs/trans_longitude_stationary.txt") as f:
# with open("D:/pythonProject/GIMs/trans_15deg_per_hour.txt") as f:
# with open("D:/pythonProject/GIMs/evolution.txt") as f:
#     for line in f:
#         stripped_line = line.strip()
#         if not stripped_line:
#             continue
#         parts = line.strip().split()
#         year, mon, day = map(int, parts[:3])
#         dt = datetime.datetime(year, mon, day)
#         records.append((dt, line.strip()))
# groups = []
# current_group = []
# start_time = None
# for dt, line in records:
#     if not current_group:
#         current_group.append(line)
#         start_time = dt
#     else:
#         time_diff = dt - start_time
#         if time_diff <= datetime.timedelta(days=2):
#             current_group.append(line)
#         else:
#             groups.append(current_group)
#             current_group = [line]
#             start_time = dt
# if current_group:
#     groups.append(current_group)
# assert len(groups) == 17


# parse double
# with open("D:/pythonProject/GIMs/evolution.txt") as f:
#     for line in f:
#         stripped_line = line.strip()
#         if stripped_line:
#             current_group.append(stripped_line)
#         else:
#             if current_group:
#                 groups.append(current_group)
#                 current_group = []
#
#     if current_group:
#         groups.append(current_group)


# plot evolution case study
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
# cmap = matplotlib.colormaps.get_cmap('tab20')
# group_colors = [cmap(i / len(groups)) for i in range(len(groups))]
# global_start_time = None
# group_data = []
# for i, group in enumerate(groups, 0):
#     times, longs, mag_lats, lts, date = parse_group_data(group)
#     if i == 0:
#         global_start_time = times[0]
#     t_hours = [(t - global_start_time).total_seconds() / 900 for t in times]
#     group_data.append({'t_hours': t_hours, 'mag_lats': mag_lats})
#     ax1.plot(t_hours, longs, marker='o', markersize=4, zorder=2)
#     ax2.plot(t_hours, mag_lats, marker='o', markersize=4, zorder=2)
#     ax3.plot(t_hours, lts, marker='o', markersize=4, zorder=2)

# t1, mag1 = group_data[0]['t_hours'], group_data[0]['mag_lats']
# t2, mag2 = group_data[1]['t_hours'], group_data[1]['mag_lats']
# t, diff = [], []
# i = j = 0
# while i < len(t1) and j < len(t2):
#     if t1[i] == t2[j]:
#         t.append(t1[i])
#         diff.append(mag1[j] - mag2[i])
#         i += 1
#         j += 1
#     elif t1[i] < t2[j]:
#         i += 1
#     else:
#         j += 1
# ax4.plot(t, diff, marker='o', markersize=4, color='k', zorder=2)
# print(max(diff), min(diff))
# print(max(diff) / min(diff))
# print(max(diff) / diff[0])
# plt.suptitle("Case1 DATE: 2009-11-02", fontsize=14)
# ax4.set_xlabel("Time Sequence (15min)", fontsize=14)
# ax1.set_ylabel("Longitude (°)", fontsize=12)
# ax2.set_ylabel("Magnetic Latitude (°)", fontsize=12)
# ax3.set_ylabel("Local Time", fontsize=12)
# ax4.set_ylabel("MLAT Difference (°)", fontsize=12)
# ax1.grid(True, linestyle='--', alpha=0.5)
# ax2.grid(True, linestyle='--', alpha=0.5)
# ax3.grid(True, linestyle='--', alpha=0.5)
# ax4.grid(True, linestyle='--', alpha=0.5)
# ax1.set_ylim(50, 180)
# ax3.set_ylim(10, 18)
# ax1.set_xlim(0, 50)

# plot trans single
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# cmap = matplotlib.colormaps.get_cmap('tab20')
# group_colors = [cmap(i / len(groups)) for i in range(len(groups))]
# for i, group in enumerate(groups, 0):
#     times, longs, mag_lats, lts, date = parse_group_data(group)
#     if mag_lats[0] > mag_lats[-1]:
#         ax1.plot(longs, mag_lats, color=group_colors[i], marker='o', markersize=4, zorder=2)
#         ax1.scatter(longs[0], mag_lats[0], color='k', s=40, zorder=3, marker='+')
#         t_hours = [(t - times[0]).total_seconds() / 900 for t in times]
#         ax2.plot(t_hours, lts, color=group_colors[i], marker='o', markersize=4, label=f'{date.strftime("%Y-%m-%d")}')
# templates = {
#     'a': "Fixed Lon | MLAT↑",
#     'b': "Fixed Lon | MLAT↓",
#     'c': "West Drift≈Spin | MLAT↑",
#     'd': "West Drift≈Spin | MLAT↓"
# }
# plt.suptitle("(b) Fixed Lon | MLAT↓", fontsize=14)
# ax1.set_xlabel('Longitude(°)', fontsize=12)
# ax1.set_ylabel('Magnetic Latitude(°)', fontsize=12)
# ax2.set_xlabel("Time Sequence (15min)", fontsize=12)
# ax2.set_ylabel("LT", fontsize=12)
# ax1.set_ylim(-15, 15)
# ax1.set_xlim(-40, 120)
# ax2.set_ylim(10, 18)
# ax2.set_xlim(0, 25)
# ax1.grid(True, linestyle='--', alpha=0.5)
# ax2.grid(True, linestyle='--', alpha=0.5)
# ax2.legend(loc='upper right', bbox_to_anchor=(0.97, 1), frameon=False, fontsize=10)




# Dst corr
# i, j = 0, 0
# single_coherent_dst = []
# while i < len(single_crest_simple) and j < len(data_dst):
#     if 1 <= single_crest_simple[i][1] <= 2 or 11 <= single_crest_simple[i][1] <= 12:
#         if np.array_equal(single_crest_simple[i][:4], data_dst[j][:4]):
#             single_coherent_dst.append(data_dst[j][-1])
#             i += 1
#         else:
#             j += 1
#     else:
#         i += 1
# sns.histplot(single_coherent_dst, element='step')
# print(len([x for x in single_coherent_dst if x >= 0]), len([x for x in single_coherent_dst if x >= 0]) / len(single_coherent_dst))
# print(len([x for x in single_coherent_dst if -25 <= x < 0]), len([x for x in single_coherent_dst if -25 <= x < 0]) / len(single_coherent_dst))
# print(len([x for x in single_coherent_dst if -50 <= x < -25]), len([x for x in single_coherent_dst if -50 <= x < -25]) / len(single_coherent_dst))
# print(len([x for x in single_coherent_dst if -100 <= x < -50]), len([x for x in single_coherent_dst if -100 <= x < -50]) / len(single_coherent_dst))
# print(len([x for x in single_coherent_dst if x < -100]), len([x for x in single_coherent_dst if x < -100]) / len(single_coherent_dst))
plt.tight_layout()
plt.show(block=True)

