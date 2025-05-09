#!/usr/bin/env python3
'''
PURPOSE:
    Core script of the feature extraction program:
    Identify TEC intensification regions from given global TEC maps.

    Output TEC intensification region characteristics to an ASCII file. The
    characteristics for each TEC intensification region include: the maximum
    TEC, the geographic latitude and longitude of the maximum TEC, the median
    TEC, the minimum TEC, Regional Electron Content (REC), relative size of 
    the intensification region, and optionally the number of GNSS ground 
    stations within/nearby the intensification region.

    Optionally make two types of plots: 1. maps showing the image processing 
    steps 2. global TEC map with marked intensification regions and locations 
    of their regional TEC maxima. 

AUTHOR:
    Xing Meng, Natalie Maus, Olga Verkhoglyadova

LICENSE:
    Copyright 2023 California Institute of Technology                                             
    Licensed under the Apache License, Version 2.0 (the "License");                               
    you may not use this code except in compliance with the License.                              
    You may obtain a copy of the License at                                                       
    http://www.apache.org/licenses/LICENSE-2.0
                                             
'''

import glob, argparse, copy
from os import path, makedirs
import re, subprocess, sys
import numpy as np
import cv2
from datetime import datetime, timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap


# --------------------------------------------------------------------------------
# Defined functions
# --------------------------------------------------------------------------------

def read_netcdf_map(mapfile):
    '''
    Read one JPL GIM NetCDF file (jpld or jpli) and write data into a 
    dictionary. A typical jpld/jpli file contains one day of data, including 
    variables time, lon, lat, tecmaps, etc, at 15-minute resolution, which 
    means there are 96 timestamps in the file. 

    Input: JPL GIM NetCDF file name, path included
    Output: arrays of times, lats, lons, tecmaps
 
    '''
    
    print('reading '+ mapfile + '...')

    mapdata = Dataset(mapfile,'r')   
    time_in_seconds = np.asarray(mapdata.variables['time'][:])
    lats = np.asarray(mapdata.variables['lat'][:])
    lons = np.asarray(mapdata.variables['lon'][:])
    tecmaps = np.asarray(mapdata.variables['tecmap'])  # TEC值三维数组 时间，经度，纬度
    mapdata.close()

    # convert seconds to datetime for times list
    # note that jpli and jpld files have "varepoch" variable but they are
    # formatted differently, unlike the same-formatted "time" variable.
    times = []
    for it, t in enumerate(time_in_seconds):
        times.append(datetime(2000,1,1,12,0,0) + timedelta(seconds=t))

    assert(len(times) == np.shape(tecmaps)[0])

    return times, lats, lons, tecmaps

# --------------------------------------------------------------------------------

def read_ionex_map(mapfile):
    '''
    Open one IONEX TEC map file and read the lat, lon, epoch, tec into arrays

    '''

    print('reading '+ mapfile + '...')

    # Some regexps that will be used to grep the data from the IONEX file. 
    # note for 'epoch': excludes begining and ending characters, only the
    # time remains. re.compile(r" +START OF TEC MAP.*? +[0-9]+ 
    # +EPOCH OF CURRENT MAP", re.DOTALL|re.MULTILINE) will match all as
    # 'lon', 'lat', etc.
    INX_RE = {
        'lon' : re.compile(r" +[0-9\.\-]+? *?[0-9\.\-]+? *?[0-9\.\-]+? +"+
                           "LON1 / LON2 / DLON"),
        'lat' : re.compile(r" +[0-9\.\-]+? *?[0-9\.\-]+? *?[0-9\.\-]+? +"+
                           "LAT1 / LAT2 / DLAT"),
        'exponent' : re.compile(r" +[0-9\-]+? +?EXPONENT"),
        'epoch' : re.compile(r"START OF TEC MAP.*?\n(.*?)EPOCH OF CURRENT MAP",
                             re.DOTALL|re.MULTILINE),
        'tec' : re.compile(r" +[0-9]+ +START OF TEC MAP.*? +[0-9]+ +END OF TEC MAP",
                           re.DOTALL|re.MULTILINE),
    }

    # View content of *Z file using zcat
    mapdata = subprocess.check_output(['zcat', mapfile], \
                                      universal_newlines=True)
    ## for JPLG*.??I files (unzipped)
    #ionex = subprocess.check_output(['cat', ionex_file],
    #                                universal_newlines=True)

    # Read lon, lat, tec, and time info as strings
    lon_str = INX_RE['lon'].findall(mapdata)
    lat_str = INX_RE['lat'].findall(mapdata)
    exponent_str = INX_RE['exponent'].findall(mapdata)
    time_str = INX_RE['epoch'].findall(mapdata)
    tec_str = INX_RE['tec'].findall(mapdata)

    # construct lon, lat, time arrays
    lonstart,lonend,dlon = list(map(float,lon_str[0].split()[0:3]))
    lons = np.arange(lonstart,lonend+dlon,dlon)
    nlons = len(lons)
    latstart,latend,dlat = list(map(float,lat_str[0].split()[0:3]))
    lats = np.arange(latstart,latend+dlat,dlat)
    nlats = len(lats)
    print('lons=',lons,'lats=',lats)

    exponent = int(exponent_str[0].split()[0])

    # times is a list of datetime objects
    ntimes = len(time_str)
    times = []
    for itime, t_str in enumerate(time_str):
        times.append(datetime.strptime(t_str.strip(), \
                                       '%Y    %m    %d    %H    %M    %S'))
    assert(len(times) == ntimes)

    # read tec values into tecmaps, tecmaps array is initially filled with -1.0
    tecmaps = np.full((ntimes,nlats,nlons),-1.0)
    ilat = -1
    data = [] # contains tec for all longitudes at a given lat and at a given time
    for itime, tec in enumerate(tec_str):
        lines = tec.split('\n')
        for iline, line in enumerate(lines):
            if 'MAP' in line:
                continue
            elif 'LAT/' in line:
                if len(data) != 0:
                    # if data is not empty, len(data) == nlons is false after
                    # the last iteration (except itime=0)
                    sys.exit('ERROR: nlons does not match, data list not empty', data)
                # check if LON1/LON2/DLON match with the lat and lon info from header
                assert(list(map(float, [line[8:14], line[14:20], line[20:26]])) \
                       == [lonstart, lonend, dlon])
                # get the lat index
                ilat = list(lats).index(float(line[2:8]))
                continue
            else:
                # for lines with numbers only, these are tec values
                # convert tec unit to TECU
                def convert(x):
                    if x.strip() == '*****':
                        return -999
                    else:
                        return float(x) * 10**exponent
                data.extend(list(map(convert,line.split())))
                if len(data) == nlons:
                    assert(ilat >= 0)
                    tecmaps[itime,ilat,:] = data
                    # empty data list to be filled with the next lat
                    data = []
    if -1.0 in tecmaps:
        print('WARNING: tecmap is not fully filled')

    return times, lats, lons, tecmaps

# --------------------------------------------------------------------------------

def read_stationloc(stnlocfile, lats, lons):
    '''
    Read one GNSS station location file and return lists of station
    latitude and longitude, map onto the tecmap grid

    Input: GNSS station location file name, path included; lats, lons.
           The station location file should have station lat and lon
           as the 2nd and 3rd column.
    Output: station latitudes, station longitudes, the lat lon indices with
            the lat and lon values closest to station lat and lon.

    Note: the method of mapping stations onto grid may not work well with
          a coarse mesh.
 
    '''
    
    print('reading '+ stnlocfile + '...')

    stnlats = []
    stnlons = []
    stnlatindices = []
    stnlonindices = []
    f = open(stnlocfile, 'r')
    for line in f:
        record = line.split()
        stnlat = float(record[1])        
        stnlon = float(record[2])
        if stnlon > 180.0:
            stnlon = stnlon - 360.0  # convert to -180 --> 180 range
        # append to stn coordinates lists
        stnlats.append(stnlat)
        stnlons.append(stnlon)
            
        # map stations onto the tecmap grid, record the grid indices        
        stnlatindices.append((np.fabs(lats-stnlat)).argmin())
        stnlonindices.append((np.fabs(lons-stnlon)).argmin())

    f.close()
    assert(len(stnlats) == len(stnlons))
    assert(len(stnlatindices) == len(stnlats))
    assert(len(stnlonindices) == len(stnlons))
    
    return stnlats, stnlons, stnlatindices, stnlonindices

# --------------------------------------------------------------------------------

def find_intensifications(tecmap_original, lats, lons, stnlatindices, stnlonindices, \
                          threshold_tecpercentile, threshold_laplacian, \
                          threshold_size, boundary_width):
    '''
    find TEC intensification regions from one TEC map using OpenCV image processing 
    methods
    
    Input: gridded TEC map, latitude and longitude grids; 
           gnss station locations mapped to the map grid;
           threshold_tecpercentile, threshold_laplacian, 
           threshold size of TEC intensification; 
           boundary_width in degrees to pad to each intensification region for
           counting stations
          
    Output:  number of intensification regions, characteristics of each 
             intensification (maximum TEC, latitude and longitude of maximum TEC, 
             median TEC, minimum TEC, regional electron content/GEC, 
             relative size of the region);
             intermediate TEC map processing result: laplacian, 
             tecmap after dilatation and erosion, labels 
  
    '''

    nlats = len(lats)
    nlons = len(lons)
    # convert lat and lon grid spacing in deg to radian, for calculating rec
    dlat = np.radians(abs(lats[1]-lats[0]))
    dlon = np.radians(abs(lons[1]-lons[0]))
    
    # make a copy of tecmap, later on tecmap will be modified
    tecmap = copy.deepcopy(tecmap_original)

    # use laplacian filter to get gradient of each point in image
    # cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])
    # default ksize=1, scale=1, delta=0
    laplacian = cv2.Laplacian(tecmap, -1)  

    # set all data points less than the tec threshold to zero, and set all
    # data points with laplacian greater than threshold_laplacian to zero
    threshold_tec = np.percentile(tecmap_original,threshold_tecpercentile)
    for i in range(nlats):
        for j in range(nlons):
            if tecmap_original[i,j] <= threshold_tec:
                tecmap[i,j] = 0 
            if laplacian[i,j] > threshold_laplacian:
                tecmap[i,j] = 0
                    
    # the new tecmap after the Laplacian step            
    # values beyond (0,255) will be reset to value between (0,255)
    tecmap_laplacian = np.uint8(tecmap)

    # apply dilation and erosion
    # kernel of size 3 seems work best with threslap=0
    kernel = np.ones((3,3), np.uint8)
    tecmap_dilate = cv2.dilate(tecmap_laplacian, kernel, iterations=1)
    tecmap_erode = cv2.erode(tecmap_dilate, kernel, iterations=1)

    # clean all noise after dilatation and erosion
    tecmap_final = cv2.medianBlur(tecmap_erode, 7)

    # computes the connected components labeled image of boolean image and also
    # produces a statistics output for each label. 
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(tecmap_final)

    # set number of intensification region for now
    num_intensifications = num_components  

    # examine each connected component (TEC intensification)
    # mark components to remove/merge
    loc_leftside = []
    loc_rightside = []
    components_leftside = []
    components_rightside = []
    components_small = []
    components_intersectleft = []
    components_intersectright = []
    for icomponent in range(num_components):
	# retrieving the width of the bounding box of the component, in pixels
        width_ = stats[icomponent, cv2.CC_STAT_WIDTH]
	# retrieving the height of the bounding box of the component, in pixels
        height = stats[icomponent, cv2.CC_STAT_HEIGHT]
	# retrieving the leftmost coordinate of the bounding box of the component
        x = stats[icomponent, cv2.CC_STAT_LEFT]
	# retrieving the topmost coordinate of the bounding box of the component
        y = stats[icomponent, cv2.CC_STAT_TOP]
        # size of intensification region in number of grid cells (pixels)
        area = stats[icomponent, cv2.CC_STAT_AREA]

        if area > threshold_size  and width_ < nlons: 
            # store information about leftside and rightside components to check
            # wraps later
            if x == 0: 
                loc_leftside.append([y, y+height])
                components_leftside.append(icomponent)
            if (x + width_) == nlons: 
                loc_rightside.append([y, y+height])
                components_rightside.append(icomponent)
        elif width_ == nlons:
            #print("...... background ", icomponent)
            component_background = icomponent
            num_intensifications = num_intensifications - 1
            pass
        else: 
            ### XM: a small intensification can intersect with a large one on the
            ### other side, should take this case into account?
            #print("...... intensification region too small, remove ", icomponent)
            components_small.append(icomponent)
            num_intensifications = num_intensifications - 1 

    # flag to mark a matched rightside intensification
    rightside_matched = [0]*len(loc_rightside)  
    # check intensification region wraps around from left side of the
    # image to right, should thus be counted as one region rather than two
    if len(loc_leftside) != 0 and len(loc_rightside) != 0 :
        print("...... possilbe wrap")
        for ileft, loc_left in enumerate(loc_leftside):
            if len(loc_rightside) != 0:  
                for iright, loc_right in enumerate(loc_rightside):
                    if rightside_matched[iright]:
                        # skip intensifications already matched up
                        continue                        
                    loc1 = max(loc_left[0], loc_right[0])
                    loc2 = min(loc_left[1], loc_right[1])
                    if (loc1 > loc2):
                        print("......... no intersection") 
                    else:
                        interstn = loc2 - loc1
                        if interstn > 2:
                            #print("......... intersection!", interstn)
                            components_intersectleft.append(components_leftside[ileft])
                            components_intersectright.append(components_rightside[iright])
                            rightside_matched[iright] = 1
                            num_intensifications = num_intensifications - 1 
                            break  # goes to the next loc_left

    # we have gathered info about intensifications to remove and merge
    print('...... small components to remove: ', components_small)
    print('...... intersect components to merge: ', components_intersectleft, \
          components_intersectright)
    # an intersect component at the left edge of the map should always have
    # a corresponding component at the right edge of the map
    assert(len(components_intersectleft) == len(components_intersectright)) 
    # make sure the background one is found
    assert(component_background >= 0) 

    # find out the characteristics for each final intensification region
    # special treatment for intersect components
    tecmaxima = []
    lats_tecmaxima = []
    lons_tecmaxima = []
    tecmedian = []
    tecminima = []
    rec = []
    areas = []
    num_stations = []
    for icomponent in range(num_components):

        if icomponent == component_background or \
           icomponent in components_small or \
           icomponent in components_intersectright:
            continue

        elif icomponent in components_intersectleft:
            # deal with intersect components in components_intersectleft
            # and components_intersectright
            index = components_intersectleft.index(icomponent)
            # the corresponding component 
            icomponent_tomerge = components_intersectright[index]
            mask_for_tecmap = (labels != icomponent) & (labels != icomponent_tomerge)
            # append relative size: this merged intensification area/global area
            areas = areas + [(stats[icomponent, cv2.CC_STAT_AREA] + \
                              stats[icomponent_tomerge, cv2.CC_STAT_AREA])/(nlats*nlons)]

        else:
            mask_for_tecmap = (labels != icomponent)
            # append relative size: this intensification area/global area
            areas = areas + [stats[icomponent, cv2.CC_STAT_AREA]/(nlats*nlons)]

        # obtain masked map and characteristics for this intensification
        tecmap_masked = np.ma.masked_where(mask_for_tecmap, tecmap_original)
        tecmaxima = tecmaxima + [np.ma.amax(tecmap_masked)]
        argmax = np.unravel_index(tecmap_masked.argmax(), tecmap_masked.shape)

        # append lat and lon in degrees
        lats_tecmaxima = lats_tecmaxima + [lats[argmax[0]]]
        lons_tecmaxima = lons_tecmaxima + [lons[argmax[1]]]

        # append tec median and minimum
        tecmedian = tecmedian + [np.ma.median(tecmap_masked)]
        tecminima = tecminima + [np.ma.amin(tecmap_masked)]

        # append regional electron content (rec)
        rec_present = 0
        for ilat, lat in enumerate(lats):
            for ilon, lon in enumerate(lons):
                if tecmap_masked.mask[ilat,ilon]:
                    continue
                rec_present = rec_present + tecmap_masked[ilat,ilon]*dlon* \
                    np.abs(np.sin(np.radians(lat)) - \
                           np.sin(np.radians(lat)+dlat))
        rec = rec + [rec_present*6.371e6**2/1.e16]        

        # count the number of gnss stations within/nearby this intensification
        # if len(stnlatindices) != 0:
        #     coordindices = []
        #     for i in range(nlats):
        #         for j in range(nlons):
        #             if not mask_for_tecmap[i,j]:
        #                 # grid [i,j] falls into the intensification region
        #                 # extend with a boundary layer of given width
        #                 for di in range(-boundary_width,boundary_width+1):
        #                     for dj in range(-boundary_width,boundary_width+1):
        #                         coordpad = [i+di, j+dj]
        #                         if coordpad not in coordindices:
        #                             coordindices.append(coordpad)
        #     nstn = 0
        #     for istn, stnlatindex in enumerate(stnlatindices):
        #         if [stnlatindex, stnlonindices[istn]] in coordindices:
        #             nstn = nstn + 1
        #     num_stations.append(nstn)
            
    # check list lengths    
    assert(len(tecmaxima) == len(areas))
    assert(len(rec) == num_intensifications)
    # if len(num_stations) != 0:  # when num_stations is filled with values
        # assert(len(num_stations) == num_intensifications)
        
    return num_intensifications, tecmaxima, lats_tecmaxima, lons_tecmaxima, \
        tecmedian, tecminima, rec, areas, num_stations, laplacian, tecmap_final, \
        labels, tecmap_laplacian, tecmap_dilate, tecmap_erode

       

# --------------------------------------------------------------------------------

def plot_intensifications(plotdir, t, tecmap, lats, lons, stnlats, stnlons, \
                          num_intensifications, lats_tecmaxima, lons_tecmaxima, \
                          labels):
    '''
    Plot one TEC map marked with identified intensification regions and maximum 
    TEC locations

    Input: plot directory, time, original gridded TEC data, TEC data grid, 
           gnss station locations, gnss station locations, 
           labels(intensification region coverage), location of maximum TEC 
           for each intensification region

    '''

    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7

    plt.figure()
    map = Basemap(projection='gall',llcrnrlat=-90,urcrnrlat=90, \
                  llcrnrlon=-180,urcrnrlon=180, resolution='c')
    map.drawcoastlines(linewidth=0.25, color='darkgray')
    map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1])
    map.drawparallels(np.arange(-90,91,30),labels=[1,0,0,0])
    x, y = map(*np.meshgrid(lons,lats))
    map.contourf(x, y, tecmap, levels=np.linspace(0,120,20), cmap='rainbow')
    cb = plt.colorbar(shrink=0.55)
    cb.ax.set_ylabel('TEC [TECU]')
    map.contour(x,y, labels, colors='gray')
    for iregion in range(num_intensifications):                                                    
        x, y = map(lons_tecmaxima[iregion], lats_tecmaxima[iregion])
        map.plot(x, y, 'kX', ms=4)
    # if len(stnlons) != 0:
    #     for istn, stnlon in enumerate(stnlons):
    #         x,y = map(stnlon,stnlats[istn])
    #         map.plot(x,y,'wo',markersize=1)
    plt.xlabel('Longitude', labelpad=15)
    plt.ylabel('Latitude', labelpad=25)
    plt.title(datetime.strftime(t,'%Y-%m-%d %H:%M') + 'UT  ' + \
              ' TEC Map     ' +'{:2d}'.format(num_intensifications) + \
              ' Intensifications')
    plt.savefig(plotdir + '/tecmap_' + datetime.strftime(t,'%Y%m%d_%H%M') + '.pdf')
    plt.close()
    

# --------------------------------------------------------------------------------

def plot_imgprocessing(plotdir, t, tecmap, laplacian, tecmap_final, labels, \
                       tecmap_laplacian, tecmap_dilate, tecmap_erode):

    '''
    Plot intermediate image processing result for one TEC map:
    a four-panel figure containing the original TEC map,  Laplacian values, 
    result after cv2.dilate, cv2.erode, and cv2.medianBlur, as well as 
    identified objects (intensifications)

    Input: plot directory, time, tecmap, the intermediate image processing data, 
           labels (an integer 2D array of identified component indices)

    '''

    mpl.rcParams['font.size'] = 7
    mpl.rcParams['axes.titlesize'] = 7
    mpl.rcParams['xtick.labelsize'] = 7
    mpl.rcParams['ytick.labelsize'] = 7

    plt.figure()
    plt.subplot(221)
    plt.title('Original TEC Map [TECU]')
    plt.imshow(tecmap)
    plt.colorbar(shrink=0.5)

    plt.subplot(222)
    plt.title('Laplacian')
    plt.imshow(laplacian,cmap='RdBu_r',vmin=-3, vmax=3)
    plt.colorbar(shrink=0.5)

    plt.subplot(223)
    plt.title('TEC Map Dilated and Eroded')
    plt.imshow(tecmap_erode)

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0
    plt.subplot(224)
    plt.title('Component Labels')
    plt.imshow(labeled_img)
    
    plt.subplots_adjust(top=0.8, hspace=0.3, wspace=0.3)
    plt.figtext(0.5, 0.9, 'TEC map processing for ' + \
                datetime.strftime(t,'%Y-%m-%d %H:%M') + 'UT', \
                horizontalalignment='center')
    plt.savefig(plotdir + '/imgprocessing_' + \
                datetime.strftime(t,'%Y%m%d_%H%M') + '.pdf')
    plt.close()


# --------------------------------------------------------------------------------
# End of defined functions
# --------------------------------------------------------------------------------


if __name__ == '__main__':
# python D:\pythonProject\GIMs\find_intensification_.py --mapfiledir D:\pythonProject\GIMs\jpld_case_study --mapfiletype netcdf  --label case_study --plotdir D:\pythonProject\GIMs\plot
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapfiledir', help='directory of map files', required=True)
    parser.add_argument('--mapfiletype', help='type of map files, can be netcdf ' + \
                        'or ionex', required=True)
    parser.add_argument('--label', help='label to append to the output file name ' + \
                        'and plot directory name', required=True)
    parser.add_argument('--plotdir', help='directory to save plots, default is ' + \
                        './plots_[LABEL]', \
                        nargs='?', const='plots', default='plots')
    parser.add_argument('--plotdt', help='save plots every [plotdt] maps, ' + \
                        'set to 0 to not save plots, default is 12', \
                        nargs='?', type=int, const=1, default=1)
    parser.add_argument('--threstec', help='TEC threshold in percentile, between ' + \
                        '0 and 100, default 97', \
                        nargs='?', type=float, const=97, default=97)
    parser.add_argument('--threslap', help='threshold for laplacian, typically ' + \
                        'between -1 and 0, default 0', \
                        nargs='?', type=float, const=0, default=0)
    parser.add_argument('--thressiz', help='threshold for intensification minimum ' + \
                        'area in number of grid cells. Default is 65, which is ' + \
                        'good for capturing large-scale intensifications from ' + \
                        '1degX1deg maps. The value should be reduced for ' + \
                        'lower-resolution maps.', \
                        nargs='?', type=float, const=65, default=65)
    parser.add_argument('--stationloc', help='path and name of the file ' + \
                        'containing locations of GNSS stations used to ' + \
                        'generate the TEC map. If provided, count the number of ' + \
                        'GNSS stations within/nearby each intensification region ' + \
                        'and write to the output file')
    parser.add_argument('--boundarywidth', help='width in number of grid cells ' + \
                        'to add as the boundary of an intensification region, ' + \
                        'for counting the GNSS stations within the intensification ' + \
                        'including the boundary. Default is 3, which is good ' + \
                        'for 1degX1deg maps. The value should be reduced for ' + \
                        'lower-resolution maps.', \
                        nargs='?', type=int, const=3, default=3)
    args = parser.parse_args()

    # map file name extension
    if args.mapfiletype == 'netcdf':
        mapfileext = '.nc'
    elif args.mapfiletype == 'ionex':
        mapfileext = '.Z'
    else:
        sys.exit('ERROR: mapfiletype not recognized, use "netcdf" or "ionex"')
        
    # create plot directory if desired and not already exist
    if args.plotdt != 0:
        plotdir = args.plotdir + '_' + args.label
        if not path.exists(plotdir):
            makedirs(plotdir)

    # set up variables for counting GNSS stations
    if args.stationloc != None:
        do_count_stations = True
        stnlocfile = args.stationloc
    else:
        do_count_stations = False

    # initialize the output file containing intensifications
    outfile = 'intensifications_' + args.label + '.dat'
    with open(outfile,'w') as intensfile:
        intensfile.write('# Intensification regions on TEC maps\n')
        intensfile.write('# year month day hour minute second index_intensification ' + \
                         'TECmax[TECU] lat@TECmax[degree] lon@TECmax[degree] ' + \
                         'TECmedian[TECU] TECmin[TECU] REC[GECU] relative_size')
        if do_count_stations:
            intensfile.write(' num_of_stations')
        intensfile.write('\n')

    # work with all TEC map files in the directory
    mapfiles = sorted(glob.glob(args.mapfiledir + '/*' + mapfileext))
    for ifile, mapfile in enumerate(mapfiles):

        # read the TEC map file
        if mapfileext == '.nc':
            times, lats, lons, tecmaps = read_netcdf_map(mapfile)
        else:
            times, lats, lons, tecmaps = read_ionex_map(mapfile)
            
        # read station locations if applicable, for the first map file only
        # assuming other map files were produced with the same set of stations.
        # future capability extension: for multi-day runs with a unique station
        # location file for each day, modify stnlocfile to be date-specific.
        if ifile == 0:
            if do_count_stations:
                stnlats, stnlons, stnlatindices, stnlonindices = \
                    read_stationloc(stnlocfile, lats, lons)
                print('num of gnss stations = ', len(stnlats))
            else:
                stnlats = []
                stnlons = []
                stnlatindices = []
                stnlonindices = []


        # the first time stamp
        t0 = times[0]
        for itime, t in enumerate(times):

            # ignore data from the next day: most netcdf and ionex map files
            # contain one day of TEC data only, but some map files include
            # data at 0UT of the next day
            if t.day != t0.day:
                continue

            print('... working on map at ' + datetime.strftime(t,'%H:%M') + 'UT')

            # the TEC map at this time
            tecmap = tecmaps[itime]

            # find the number of intensification regions and their characteristics
            num_intensifications, tecmaxima, lats_tecmaxima, lons_tecmaxima, \
                tecmedian, tecminima, rec, areas, num_stations, \
                laplacian, tecmap_final, labels, tecmap_laplacian, \
                tecmap_dilate, tecmap_erode = \
                    find_intensifications(tecmap, lats, lons, \
                                          stnlatindices, stnlonindices, \
                                          args.threstec, \
                                          args.threslap, \
                                          args.thressiz, \
                                          args.boundarywidth)
            
            # append intensification characteristics for this tecmap into file            
            if do_count_stations:
                with open(outfile,'a') as intensfile:
                    for iregion in range(num_intensifications):
                        intensfile.write(('{:4d}'*7+'{:8.1f}'*5+'{:12.6f}'*2+'{:4d}\n') \
                                         .format(t.year, t.month, t.day, t.hour, \
                                                 t.minute, t.second, iregion+1,
                                                 tecmaxima[iregion], \
                                                 lats_tecmaxima[iregion], \
                                                 lons_tecmaxima[iregion], \
                                                 tecmedian[iregion], \
                                                 tecminima[iregion],
                                                 rec[iregion], \
                                                 areas[iregion], \
                                                 num_stations[iregion]))
            else:
                with open(outfile,'a') as intensfile:
                    for iregion in range(num_intensifications):
                        intensfile.write(('{:4d}'*7+'{:8.1f}'*5+'{:12.6f}'*2+'\n') \
                                         .format(t.year, t.month, t.day, t.hour, \
                                                 t.minute, t.second, iregion+1,
                                                 tecmaxima[iregion], \
                                                 lats_tecmaxima[iregion], \
                                                 lons_tecmaxima[iregion], \
                                                 tecmedian[iregion], \
                                                 tecminima[iregion],
                                                 rec[iregion], \
                                                 areas[iregion]))
                        
            # make and save plots if desired
            if args.plotdt !=0 and itime%args.plotdt == 0:
                plot_intensifications(plotdir, t, tecmap, lats, lons, \
                                      stnlats, stnlons, \
                                      num_intensifications, lats_tecmaxima, \
                                      lons_tecmaxima, labels)        
                # plot_imgprocessing(plotdir, t, tecmap, laplacian, \
                #                    tecmap_final, labels, \
                #                    tecmap_laplacian, tecmap_dilate, tecmap_erode)

	

