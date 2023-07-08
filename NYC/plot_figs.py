"""
This module contains user defined functions to plot figures used in the EDA notebook.
"""


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = [10, 5]
import seaborn as sns

import numpy as np
import pandas as pd
from bokeh.io import output_file, output_notebook, show, curdoc
import bokeh, bokeh.plotting, bokeh.models # check if repeated
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, LogColorMapper, BasicTicker, LogTicker, ColorBar,
    DataRange1d, Range1d, PanTool, WheelZoomTool, BoxSelectTool, ResetTool, SaveTool, CustomJS, Slider,
    Legend, LegendItem
                            )
from bokeh.models.widgets import Button, CheckboxButtonGroup, CheckboxGroup
from bokeh.models.mappers import ColorMapper, LinearColorMapper, CategoricalColorMapper
from bokeh import palettes
from bokeh.layouts import column, row, gridplot
from bokeh.plotting import figure
from bokeh.tile_providers import get_provider, WIKIMEDIA, CARTODBPOSITRON
from bokeh.transform import linear_cmap
import pyproj


def plot_points(df, colA = 'pickup_longitude', colB = 'pickup_latitude', s = 0.5, alpha = 0.5, 
                color_points = 'xkcd:lime', color_background = 'xkcd:black', figsize = (7,7),
                nyc_long_limits = (-74.257159, -73.699215), nyc_lat_limits = (40.471021, 40.987326)):

    """
    PLotting the lat and lon of the pickup locations to get a gist of the data.
    The nyc_long_limits and nyc_lat_limits constraint the plot to the NYC boundaries.
    """

    nyc_long_limits = nyc_long_limits
    nyc_lat_limits = nyc_lat_limits
    # Scaling the figure size so that the figure scales up as per the lat and lon differences
    figsize = (abs(nyc_long_limits[0] - nyc_long_limits[1])*figsize[0], 
                            abs(nyc_lat_limits[0] - nyc_lat_limits[1])*figsize[1])
    
    fig, ax = plt.subplots(1, figsize = figsize)
    _ = ax.scatter(df[colA].values, df[colB].values,
                  color = color_points, s = s, label='train', alpha = alpha)
    
    _ = ax.set_ylabel('latitude')
    _ = ax.set_xlabel('longitude')
    
    _ = plt.ylim(nyc_lat_limits)
    _ = plt.xlim(nyc_long_limits)
    _ = plt.title(colA.replace("_", " ").split(" ")[0] + ' locations')
    
    _ = ax.set_facecolor(color_background)
    _ = ax.grid(False)    #= plt.grid(b=None)

   
def distribution(x_col, data, scale = None, bins = 'auto', stat = 'percent', 
                 hue = None, row = None, col = None, legend_loc = None, figsize = (10, 7), **kwargs):
    """ 
    Function for plotting histogram of any particular x variable of a dataframe
    """
    _ = sns.set(rc={'figure.figsize': figsize})
    g = sns.displot(data = data, x = x_col, bins = bins, kde = False, 
                    stat = stat, hue = hue, row = row, col = col, height = figsize[1], 
                    aspect = figsize[0]/figsize[1], **kwargs)

    #move overall title up, and add a super title
    g.fig.subplots_adjust(top = .95)
    g.fig.suptitle('Distribution of \'{}\''.format(x_col.capitalize().replace('_', ' ')))

    _ = plt.xlabel(x_col.capitalize().replace('_', ' '))
    _ = plt.ylabel(stat)
    if (hue is not None) and (legend_loc is not None): _ = sns.move_legend(obj = g, loc = legend_loc, frameon = False)
    if scale: 
        _ = plt.xscale(scale)


def bokeh_distplot(data, category_col = ['trip_bins_minutes'], value = 'pickup_latitude',
                  plot_width=500, plot_height = 300, legend_loc = 'top_right'):
    """
        Plots the distribution of the _value_ varaible categorized by the _category_col_.
    """
    p = iqplot.histogram(
                data = data,
                cats = category_col,
                q = value, width = plot_width, height = plot_height, 
                title = "Distribution of "+value+" categorized by "+str([category_col])
                                )
    p.legend.location = legend_loc
    return(p)


def zone_plot(nyc_shp, fill_color = 'LocationID'):
    """
    Plots the zone and borough boundaries.
    """
    gjds = bokeh.models.GeoJSONDataSource(geojson = nyc_shp.to_json())
    TOOLS = "pan, wheel_zoom,reset,hover,save"
    
    plot_zone = bokeh.plotting.figure(title = "NYC Taxi Zones", tools = TOOLS,
        x_axis_location = None, y_axis_location = None)#, responsive=True)
    
    color_mapper = bokeh.models.LinearColorMapper(palette = bokeh.palettes.Viridis256)
    
    plot_zone.patches('xs', 'ys', 
              fill_color = {'field': fill_color, 'transform': color_mapper},#borough_num
              fill_alpha = 1., line_color="black", line_width = 0.5,          
              source = gjds)
    
    plot_zone.grid.grid_line_color = None
    
    hover = plot_zone.select_one(bokeh.models.HoverTool)
    hover.point_policy = "follow_mouse"
    
    hover.tooltips = [
                        ("Name", "@zone"),
                        ("Borough", "@borough"),
                        ("Zone ID", "@LocationID"),
                        ("(Lon, Lat)", "($x ˚E, $y ˚N)")
                         ]
    return(plot_zone)


def plot_single_gmaps(data, latitude_column = 'pickup_latitude', longitude_column = 'pickup_longitude', 
                         color_column = 'trip_duration', size_column = 0.5,
                         api_key = None, map_type = 'roadmap', map_zoom = 10):
    
    """
    Plot interactive plot of all data points on a google map.
    Takes in the columns of data including the lat and lon to be plotted on gmaps. 
    Arguments:
        data: dataframe including the lat and lon locations
        color_column: column name according to which the points will be colored and a colorbar will be plotted
        size_column: can be int/float or column name in string. If column name then that column will be used to scale the size of the points.
        api_key: your google maps api key
    """

    data = data.copy()
    if not isinstance(size_column, str):
        s = size_column
        size_column = 'constant'
        data[size_column] = s
        
    map_options = GMapOptions(lat = data[latitude_column].mean(), lng = data[longitude_column].mean(), 
                              map_type = map_type, zoom = map_zoom)
    
    Tools = "box_select, wheel_zoom, pan, reset, help" 
    
    # You can use either a GmapPlot or gmap to create the plot
    plot = GMapPlot(api_key = api_key, map_options = map_options, #x_range = Range1d(), y_range = Range1d(),
                    width = 500, height = 400, )  #google_, gmap, tools = Tools
    
    pan = PanTool()
    wheel_zoom = WheelZoomTool()
    box_select = BoxSelectTool() 
    reset_tool = ResetTool() 
    save_tool = SaveTool()
    
    plot.add_tools(pan, wheel_zoom, box_select, reset_tool, save_tool)
    
    plot.title.text = "NYC taxi {} locations {}".format(latitude_column.split('_')[0].capitalize(), 
                                                        data.pickup_datetime.dt.year.unique().tolist())
    
    #plot.api_key = api_key
    
    source = ColumnDataSource(
        data = dict(
            lat = data[latitude_column].tolist(),
            lon = data[longitude_column].tolist(),
            size = data[size_column].tolist(),
            color = data[color_column].tolist()
        )
    )
    
    #color_mapper = LinearColorMapper(palette="Dark2")

    if not np.issubdtype(data[color_column].dtype, np.number):
        raise TypeError('Only numeric data types can be passed as a color column')
    else:
        color_mapper = LinearColorMapper(palette = "RdYlBu5", low = np.percentile(data[color_column], 1), 
                                                              high = np.percentile(data[color_column], 99)) #bokeh.palettes.Turbo256
        color_bar = ColorBar(color_mapper = color_mapper, ticker = BasicTicker(),
                            label_standoff = 12, border_line_color = None, location = (0,0), title = color_column)
        plot.add_layout(color_bar, 'right')
    circle = Circle(x = "lon", y = "lat", fill_alpha = 0.7, size = "size", 
                    fill_color  ={'field': 'color', 'transform': color_mapper}, line_color = None)
    
    plot.add_glyph(source, circle)
        
    # removed the below because gmap already adds them by default. We can change it by using the tools property
    #plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), ResetTool(), SaveTool())
    
    #output_file("NYC_pickup_plot.html")
    #output_notebook()
    return(plot, output_file("gmap.html"))


def plot_zone_trips_counts(df: pd.DataFrame, nyc_shp: pd.DataFrame, 
                        to_plot: str = 'count', divide_by: float = 1, 
                        col_to_plot: str = "pickup_taxizone_id") -> bokeh.plotting.figure:
    
    """ 
    Plots the total number of rides or the average trip duration within all zones in NYC. 
    Args:
        df: dataframe
        nyc_shp: shape file
        to_plot: 'count' if count of rides is to be plotted or 'column name' of the column to be used as the displayed values
        divide_by: default 60 to plot the 'trip duration' column in minutes. Use 1 for any other column to be used as it is.
        col_to_plot: pickup_taxizone_id or dropoff_taxizone_id
    Returns:
        bokeh.plotting.figure
    """
    
    df = df.copy()

    if to_plot == 'count':
        counts = df.groupby(col_to_plot).size().reset_index(name='N')
        counts['N'] = counts['N']/divide_by
        tag = "Number of trips"
        ticker = LogTicker()
        cbar_title = "Total number of "+col_to_plot.split("_")[0]+"s "
        color_mapper = bokeh.models.LogColorMapper(palette = bokeh.palettes.Turbo256, low = counts.N.min(), high = counts.N.max())

    else:
        counts = df.groupby(col_to_plot)[to_plot].mean().reset_index(name='N')
        counts['N'] = counts['N']/divide_by
        tag = to_plot
        if divide_by == 60:
            tag = to_plot+" mts"

        cbar_title = tag
        ticker = BasicTicker()#LogTicker()
        color_mapper = bokeh.models.LinearColorMapper(palette = bokeh.palettes.Turbo256, 
                                                      low = np.percentile(df[to_plot]/divide_by, 5), 
                                                      high = np.percentile(df[to_plot]/divide_by, 95))
        
    counts2 = nyc_shp.merge(counts, left_on='LocationID', 
                            #right_index=True, 
                            right_on = col_to_plot,
                            how='left')
    
    gjds = bokeh.models.GeoJSONDataSource(geojson = counts2.to_json())
    TOOLS = "pan,wheel_zoom,reset,hover,save"
    title = "NYC Taxi "+col_to_plot.split("_")[0]+"s " + to_plot + " map"
    p = bokeh.plotting.figure(title = title, tools = TOOLS,
                              x_axis_location = None, y_axis_location = None,) 
                              #plot_width = np.int(1.08*500), plot_height = 500)
    
    p.patches('xs', 'ys', 
              fill_color = {'field': 'N', 'transform': color_mapper},
              fill_alpha = 1., line_color = "black", line_width=0.5,          
              source = gjds)
    
    p.grid.grid_line_color = None
    
    hover = p.select_one(bokeh.models.HoverTool)
    hover.point_policy = "follow_mouse"
   
    hover.tooltips = [
                        ("Name", "@zone"),
                        ("Borough", "@borough"),
                        (tag, "@N"),
                        ("Zone ID", "@LocationID")
                     ]


    color_bar = bokeh.models.ColorBar(
                                    color_mapper = color_mapper, orientation='horizontal',
                                    ticker = ticker,
                                    formatter=bokeh.models.PrintfTickFormatter(format = '%d'),
                                    label_standoff = 12, border_line_color = None, 
                                    location = (0,0), title = cbar_title
                                )
    
    p.add_layout(color_bar, 'below')
    
    return p


def plot_gmaps(data: pd.DataFrame, slider: bool = False, 
            latitude_column: list[str] = ['pickup_latitude', 'dropoff_latitude'], 
            longitude_column: list[str] = ['pickup_longitude', 'dropoff_longitude'],
            color_column: str = 'trip_duration', size_column: float|str = 3.0,
            api_key: str = None, 
            map_type: str = 'roadmap', map_zoom: int = 10, 
            width: int = 500, height: int = 400) -> curdoc:
        
    """
    Plots interactive plot of NYC taxi pickup and dropoff locations on the google maps. Pickup and dropoff locations are plotted side
    by side and if you select some location/s on the pickup plot the corresponding dropoff locations will be highlighted in the dropoff

    Arguments:
        data: dataframe with nyc trips info
        slider: if True, will plot a slider for selecting a particular hour in the day. default: False
        latitide_column: pickup and dropoff lat pairs
        longitude_column: pickup and dropoff lon pairs
        color_column: column in the dataframe to be used for color labeling the scatter points
        size_column: column to be used to determine the size of the scatter points. Can be constant scaler or any dataframe column.
        api_key: your Google maps API key
        map_type: roadmap, satellite, etc. 
        map_zoom: zoom level of the map.
        width: width of the plots
        height: height of the plots
    Returns:
        Bokeh callback enabled plot
        """

    #callback function
    data = data.copy()
    if not isinstance(size_column, str):
        s = size_column
        size_column = 'constant'
        data[size_column] = s
        
    def modify_plot(doc):     
        #######################################################################################################################
        # Update the checkbox values
        
        ## Checkbox for hour of the day values
        def update_hour(atrr, old, new): 
            data_to_plot = [int(checkbox_group_hour.labels[i]) for i in checkbox_group_hour.active] 
            new_data = data[data.pickup_hour.isin(data_to_plot)]
            source.data = dict(
                latp = new_data[latitude_column[0]].tolist(),
                lonp = new_data[longitude_column[0]].tolist(),
                latd = new_data[latitude_column[1]].tolist(),
                lond = new_data[longitude_column[1]].tolist(),
                size = new_data[size_column].tolist(),
                color = new_data[color_column].tolist()
            )
            
        hours = data.pickup_hour.unique().tolist()
        hours.sort()
        str1 = [str(i) for i in hours]
        checkbox_group_hour = CheckboxButtonGroup(labels = str1, active=[0, 1])
        
        ## Adding a slider for the hour also as another option
        
        def update_hour_slider(atrr, old, new): 
            hour = slider_hour.value
            new_data = data[data.pickup_hour == hour]
            source.data = dict(
                latp = new_data[latitude_column[0]].tolist(),
                lonp = new_data[longitude_column[0]].tolist(),
                latd = new_data[latitude_column[1]].tolist(),
                lond = new_data[longitude_column[1]].tolist(),
                size = new_data[size_column].tolist(),
                color = new_data[color_column].tolist()
                )

        slider_hour = Slider(title = 'Hour of the day', start = 0, end = 23, step = 1, value = 17)
        
        ## Checkbox for weekday values
        def update_weekday(atrr, old, new): 
            data_to_plot = [checkbox_group_weekday.labels[i] for i in checkbox_group_weekday.active] 
            new_data = data[data.pickup_weekday.isin(data_to_plot)]
            source.data = dict(
                latp = new_data[latitude_column[0]].tolist(),
                lonp = new_data[longitude_column[0]].tolist(),
                latd = new_data[latitude_column[1]].tolist(),
                lond = new_data[longitude_column[1]].tolist(),
                size = new_data[size_column].tolist(),
                color = new_data[color_column].tolist()
            )
            
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        checkbox_group_weekday = CheckboxButtonGroup(labels = weekdays, active=[0, 1])
        
        
        #########################################################################################################
        # CREATE THE PLOT
        # Assigning map options
        map_options = GMapOptions(lat = data[latitude_column[0]].mean(), lng = data[longitude_column[0]].mean(), 
                                      map_type = map_type, zoom = map_zoom)
        
        assert len(latitude_column) == len(longitude_column), "latitude and longitude list should be same"
            
        # Pickup plot
        plotP = GMapPlot(x_range = Range1d(), y_range = Range1d(), map_options = map_options, 
                         width = width, height = height)
        
        # Dropoff plot
        plotD = GMapPlot(x_range = Range1d(), y_range = Range1d(), map_options = map_options, 
                         width = width, height = height)
        
        # ASsigning the Google API keys
        plotP.api_key = api_key
        plotD.api_key = api_key
        
        source = ColumnDataSource(
            data = dict(
                latp = data[latitude_column[0]].tolist(),
                lonp = data[longitude_column[0]].tolist(),
                latd = data[latitude_column[1]].tolist(),
                lond = data[longitude_column[1]].tolist(),
                size = data[size_column].tolist(),
                color = data[color_column].tolist()
            )
        )
        
        plotP.title.text = "NYC taxi pickup locations {}".format(
                                                      data.pickup_datetime.dt.year.unique().tolist())
        
        plotD.title.text = "NYC taxi dropoff locations {}".format(
                                                      data.dropoff_datetime.dt.year.unique().tolist())
        #RdYlGn5, RdBu4
        # Add color mapper for the color bar
        color_mapper = LinearColorMapper(palette = "RdYlGn8", low = np.percentile(data[color_column], 1), 
                                                           high = np.percentile(data[color_column], 99))
        
        # Add the scatter points
        circleP = Circle(x = "lonp", y = "latp", fill_alpha = 0.8, size = "size", 
                        fill_color={'field': 'color', 'transform': color_mapper}, line_color = None)

        circleD = Circle(x = "lond", y = "latd", fill_alpha = 0.8, size = "size", 
                        fill_color={'field': 'color', 'transform': color_mapper}, line_color = None)
        
        # Only the user selected scatter points will be highlighted
        #selected_circle = Circle(fill_alpha = 1) #, fill_color = "firebrick", line_color = None)
        nonselected_circle = Circle(fill_alpha = 0.1, fill_color = "grey", line_color= None)
        ###############
        
        plotP.add_glyph(source, circleP, nonselection_glyph = nonselected_circle)#, selection_glyph = selected_circle, )
        plotD.add_glyph(source, circleD, nonselection_glyph = nonselected_circle)#, selection_glyph = selected_circle, )
        
        
        # Add color bar. ADding to only one of the plots becaue their x and y axis are scaled together
        color_bar = ColorBar(color_mapper = color_mapper, ticker = BasicTicker(),
                             label_standoff = 12, border_line_color = None, location = (0,0), title = color_column)
        
        plotD.add_layout(color_bar, 'right')
        
        # Add the basic interactive tools
        plotP.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), ResetTool(), SaveTool())
        plotD.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), ResetTool(), SaveTool())
        #########################################################################################################
        
        #plots.append(plot)
            
        # Keep the ranges same for both plots so that when one plot is zoomed in the other will too
        # For now switching this off because when zooming in or panning one plot the points in other plot were swaying off
        # their original geo locations
        #plotD.x_range = plotP.x_range
        #plotD.y_range = plotP.y_range
        
        # Create the layout with widgets and actual plots
        if slider:
            rowl = row(plotP, plotD)
            layout = column(checkbox_group_weekday, column(slider_hour), rowl)#)
            slider_hour.on_change('value', update_hour_slider)
        else:
            rowl = row(plotP, plotD)
            layout = column(checkbox_group_hour, checkbox_group_weekday, rowl)#row(plotP, plotD))
            checkbox_group_hour.on_change('active', update_hour)
        
        checkbox_group_weekday.on_change('active', update_weekday)
        
        # add the layout to curdoc
        doc.add_root(layout)
        output_notebook()
    
    return(modify_plot) 


def plot_gmaps_streamlit(data: pd.DataFrame, 
                        latitude_column: list[str] = ['pickup_latitude', 'dropoff_latitude'], 
                        longitude_column: list[str] = ['pickup_longitude', 'dropoff_longitude'],
                        color_column: str = 'trip_duration', 
                        size_column: float|str = 3.0,
                        api_key: str = None, 
                        map_type: str = 'roadmap', map_zoom: int = 10, 
                        width: int = 500, height: int = 400) -> bokeh.layouts:
        
    """
    Plots interactive plot of NYC taxi pickup and dropoff locations on bokeh Google mapss. Pickup and dropoff locations are plotted side
    by side and if you select some location/s on the pickup plot the corresponding dropoff locations will be highlighted in the dropoff.
    
    Arguments:
        data: pandas dataframe with latitude and longitude columns listed below
        latitude_column: a list of latitude columns. The index 0 column serves as the y axis for the left plot, and index 1 column for the right plot. 
        longitude_column: a list of longitide columns. The index 0 column serves as the x axis for the left plot, and index 1 column for the right plot. 
        color_column: column to use for assigning colors to the points
        size_column: column used to determine the size of the scatter points. Can be constant scaler or any dataframe column.
        api_key: your Google maps API key
        map_type: roadmap, satellite, etc. 
        map_zoom: zoom level of the map.
        width: width of the plots
        height: height of the plots
    Returns:
        Bokeh layout plot (bokeh.layouts) -> row(left, right) 
    """

    #callback function
    data = data.copy()

    if not isinstance(size_column, str):
        s = size_column
        size_column = 'constant'
        data[size_column] = s
    
    #########################################################################################################
    # CREATE THE PLOT
    # Assigning map options
    map_options = GMapOptions(lat = data[latitude_column[0]].mean(), lng = data[longitude_column[0]].mean(), 
                            map_type = map_type, zoom = map_zoom)
    
    assert len(latitude_column) == len(longitude_column), "latitude and longitude list should be same"
        
    # Pickup plot
    plotP = GMapPlot(x_range = Range1d(), y_range = Range1d(), map_options = map_options, 
                        width = width, height = height)
    
    # Dropoff plot
    plotD = GMapPlot(x_range = Range1d(), y_range = Range1d(), map_options = map_options, 
                        width = width, height = height)
    
    # ASsigning the Google API keys
    plotP.api_key = api_key
    plotD.api_key = api_key
    
    source = ColumnDataSource(
        data = dict(
            latp = data[latitude_column[0]].tolist(),
            lonp = data[longitude_column[0]].tolist(),
            latd = data[latitude_column[1]].tolist(),
            lond = data[longitude_column[1]].tolist(),
            size = data[size_column].tolist(),
            color = data[color_column].tolist()
        )
    )
    
    plotP.title.text = "NYC taxi pickup locations {}".format(
                                                    data.pickup_datetime.dt.year.unique().tolist())
    
    plotD.title.text = "NYC taxi dropoff locations {}".format(
                                                    data.dropoff_datetime.dt.year.unique().tolist())
    #RdYlGn5, RdBu4
    # Add color mapper for the color bar
    color_mapper = LinearColorMapper(palette = "RdYlGn8", low = np.percentile(data[color_column], 1), 
                                                        high = np.percentile(data[color_column], 99))
    
    # Add the scatter points
    circleP = Circle(x = "lonp", y = "latp", fill_alpha = 0.8, size = "size", 
                    fill_color={'field': 'color', 'transform': color_mapper}, line_color = None)

    circleD = Circle(x = "lond", y = "latd", fill_alpha = 0.8, size = "size", 
                    fill_color={'field': 'color', 'transform': color_mapper}, line_color = None)
    
    # Only the user selected scatter points will be highlighted
    #selected_circle = Circle(fill_alpha = 1) #, fill_color = "firebrick", line_color = None)
    nonselected_circle = Circle(fill_alpha = 0.1, fill_color = "grey", line_color= None)
    
    plotP.add_glyph(source, circleP, nonselection_glyph = nonselected_circle)
    plotD.add_glyph(source, circleD, nonselection_glyph = nonselected_circle)
    
    # Add color bar. ADding to only one of the plots becaue their x and y axis are scaled together
    color_bar = ColorBar(color_mapper = color_mapper, ticker = BasicTicker(),
                        label_standoff = 12, border_line_color = None, location = (0,0), title = color_column)
    
    plotD.add_layout(color_bar, 'right')
    
    # Add the basic interactive tools
    plotP.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), ResetTool(), SaveTool())
    plotD.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), ResetTool(), SaveTool())
    
    # Create the layout with plots
    layout = row(plotP, plotD)

    return(layout) 


def plot_cartmaps_streamlit(data: pd.DataFrame, 
                        latitude_column: list[str] = ['pickup_MercatorY', 'dropoff_MercatorY'], 
                        longitude_column: list[str] = ['pickup_MercatorX', 'dropoff_MercatorX'],
                        map_tile_type: bokeh.tile_providers = CARTODBPOSITRON,
                        nyc_long_limits: tuple = (-74.257159, -73.699215), 
                        nyc_lat_limits: tuple = (40.471021, 40.987326),
                        color_column: str = 'trip_duration', 
                        lower_color_lim: float|None = None, upper_color_lim: float|None = None,
                        size_column: float|str = 5.0,
                        width: int = 800, height: int = 700, fill_alpha: float = 1) -> bokeh.layouts:
        
    """
    Plots interactive plot of NYC taxi pickup and dropoff locations on bokeh tile maps. Pickup and dropoff locations are plotted side
    by side and if you select some location/s on the pickup plot the corresponding dropoff locations will be highlighted in the dropoff.

    Arguments:
        data: pandas dataframe with latitude and longitude columns listed below
        latitude_column: a list of latitude columns. The index 0 column serves as the y axis for the left plot, and index 1 column for the right plot. 
        longitude_column: a list of longitide columns. The index 0 column serves as the x axis for the left plot, and index 1 column for the right plot. 
        map_tile_type: one of the options from bokeh.tile_providers
        nyc_long_limits: x axis limits for both the left and right plots
        nyc_lat_limits: y axis limits for both the left and right plots
        color_column: column to use for assigning colors to the points
        lower_color_lim: lower limit for the colorbar
        upper_color_lim: upper limit for the color bar
        size_column: column used to determine the size of the scatter points. Can be constant scaler or any dataframe column.
        width: width of the plots
        height: height of the plots
        fill_alpha: transparency (1 being complete opaque, 0 being all transparent)
    Returns:
        Bokeh layout plot (bokeh.layouts) -> row(left, right) 
        """

    data = data.copy()

    if not isinstance(size_column, str):
        s = size_column
        size_column = 'constant'
        data[size_column] = s

    source = ColumnDataSource(
        data = dict(
            latp = data[latitude_column[0]].tolist(),
            lonp = data[longitude_column[0]].tolist(),
            latd = data[latitude_column[1]].tolist(),
            lond = data[longitude_column[1]].tolist(),
            zone_name_p = data['pickup_zone_name'].tolist(),
            zone_name_d = data['dropoff_zone_name'].tolist(),
            size = data[size_column].tolist(),
            color = data[color_column].tolist()
        )
    )

    # Add color mapper for the color bar
    lower_color_lim = np.percentile(data[color_column], 1) if lower_color_lim is None else lower_color_lim
    upper_color_lim = np.percentile(data[color_column], 99) if upper_color_lim is None else upper_color_lim
    color_mapper = LinearColorMapper(palette = "RdYlGn8", low = lower_color_lim,
                                                        high = upper_color_lim,
                )

    # CREATE THE PLOT

    # Setting coordinate system
    inProj = pyproj.CRS('epsg:4326') 
    outProj = pyproj.CRS('epsg:3857')

    transform_to_lat_lon = pyproj.Transformer.from_crs(inProj, outProj, always_xy = True)
    nyc_lon1, nyc_lat1 = transform_to_lat_lon.transform(nyc_long_limits[0], nyc_lat_limits[0])
    nyc_lon2, nyc_lat2 = transform_to_lat_lon.transform(nyc_long_limits[1], nyc_lat_limits[1])

    map_tile_p = get_provider(map_tile_type) #CARTODBPOSITRON or WIKIMEDIA
    map_tile_d = get_provider(map_tile_type) #CARTODBPOSITRON or WIKIMEDIA

    # Setting common figure kwargs
    common_kwargs = {
                    'plot_width': width, 'plot_height': height,
                    'x_range': (nyc_lon1, nyc_lon2), 
                    'y_range': (nyc_lat1, nyc_lat2),
                    'x_axis_type': "mercator", 'y_axis_type': "mercator",
                    'tooltips': [
                            ("Pickup Zone", "@zone_name_p"), ("Dropoff Zone", "@zone_name_d"), 
                            ("Trip duration (mts)", "@color")
                            ],
                    'tools': "pan, box_select, wheel_zoom, box_zoom, reset",
                    'active_drag': "box_select",
                    'active_inspect': None
    }
    
    # Pickup locations
    plotP = figure(
            title = "NYC taxi pickup locations {}".format(
                                                    data.pickup_datetime.dt.year.unique().tolist()),
            **common_kwargs
            )

    circleP = Circle(x = "lonp", y = "latp",
            size = 'size',
            fill_color={'field': 'color', 'transform': color_mapper}, line_color = None,
            fill_alpha = fill_alpha)

    plotP.add_tile(map_tile_p)

    # Dropoff locations
    plotD = figure(
        title = "NYC taxi dropoff locations {}".format(
                                                    data.dropoff_datetime.dt.year.unique().tolist()),
            **common_kwargs 
                )

    circleD = Circle(x = "lond", y = "latd",
            size = 'size',
            fill_color={'field': 'color', 'transform': color_mapper}, line_color = None,
            fill_alpha = fill_alpha)

    plotD.add_tile(map_tile_d)

    # Only the user selected scatter points will be highlighted
    nonselected_circle = Circle(fill_alpha = 0.1, fill_color = "grey", line_color = None)
    
    plotP.add_glyph(source, circleP, nonselection_glyph = nonselected_circle)
    plotD.add_glyph(source, circleD, nonselection_glyph = nonselected_circle)

    # Adding color bar
    color_bar = ColorBar(color_mapper = color_mapper, ticker = BasicTicker(),
                        label_standoff = 12, border_line_color = None, location = (0,0), title = color_column)

    plotP.add_layout(color_bar, 'left')
    plotD.add_layout(color_bar, 'right')

    layout = gridplot([plotP, plotD], ncols = 2, sizing_mode = "scale_both", merge_tools = False) 
    
    return(layout) 


def plot_src_to_dest_arrows(data: pd.DataFrame, 
                        latitude_column: list[str] = ['pickup_MercatorY', 'dropoff_MercatorY'], 
                        longitude_column: list[str] = ['pickup_MercatorX', 'dropoff_MercatorX'],
                        pickup_zone_name_col = 'pickup_zone_name', 
                        dropoff_zone_name_col = 'dropoff_zone_name',
                        pickup_or_dropoff = 'pickup_zone_name',
                        map_tile_type: bokeh.tile_providers = CARTODBPOSITRON,
                        nyc_long_limits: tuple = (-74.257159, -73.699215), 
                        nyc_lat_limits: tuple = (40.471021, 40.987326),
                        custom_title: str|None = None,
                        color_column: str = 'trip_duration_minutes', 
                        lower_color_lim: float|None = None, upper_color_lim: float|None = None,
                        size_column: float| tuple[float, float] | str = 5.0,
                        width: int = 800, height: int = 700, 
                        line_width = 2, fill_alpha: float = 1) -> bokeh.plotting.figure:
        
    """
    Plots interactive lines from taxi pickup zone/s to dropoff zone/s on bokeh tile maps.

    Arguments:
        data: pandas dataframe with latitude and longitude columns listed below
        latitude_column: a list of latitude columns. The index 0 column serves as the y axis for the left plot, and index 1 column for the right plot. 
        longitude_column: a list of longitide columns. The index 0 column serves as the x axis for the left plot, and index 1 column for the right plot. 
        pickup_zone_name_col: column with pickup zone names
        dropoff_zone_name_col: columsn with dropoff zone names
        pickup_or_dropoff: which zone to focus on -> pickup_zone_name or dropoff_zone_name. If the data is filtered to focus on trips coming to a particular zone, then pass 'pickup_zone_name',..
            else, pass 'dropoff_zone_name'.
        map_tile_type: one of the options from bokeh.tile_providers
        nyc_long_limits: x axis limits for both the left and right plots
        nyc_lat_limits: y axis limits for both the left and right plots
        custom_title: title for the figure
        color_column: column to use for assigning colors to the points
        lower_color_lim: lower limit for the colorbar
        upper_color_lim: upper limit for the color bar
        size_column: column used to determine the size of the scatter points. Can be constant scaler or any dataframe column.
        width: width of the plots
        height: height of the plots
        line_width: width of lines plotted from source to destination. 
        fill_alpha: transparency (1 being complete opaque, 0 being all transparent)
    Returns:
        Bokeh bokeh.plotting.figure plot
        """

    data = data.copy()
    years_trips = data.pickup_datetime.dt.year.unique().tolist() # to be used in the title

    # Calculating the mean trip duration mts (color column), and the mean lats and lons for pickup and dropoff locations per pickup and dropoff zones respectively
    lat_lon_df = data.groupby(by = [pickup_zone_name_col, dropoff_zone_name_col]).mean()\
                        [[color_column, *latitude_column, *longitude_column]]

    # Counting the number of trips per pair of pickup and dropoff zones to be used in tooltips
    count_df = data.groupby(by = [pickup_zone_name_col, dropoff_zone_name_col]).count()[[color_column]].rename(columns={color_column : 'Count'})

    data = count_df.join(lat_lon_df).reset_index()
    #data = lat_lon_df[lat_lon_df[pickup_zone_name_col].isin(['JFK Airport'])] #['JFK Airport', 'LaGuardia Airport', 'Newark Airport']

    if not isinstance(size_column, str):
        if isinstance(size_column, tuple): 
            s_lower, s_upper = size_column
            size_column = 'constant'
            data[size_column] = ((s_upper - s_lower) * (data['Count'] - data['Count'].min()))\
                                / (data['Count'].max() - data['Count'].min()) + s_lower
        else:
            s = size_column
            size_column = 'constant'
            data[size_column] = s

    print(f"trips df: {data.info()} \n")

    # Defining the list of lines for source to destination
    lines_x, lines_y = [], []

    for lon_dest, lat_dest, lon_orig, lat_orig in data[[longitude_column[1], latitude_column[1], longitude_column[0], latitude_column[0]]].values:
        lines_x.append([lon_orig, lon_dest])
        lines_y.append([lat_orig, lat_dest])

    # Defining the data source
    source = ColumnDataSource(
        data = dict(
            latp = data[latitude_column[0]].tolist(),
            lonp = data[longitude_column[0]].tolist(),
            latd = data[latitude_column[1]].tolist(),
            lond = data[longitude_column[1]].tolist(),
            Count = data["Count"].tolist(),
            trip_duration_minutes = data[color_column].tolist(),
            zone_name_p = data[pickup_zone_name_col].tolist(),
            zone_name_d = data[dropoff_zone_name_col].tolist(),
            xs = lines_x,
            ys = lines_y,
            size = data[size_column].tolist(),
            color = data[color_column].tolist()
        )
    )

    # Add color mapper for the color bar
    lower_color_lim = np.percentile(data[color_column], 1) if lower_color_lim is None else lower_color_lim
    upper_color_lim = np.percentile(data[color_column], 99) if upper_color_lim is None else upper_color_lim
    color_mapper = LinearColorMapper(palette = "RdYlGn8", low = lower_color_lim,
                                                        high = upper_color_lim,
                )

    # Setting coordinate system
    inProj = pyproj.CRS('epsg:4326') 
    outProj = pyproj.CRS('epsg:3857')

    transform_to_lat_lon = pyproj.Transformer.from_crs(inProj, outProj, always_xy = True)
    nyc_lon1, nyc_lat1 = transform_to_lat_lon.transform(nyc_long_limits[0], nyc_lat_limits[0])
    nyc_lon2, nyc_lat2 = transform_to_lat_lon.transform(nyc_long_limits[1], nyc_lat_limits[1])

    # Setting the map tile to plot on
    map_tile_d = get_provider(map_tile_type)

    # Setting common figure kwargs
    common_kwargs = {
                    'plot_width': width, 'plot_height': height,
                    'x_range': (nyc_lon1, nyc_lon2), 
                    'y_range': (nyc_lat1, nyc_lat2),
                    'x_axis_type': "mercator", 'y_axis_type': "mercator",
                    'tooltips': [
                        ("Source", "@zone_name_p"), ("Destination", "@zone_name_d"), 
                        ("Count", "@Count"), ("Avg. trip duration mts", "@trip_duration_minutes")
                        ],
                    'tools': "pan, box_select, wheel_zoom, box_zoom, reset",
                    'active_drag': "box_select",
                    'active_inspect': None
    }

    # Create the plot
    custom_title = f"NYC taxi trips to/from the selected zones {years_trips}" if custom_title is None else custom_title
    p = figure(title = custom_title, **common_kwargs)
    p.add_tile(map_tile_d)

    # Adding the lines from source to destination
    p.multi_line('xs', 'ys', source = source, line_width = line_width, 
                color = linear_cmap('color', "RdYlGn8", 
                        low = lower_color_lim,
                        high = upper_color_lim
                        )
            )

    # Adding the color bar for reference
    color_bar_title = ' '.join([str(i).capitalize() for i in color_column.split('_')]) if isinstance(color_column, str) else color_column
    color_bar = ColorBar(color_mapper = color_mapper, ticker = BasicTicker(),
                        label_standoff = 12, border_line_color = None, location = (0,0), title = color_bar_title)

    # Adding the pickup locations
    fill_dict = {'field': 'color', 'transform': color_mapper} if pickup_or_dropoff == dropoff_zone_name_col else 'blue'
    size = "size" if pickup_or_dropoff == dropoff_zone_name_col else 6
    p.circle(x = "lonp", y = "latp",
        size = size, alpha = fill_alpha, 
        fill_color = fill_dict,
        source = source)

    # Adding the dropoff locations
    fill_dict = {'field': 'color', 'transform': color_mapper} if pickup_or_dropoff == pickup_zone_name_col else 'blue'
    size = "size" if pickup_or_dropoff == pickup_zone_name_col else 6
    p.circle(x = "lond", y = "latd",
            size = size, alpha = fill_alpha, 
            fill_color = fill_dict,
            source = source)

    p.add_layout(color_bar, 'right')

    return(p) 
