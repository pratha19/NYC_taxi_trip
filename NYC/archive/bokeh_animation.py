"""
Animating the bokeh NYC trip plot. 
"""

# Importing libraries
import sys
sys.path.append('NYC')

from NYC.data_fetch_n_save import read_parquet_file
from bokeh.models import Button, Slider, ColumnDataSource, Circle, ColorBar, BasicTicker
from bokeh.plotting import figure, column, row
from bokeh.layouts import column, row, gridplot
from bokeh.io import curdoc
from bokeh.models.mappers import LinearColorMapper
import pyproj
from bokeh.tile_providers import get_provider, CARTODBPOSITRON


# Reading the file
FINAL_FILE_LOCATION = 'data/processed/nyc_with_zones_animation.parquet'
nyc = read_parquet_file(FINAL_FILE_LOCATION)

# Filtering based on location
data = nyc[nyc['pickup_zone_name'] == 'JFK Airport']

# Defining fields to be used in the plot     
latitude_column = ['pickup_MercatorY', 'dropoff_MercatorY']
longitude_column = ['pickup_MercatorX', 'dropoff_MercatorX']
map_tile_type = CARTODBPOSITRON
nyc_long_limits = (-74.257159, -73.699215)
nyc_lat_limits = (40.471021, 40.987326)
color_column = 'trip_duration_minutes'
size_column = 6.0
width = 700
height = 600
fill_alpha = 1
lower_color_lim = 0
upper_color_lim = 60

if not isinstance(size_column, str):
    s = size_column
    size_column = 'constant'
    data[size_column] = s

# Defining the data source
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
color_mapper = LinearColorMapper(palette = "RdYlGn8", low = lower_color_lim,
                                                    high = upper_color_lim,
            )

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

## Define Callbacks
curr_hour = 0

def update_chart():
    global curr_hour
    if curr_hour == 24:
        curr_hour = 0
    slider.value = curr_hour
    curr_hour += 1

layout = gridplot([plotP, plotD], ncols = 2, sizing_mode = "scale_both", merge_tools = False) 

## Define Widgets
slider = Slider(start = 0, end = 23, value = 0, step = 1, title = "Hour of day")
btn = Button(label = "Play")

callback = None
def execute_animation():
    global callback
    if btn.label == "Play":
        btn.label = "Pause"
        callback = curdoc().add_periodic_callback(update_chart, 750)
    else:
        btn.label = "Play"
        curdoc().remove_periodic_callback(callback)

def update_hour_slider(atrr, old, new): 
    hour = slider.value
    new_data = data[data.pickup_hour == hour]
    source.data = dict(
        latp = new_data[latitude_column[0]].tolist(),
        lonp = new_data[longitude_column[0]].tolist(),
        latd = new_data[latitude_column[1]].tolist(),
        lond = new_data[longitude_column[1]].tolist(),
        size = new_data[size_column].tolist(),
        color = new_data[color_column].tolist()
        )

## Register Callbacks
btn.on_click(execute_animation)
slider.on_change("value", update_hour_slider)

## GUI
curdoc().add_root(column(btn, slider, layout))