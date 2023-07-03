# Author: pratha19/Github

"""
Creates Interactive Visulaization of NYC taxi trips (pickup and dropoffs) using Bokeh. 
"""

import sys
sys.path.append('NYC')

# Importing the required packages
import pandas as pd
from NYC.plot_figs import plot_cartmaps_streamlit, plot_gmaps_streamlit, plot_src_to_dest_arrows
from bokeh.tile_providers import WIKIMEDIA, CARTODBPOSITRON
from bokeh.layouts import row, gridplot
import streamlit as st


# ===========================================================================================================


class StreamlitMaps:

    def __init__(self, nyc: pd.DataFrame):
        """
        Initialize with the dataframe to work on. 
        """

        self.nyc = nyc


    def add_sidebar_pickup_or_dropoff(self, options = ['pickup_zone', 'dropoff_zone']):
        """
        Add a streamlit sidebar to either filter the pickup zones based on selected zone or the dropoff zones.
        """

        pickup_or_dropoff_ed = st.sidebar.selectbox(label = 'Select to focus on either the pickup or dropoff zones', 
                                        options = options, index = 0)
        pickup_or_dropoff = f"{pickup_or_dropoff_ed}_name" # this is how the column is named in the df

        return pickup_or_dropoff_ed, pickup_or_dropoff


    def add_sidebar_hour(self, default_hour = 17):
        """
        Add a streamlit sidebar to filter using the hour of the day.
        """

        show_all_hours = st.sidebar.checkbox(label = f"Show for all hours", value = False)

        if not show_all_hours :
            slider_hour = [st.sidebar.slider(label = 'Hour of the day', min_value = 0, max_value = 23, 
                                        value = default_hour, step = 1)]
        else:
            slider_hour = range(0, 24)

        return slider_hour

    
    def add_sidebar_weekday(self):
        """
        Add a streamlit sidebar to filter using week of the day, or to select all weekdays.
        """

        show_all_weekdays = st.sidebar.checkbox(label = f"Show for all weekdays", value = True)

        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if not show_all_weekdays :
            checkbox_weekday = st.sidebar.multiselect("Select the day of the week", options = weekdays, default = ['Saturday', 'Sunday']) 
        else:
            checkbox_weekday = weekdays
        
        return checkbox_weekday


    def add_sidebar_select_zone(self, pickup_or_dropoff_ed: str, pickup_or_dropoff: str, show_all_zones_default: bool = True, 
                                default_zones: list[str] = ['JFK Airport', 'LaGuardia Airport', 'Newark Airport']):
        """
        Add a streamlit sidebar to select from multiple taxi zones to focus on, or select all zones.
        """

        unique_taxi_zones = list(set(list(self.nyc['pickup_zone_name'].unique()) + list(self.nyc['dropoff_zone_name'].unique())))
        unique_taxi_zones = sorted([zone_i if isinstance(zone_i, str) else 'Unknown' for zone_i in unique_taxi_zones])

        show_all_zones = st.sidebar.checkbox(label = f"Show all zones in NYC: {pickup_or_dropoff_ed}", 
                                    value = show_all_zones_default)

        default_zones = default_zones if(all(x in unique_taxi_zones for x in default_zones)) else [] # Avoiding a case where one of the defaults isn't present in the data

        if not show_all_zones:
            zone_name_list = st.sidebar.multiselect(f"Select {pickup_or_dropoff} taxi zones to focus on", options = unique_taxi_zones,
                                            max_selections = None, default = default_zones)
        else:
            zone_name_list = unique_taxi_zones

        return zone_name_list


    def streamlit_points(self, is_gmaps: bool = False, api_key: str = None):
        """
        Plot a bokeh tile map or google maps figure. 
        Arguments:
            is_gmaps: if True and api_key non None, uses Google maps as the background for the plots. 
                else uses bokeh tile map as the background. 
            api_key: Google maps API key to  be used if plotting on Google maps background. 
        Returns:
            A bokeh layout of left and right plots. 
            Left plot: Taxi trip pickups in the selected taxi zones, if the zone filter is pickup_zone, 
                or else, pickup locations for all trips terminating in the dropoff locations in the right chart. 
            Right plot: Taxi trip dropoffs in the selected taxi zones, if the zone filter is dropoff_zone, 
                or else, dropoff locations for all trips originating from the pickup locations in the left chart. 
        """

        st.sidebar.write(":blue[USER SELECTION OPTIONS]")

        # Selecting the pickup or dropoff zones to visualize
        pickup_or_dropoff_ed, pickup_or_dropoff = self.add_sidebar_pickup_or_dropoff()

        # Slider for selecting hour of the day
        slider_hour = self.add_sidebar_hour()

        # Selecting the weekday
        checkbox_weekday = self.add_sidebar_weekday()

        # Getting a list of the available taxi zones, for the user to select from 
        zone_name_list = self.add_sidebar_select_zone(pickup_or_dropoff_ed, pickup_or_dropoff)

        # Filtering the data based on user selection
        print('Filtering the dataframe based on user selection...')
        df_to_plot = self.nyc[(self.nyc[pickup_or_dropoff].isin(zone_name_list)) \
                        & (self.nyc['pickup_weekday'].isin(checkbox_weekday)) \
                        & (self.nyc['pickup_hour'].isin(slider_hour)) \
                        ]
        
        print('Done filtering the data')
        print(f"Number of trips selected: {df_to_plot.shape[0]}, out of {self.nyc.shape[0]}")

        # Plotting the bokeh plot
        print('Plotting the bokeh maps plot now....')

        if is_gmaps:
            assert api_key is not None, "in order to use google maps, you need to pass in a valid google maps api key."

            intrc_trips_loc_p = plot_gmaps_streamlit(
                                        df_to_plot, 
                                        latitude_column= ['pickup_latitude', 'dropoff_latitude'], 
                                        longitude_column = ['pickup_longitude', 'dropoff_longitude'],
                                        color_column = 'trip_duration_minutes', size_column = 5.0,
                                        api_key = api_key, map_type = 'roadmap', map_zoom = 10,
                                        width = 700, height = 600
                                        )
        else:
            intrc_trips_loc_p = plot_cartmaps_streamlit(
                                    data = df_to_plot, 
                                    latitude_column = ['pickup_MercatorY', 'dropoff_MercatorY'], 
                                    longitude_column= ['pickup_MercatorX', 'dropoff_MercatorX'], 
                                    map_tile_type = CARTODBPOSITRON, 
                                    nyc_long_limits = (-74.257159, -73.677215), #(-74.257159, -73.699215), 
                                    nyc_lat_limits = (40.471021, 40.987326), #(40.471021, 40.987326)
                                    color_column = 'trip_duration_minutes',
                                    size_column = 6, 
                                    width = 700, height = 600
                                    )
        
        return intrc_trips_loc_p


    def add_notes_points(self):
        """
        Adding notes to help the user better navigate the streamlit points plot
        """

        st.markdown("\n")
        st.markdown("### :red[Notes]")
        st.markdown("- The left plot shows the pickup locations in the selected taxi zones for all trips, if the selected focus zone filter is pickup_zone \
            or else, it shows the corresponding pickup locations for only those trips that end in the dropoff locations shown in the right plot")
        st.markdown("- The right plot shows the dropoff locations in the selected taxi zones for all trips, if the selected focus zone filter is dropoff_zone \
            or else, it shows the corresponding dropoff locations for only those trips that start from the pickup locations shown in the left plot")
        st.markdown("- The circles are colored per the trip duration colorbar.")
        st.markdown("- Use the Box Select tool to select a subset of the map area to focus on. The left and right plots are interconnected. So, say, you \
            select an area on the left plot to focus on, the corresponding points on the right plot will be highlighted as well.")
        st.markdown("- Select the Hover tool to show the details when you hover over the points on the map.")
        st.markdown("- Select the Box Zoom tool to zoom in on any specific area on the map.")
    
    
    def add_sidebar_point_size(self, default_fixed_size: bool = False, min_value: int = 2, max_value: int = 30, 
                                default_fixed_value: int = 5, default_size_range: list[int] = [5, 15]):
        """
        Add a streamlit sidebar to allow the user to adjust the size of the plotted points on the map. 
        """

        fixed_size = st.sidebar.checkbox(label = f"Show all the zone points at the same size", value = default_fixed_size)

        if fixed_size:
            size_tuple = st.sidebar.slider('Select the size of the zone point circles',
                                        min_value = min_value, max_value = max_value, value = default_fixed_value)
        else:
            size_tuple = st.sidebar.select_slider('Select lower and upper limits for size of the zone point circles. The size indicates the number of trips.',
                                            options = range(min_value, max_value+1), value = default_size_range)

        return size_tuple


    def add_sidebar_colorlim(self, default_auto_color_lim: bool = False, min_val: int = 0, max_value: int = 120,
                        default_range: list[int] = [0, 60]):
        """
        Add a streamlit sidebar to allow the user to adjust the scale of the colorbar on the map. 
        """

        auto_color_lim = st.sidebar.checkbox(label = f"Select auto lower and upper limits for colorbar", value = default_auto_color_lim)

        if auto_color_lim:
            lower_color_lim, upper_color_lim = None, None
        else:
            lower_color_lim, upper_color_lim = st.sidebar.select_slider('Select lower and upper limits of the color bar',
                                                    options = range(min_val, max_value+1), value = default_range)

        return lower_color_lim, upper_color_lim


    def streamlit_lines(self):
        """
        Plot a bokeh tile map figure. 
        Returns:
            A bokeh layout with a left and right plot. 
            Left plot: shows lines connecting the dropoff locations of the taxi trips originating from the selected
                taxi zone/s by the user. 
            Right plot: shows lines connecting the pickup locations of the taxi trips terminating at the selected
                taxi zone/s by the user. 
        """

        st.sidebar.write(":blue[USER SELECTION OPTIONS]")

        # Slider for selecting hour of the day
        slider_hour = self.add_sidebar_hour()

        # Selecting the weekday
        checkbox_weekday = self.add_sidebar_weekday()

        # Getting a list of the available taxi zones, for the user to select from 
        zone_name_list = self.add_sidebar_select_zone(pickup_or_dropoff_ed = '', pickup_or_dropoff = '', 
                                            show_all_zones_default = False, 
                                            default_zones = ['JFK Airport', 'LaGuardia Airport', 'Newark Airport'])

        # Allow the user to change the size of the scatter points on the map
        size_tuple = self.add_sidebar_point_size()

        # Allow the user to change the colorbar's lower and upper limit
        lower_color_lim, upper_color_lim = self.add_sidebar_colorlim()

        # Filtering the data based on user selection
        print('Filtering the dataframe based on user selection...')
        df_to_plot_pickup = self.nyc[(self.nyc['pickup_zone_name'].isin(zone_name_list)) \
                        & (self.nyc['pickup_weekday'].isin(checkbox_weekday)) \
                        & (self.nyc['pickup_hour'].isin(slider_hour)) \
                        ]
                        
        df_to_plot_dropoff = self.nyc[(self.nyc['dropoff_zone_name'].isin(zone_name_list)) \
                        & (self.nyc['pickup_weekday'].isin(checkbox_weekday)) \
                        & (self.nyc['pickup_hour'].isin(slider_hour)) \
                        ]
        
        print('Done filtering the data')
        print(f"Number of pickup trips selected: {df_to_plot_pickup.shape[0]:,} (out of {self.nyc.shape[0]:,})")
        print(f"Number of dropoff trips selected: {df_to_plot_dropoff.shape[0]:,} (out of {self.nyc.shape[0]:,})")
        st.markdown(f"##### Total number of trips: {self.nyc.shape[0]:,}")

        # Plotting the bokeh plot
        print('Plotting the bokeh maps plot now....')
        common_kwargs = {'latitude_column': ['pickup_MercatorY', 'dropoff_MercatorY'], 
                        'longitude_column': ['pickup_MercatorX', 'dropoff_MercatorX'], 
                        'pickup_zone_name_col': 'pickup_zone_name', 
                        'dropoff_zone_name_col': 'dropoff_zone_name', 
                        'map_tile_type': CARTODBPOSITRON, 
                        'nyc_long_limits': (-74.25, -73.7),
                        'nyc_lat_limits': (40.5, 40.9), 
                        'color_column': 'trip_duration_minutes', 
                        'lower_color_lim': lower_color_lim, 'upper_color_lim': upper_color_lim,
                        'size_column': size_tuple,
                        'width': 800, 'height': 700, 
                        'line_width': 2, 'fill_alpha': 1
        }

        # Plotting the lines from the selected taxi zone to other zone/s
        lines_p_pickup = plot_src_to_dest_arrows(data = df_to_plot_pickup, 
                                        pickup_or_dropoff = 'pickup_zone_name',
                                        custom_title = "Trips starting from the selected taxi zone/s (Blue circles are pickup points) \n"
                                        f"Number of trips selected: {df_to_plot_pickup.shape[0]:,}", 
                                        **common_kwargs
                                    )

        # Plotting the lines from other zones to the selected taxi zone/s
        lines_p_dropoff = plot_src_to_dest_arrows(data = df_to_plot_dropoff, 
                                        pickup_or_dropoff = 'dropoff_zone_name',
                                        custom_title = "Trips ending at the selected taxi zone/s (Blue circles are dropoff points) \n"
                                        f"Number of trips selected: {df_to_plot_dropoff.shape[0]:,}", 
                                        **common_kwargs
                                    )

        # Combining the two plots into a single layout
        layout = gridplot([lines_p_pickup, lines_p_dropoff], ncols = 2, 
                        sizing_mode = "scale_both", merge_tools = False) # using grid plot helps avoid the huge padding appplied by row() 

        return layout


    def add_notes_lines(self):
        """
        Adding notes to help the user better navigate the streamlit lines plot
        """

        st.markdown("\n")
        st.markdown("### :red[Notes]")
        st.markdown("- The left plot shows the trips starting from the selected taxi zone/s shown in blue, whereas the right plot shows the trips ending at the selected \
            taxi zone/s shown in blue.")
        st.markdown("- The lines and circles (except the blue ones) are colored per the trip duration colorbar.")
        st.markdown("- To get an idea of the count of trips that each location/circle represents, you can play with the size selector. \
            The lower and upper size limits are there to ensure that the small sized circles don't get too small and the large ones don't get too large.")
        st.markdown("- Use the Box Select tool to select a subset of taxi zones to focus on.")
        st.markdown("- Select the Hover tool to show the details when you hover over the points on the map.")
        st.markdown("- Select the Box Zoom tool to zoom in on any specific area on the map.")