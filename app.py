import panel as pn
import holoviews as hv
import xarray as xr
import pandas as pd
import datetime as dt
import numpy as np
from datetime import date
from datetime import datetime
from holoviews import dim, opts
from bokeh.models import HoverTool
import ipywidgets as ipw
import panel.widgets as pnw
import plotly.express as px
from matplotlib import pyplot as plt
import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import geoviews as gv
import cartopy.crs as ccrs
import seaborn as sns
from matplotlib import colors as mcolors
import shutup
shutup.please()
hv.extension('bokeh')
pn.extension('tabulator', sizing_mode="stretch_width")

# from holoviews import opts
# hv.extension('bokeh')

## Make sure to upload 'Tmax_tmin.nc' to files on the left everytime notebook is opened again
data = xr.open_dataset('Tmax_Tmin.nc')
tmax = data.tmax
tmin = data.tmin
# data

time = data.time.values.tolist()
city = data.City.values.tolist()
lat = data.latitude.values
lon = data.longitude.values
regions = ['Northeast', 'Midwest', 'South', 'West']

# Using a dictionary to map a city to its region (South, West, Northeast, Midwest)
city_to_region = {
    'Birmingham': 'South', 'Mobile': 'South', 'Phoenix': 'West', 'Bullhead City': 'West', 'Little Rock': 'South',
    'Bentonville': 'South', 'Los Angeles': 'West', 'Redding': 'West', 'Denver': 'West', 'Colorado Springs': 'West',
    'Bridgeport': 'Northeast', 'Middletown': 'Northeast', 'Wilmington': 'South', 'Dover': 'South', 'Washington': 'Northeast',
    'Jacksonville': 'South', 'Pensacola': 'South', 'Atlanta': 'South', 'Savannah': 'South', 'Boise City': 'West',
    'Pocatello': 'West', 'Chicago': 'Midwest', 'Quincy': 'Northeast', 'Indianapolis': 'Midwest', 'Evansville': 'Midwest',
    'Des Moines': 'Midwest', 'Dubuque': 'Midwest', 'Wichita': 'Midwest', 'Shawnee': 'Midwest', 'Louisville': 'South',
    'Bowling Green': 'South', 'New Orleans': 'South', 'Bossier City': 'South', 'Portland': 'West', 'Baltimore': 'Northeast',
    'Frederick': 'Northeast', 'Boston': 'Northeast', 'Westfield': 'Midwest', 'Detroit': 'Midwest', 'Grand Rapids': 'Midwest',
    'Minneapolis': 'Midwest', 'Rochester': 'Northeast', 'Jackson': 'South', 'Southaven': 'South', 'Kansas City': 'Midwest',
    'Florissant': 'Midwest', 'Billings': 'West', 'Missoula': 'West', 'Omaha': 'Midwest', 'Lincoln': 'Midwest', 'Las Vegas': 'West',
    'Carson City': 'West', 'Manchester': 'Northeast', 'Nashua': 'Northeast', 'Newark': 'Northeast', 'Camden': 'Northeast',
    'Albuquerque': 'West', 'Las Cruces': 'West', 'New York': 'Northeast', 'Buffalo': 'Northeast', 'Charlotte': 'South',
    'Rocky Mount': 'South', 'Fargo': 'Midwest', 'Bismarck': 'Midwest', 'Columbus': 'Midwest', 'Warren': 'Midwest',
    'Oklahoma City': 'South', 'Tulsa': 'South', 'Medford': 'West', 'Philadelphia': 'Northeast', 'Pittsburgh': 'Northeast',
    'Providence': 'Northeast', 'Woonsocket': 'Northeast', 'Columbia': 'South', 'Charleston': 'South', 'Sioux Falls': 'Midwest',
    'Rapid City': 'Midwest', 'Memphis': 'South', 'Johnson City': 'South', 'Houston': 'South', 'Amarillo': 'South',
    'Salt Lake City': 'West', 'Spanish Fork': 'West', 'Burlington': 'Northeast', 'Virginia Beach': 'South', 'Roanoke': 'South',
    'Seattle': 'West', 'Spokane': 'West', 'Huntington': 'Midwest', 'Milwaukee': 'Midwest', 'La Crosse': 'Midwest',
    'Cheyenne': 'West', 'Casper': 'West', 'Toronto': 'Northeast'
}

df = tmax.to_dataframe().reset_index()
df_tmin = tmin.to_dataframe().reset_index()
df['tmin'] = df_tmin['tmin']
# df = df.loc[:,['City', 'time', 'tmax', 'tmin']]

#removing duplicate 'Portland' column
df = df.loc[:,~df.columns.duplicated()].copy()
df = df.rename(columns={'tmax': 'Max Temp (Fahrenheit)', 'tmin': 'Min Temp (Fahrenheit)'})

#Adding necessary columns for region
df['Region'] = df['City'].map(city_to_region)

#Adding necessary time columns
df['time'] = pd.to_datetime(df['time'])
df['Year-Month'] = df['time'].dt.strftime('%Y-%m')
df['Year'] = df['time'].dt.year.astype('int')
df['Month'] = df['time'].apply(lambda x: x.strftime('%B'))

#Adding columns for the Map visualization
df['Yearly Max'] = df.groupby(['City', 'Year'])['Max Temp (Fahrenheit)'].transform('max')
df['Yearly Min'] = df.groupby(['City', 'Year'])['Max Temp (Fahrenheit)'].transform('min')
df = df.rename(columns = {'latitude': 'Latitude', 'longitude': 'Longitude', 'Max Temp': 'Max Temp (Fahrenheit)', 'Min Temp': 'Min Temp (Fahrenheit)'})

# making DataFrame pipleine interactive
idf = df.interactive()
# idf.head()

#| title: Temperature Data Over Time
months = df['Month'].unique().tolist()
#Widget to select multiple cities
select_city = pn.widgets.MultiChoice(
    name='Select Cities',
    options=city,
    value = ['New York', 'Los Angeles']
)
#Widget to toggle max/min temperature
max_or_min = pn.widgets.RadioBoxGroup(name='Max/Min Temperature', options=['Max Temp (Fahrenheit)', 'Min Temp (Fahrenheit)'])
#Widget to select a month
select_month = pn.widgets.Select(
    name='Select Month',
    options=months+['All Months'],
    value = 'August'
)
#Widget to select number of years to include
numyears = pn.widgets.IntRangeSlider(
    name = "Year Range",
    start = 1950,
    end = 2021,
    value = (2016, 2021),
    step = 1
)
# ipipeline_1 = idf[(idf['City'].isin(select_city.param.value)) &
#                     (idf['Year'] >= numyears.param.value_start) &
#                     (idf['Year'] <= numyears.param.value_end)]

# line_plot = ipipeline_1.hvplot(x='time', y=max_or_min, by='City', line_width=0.8,
#                                 height=350, width=1500,
#                                 cmap='Accent', ylim=(-40, 120))


@pn.depends(select_city.param.value, max_or_min.param.value, numyears.param.value_start, numyears.param.value_end)
def create_line(cities, max_min, year_start, year_end):
    constrained = df[(df['City'].isin(cities)) & (df['Year'] >= year_start) &
                    (df['Year'] <= year_end)]
    upper = max(constrained[max_min])+5
    lower = min(constrained[max_min])-5
    fig = constrained.hvplot(x='time', y=max_or_min, by='City', line_width=0.8,
                                height=350, width=1500,
                                cmap='Accent', ylim=(-40, 120))
    return fig
                   
# layout = pn.Column(select_city, max_or_min, numyears, create_line)
# layout.servable()
# layout

#| title: Temperature Box Plot
pn.extension('plotly')
max_or_min = pn.widgets.RadioBoxGroup(name='Max/Min Temperature', options=['Max Temp (Fahrenheit)', 'Min Temp (Fahrenheit)'])

select_city = pn.widgets.MultiChoice(
    name='Select Cities',
    options=city,
    value = ['New York', 'Los Angeles']
)
select_month = pn.widgets.Select(
    name='Select Month',
    options=months+['All Months'],
    value = 'August'
)
numyears = pn.widgets.IntRangeSlider(
    name = "Data start date",
    start = 1950,
    end = 2021,
    value = (2016, 2021),
    step = 1,
)
max_or_min = pn.widgets.RadioBoxGroup(name='Max/Min Temperature', options=['Max Temp (Fahrenheit)', 'Min Temp (Fahrenheit)'])

@pn.depends(select_city.param.value, max_or_min.param.value, numyears.param.value_start, numyears.param.value_end)
def create_line(cities, max_min, year_start, year_end):
    constrained = df[(df['City'].isin(cities)) & (df['Year'] >= year_start) &
                    (df['Year'] <= year_end)]
    upper = max(constrained[max_min])+5
    lower = min(constrained[max_min])-5
    fig = constrained.hvplot(x='time', y=max_or_min, by='City', line_width=0.8,
                                height=350, width=1500,
                                cmap='Accent', ylim=(-40, 120))
    return fig

@pn.depends(select_city.param.value, select_month.param.value, max_or_min.param.value, numyears.param.value_start, numyears.param.value_end)
def create_box(cities, month, max_min, year_start, year_end):
    if month !='All Months':
        constrained = df[(df['City'].isin(cities)) & (df['Year'] >= year_start) & (df['Year'] <= year_end) & (df['Month']==month)]
    else:
        constrained = df[(df['City'].isin(cities)) & (df['Year'] >= year_start) & (df['Year'] <= year_end) & (df['Month'].isin(months))]
    upper = max(constrained[max_min])+5
    lower = min(constrained[max_min])-5
    fig = px.box(constrained, x = 'City', y = max_min, color='City', points='all', boxmode='overlay', height=700)
    if month =='All Months':
        point_size = (3 - (2021 - year_start)*0.01)
    else:
        point_size = (5 - (2021 - year_start)*0.005)
    fig.update_traces(jitter=0.3+((2021-year_start)*0.005),
                      opacity=0.8,
                      pointpos=0,
                      marker_size=point_size
                     )
    fig.update_yaxes(range = [-10, 125])
    fig.update_layout(width=700, height=600)
    return fig

@pn.depends(select_city.param.value, select_month.param.value, max_or_min.param.value, numyears.param.value_start, numyears.param.value_end)
def create_hist(cities, month, max_min, year_start, year_end):
    if month !='All Months':
        constrained = df[(df['City'].isin(cities)) & (df['Year'] >= year_start) &
                    (df['Year'] <= year_end) & (df['Month']==month)]
    else:
        constrained = df[(df['City'].isin(cities)) & (df['Year'] >= year_start) &
                    (df['Year'] <= year_end) & (df['Month'].isin(months))]
    upper = max(constrained[max_min])+5
    lower = min(constrained[max_min])-5
    fig = px.histogram(constrained, x=max_min, color="City", marginal="rug", hover_data=constrained.columns)
    fig.update_layout(width=700, height=600, barmode='overlay')
    fig.update_traces(opacity=0.75)
    return fig

grouped_df = df.groupby(['Year-Month', 'Region']).agg({'Year': 'first', 'Max Temp (Fahrenheit)': 'mean', 'Min Temp (Fahrenheit)': 'mean', "time": "first"}).reset_index()
grouped_df.rename(columns={'Max Temp (Fahrenheit)': 'Max Temp Mean', 'Min Temp (Fahrenheit)': 'Min Temp Mean'}, inplace=True)
grouped_idf = grouped_df.interactive()

# converting grouped_df_tmax to datetime
grouped_df['Year-Month'] = pd.to_datetime(grouped_df['Year-Month'])
# grouped_idf.head()

df_grouped2 = df.groupby(["Year", "City"]).agg({'Max Temp (Fahrenheit)': 'mean', 'Min Temp (Fahrenheit)': 'mean', 'Region': 'first'}).reset_index()
df_grouped2.rename(columns={'Max Temp (Fahrenheit)': 'Max Temp Mean', 'Min Temp (Fahrenheit)': 'Min Temp Mean'}, inplace=True)
idf_grouped2 = df_grouped2.interactive()
# idf_grouped2.head()

# create another numyears slider
numyears_2 = pn.widgets.IntSlider(
    name="Year",
    start=1950,
    end=2021,
    value=2002,
    step=1,
)

# creating the points plot based on the selected number of years
@pn.depends(numyears_2.param.value)
def create_points(numyears_2):
    # Calculate the cutoff year
    cutoff_year = numyears_2

    # filtering based on the selected number of years
    filtered_data = df_grouped2[df_grouped2['Year'] == cutoff_year]

    # creating points plot
    points = hv.Points(
        filtered_data, ['Min Temp Mean', 'Max Temp Mean'],
        ["Year", "Max Temp Mean", "Min Temp Mean", "City", "Region"]
    ).sort('City')

    # defining tooltips for the hover tool
    tooltips = [
        ('City', '@City'),
        ('Year', '@Year'),
        ("Min Temp Mean", "@{Min Temp Mean}"),
        ("Max Temp Mean", "@{Max Temp Mean}")
    ]
    hover = HoverTool(tooltips=tooltips)

    # apply our options to the plot
    return points.opts(
        tools=[hover], color='Region',
        line_color='black', size=10, cmap='tab10',
        width=600, height=400, show_grid=True,
        title='Mean Temperature by Region',
        ylim=(40, 100),
        xlim=(25, 70)
    )

# # display plot and widgets
# layout = pn.Column(numyears_2, create_points)
# layout.servable()


df2 = data.to_dataframe()
df2 = df2.reset_index()
df2['year'] = df2['time'].dt.year
df2['Max Temp (Fahrenheit)'] = df2.groupby(['City', 'year'])['tmax'].transform('max')
df2['Min Temp (Fahrenheit)'] = df2.groupby(['City', 'year'])['tmin'].transform('min')
df2 = df2.drop(['time', 'tmax', 'tmin'], axis = 1)
df2 = df2.drop_duplicates()
df2 = df2.rename(columns = {'latitude': 'Latitude', 'longitude': 'Longitude', 'year': 'Year'})
df2 = df2.reset_index().drop(['index'], axis = 1)

year_slider = pn.widgets.IntSlider(name = 'Year', start=df['Year'].min(), end=df['Year'].max(), value=df['Year'].min(), step=1)
# temperature_type_dropdown = pn.widgets.RadioBoxGroup(name='Max/Min Temperature', options=['Max Temp (Fahrenheit)', 'Min Temp (Fahrenheit)'])

# idf2 = df2.interactive()
# ipipeline_2 = idf2[idf2['Year'] == year_slider]

@pn.depends(year_slider.param.value)
def create_min_temp_map(year_slider):
    constrained = df2[df2['Year'] == year_slider]
    fig = constrained.hvplot(
        'Longitude',
        'Latitude',
        geo=True,
        xlim=(-135, -65),
        ylim=(20, 50),
        c= 'Min Temp (Fahrenheit)',
        clim = (-40, 40),
        colorbar = True,
        cmap='Blues_r',
        title= f'Min Temperature Map',
        hover_cols=['City', 'Avg_Temperature'],
        tiles='CartoLight',
        kind = 'points',
        line_color='black',
        size = 50
    )
    return fig

@pn.depends(year_slider.param.value)
def create_max_temp_map(year_slider):
    constrained = df2[df2['Year'] == year_slider]
    fig = constrained.hvplot(
        'Longitude',
        'Latitude',
        geo=True,
        xlim=(-135, -65),
        ylim=(20, 50),
        c= 'Max Temp (Fahrenheit)',
        clim = (80, 120),
        colorbar = True,
        cmap='Reds',
        title= f'Max Temperature Map',
        hover_cols=['City', 'Avg_Temperature'],
        tiles='CartoLight',
        kind = 'points',
        line_color='black',
        size = 50
    )
    return fig

# layout = pn.Column(year_slider, pn.Row(create_min_temp_map, create_max_temp_map))
# layout.servable
# layout

df2 = data.to_dataframe()
df2 = df2.reset_index()
df2['year'] = df2['time'].dt.year
df2['Max Temp (Fahrenheit)'] = df2.groupby(['City', 'year'])['tmax'].transform('max')
df2['Min Temp (Fahrenheit)'] = df2.groupby(['City', 'year'])['tmin'].transform('min')
df2 = df2.drop(['time', 'tmax', 'tmin'], axis = 1)
df2 = df2.drop_duplicates()
df2 = df2.rename(columns = {'latitude': 'Latitude', 'longitude': 'Longitude', 'year': 'Year'})
df2 = df2.reset_index().drop(['index'], axis = 1)

year_slider = pn.widgets.IntSlider(name = 'Year', start=df['Year'].min(), end=df['Year'].max(), value=df['Year'].min(), step=1)
# temperature_type_dropdown = pn.widgets.RadioBoxGroup(name='Max/Min Temperature', options=['Max Temp (Fahrenheit)', 'Min Temp (Fahrenheit)'])

idf2 = df2.interactive()
ipipeline_2 = idf2[idf2['Year'] == year_slider]
min_plot = ipipeline_2.hvplot(
        'Longitude',
        'Latitude',
        geo=True,
        xlim=(-135, -65),
        ylim=(20, 50),
        c= 'Min Temp (Fahrenheit)',
        clim = (-40, 40),
        colorbar = True,
        cmap='Blues_r',
        title= f'Min Temperature Map',
        hover_cols=['City', 'Avg_Temperature'],
        tiles='CartoLight',
        kind = 'points',
        line_color='black',
        size = 40
    )

max_plot = ipipeline_2.hvplot(
        'Longitude',
        'Latitude',
        geo=True,
        xlim=(-135, -65),
        ylim=(20, 50),
        c= 'Max Temp (Fahrenheit)',
        clim = (80, 120),
        colorbar = True,
        cmap='Reds',
        title= f'Max Temperature Map',
        hover_cols=['City', 'Avg_Temperature'],
        tiles='CartoLight',
        kind = 'points',
        line_color='black',
        size = 40
    )
# pn.Column(min_plot, max_plot)

models = xr.open_dataset('Model_Percent_Change.nc')
# models

table = models.to_dataframe().reset_index()
table = table.loc[:, ~table.columns.duplicated()].copy()
table = table.drop(columns = ['lat', 'lon', 'quantile'], axis = 1)
# table.head(5)

#Creating widget to select the city
select_1 = pn.widgets.Select(name='Select City', options= sorted(table['City'].unique().tolist()))

#Function to create box plot
@pn.depends(select_1.param.value)
def create_plot(city):
    selected_city_table = table[table[('City')] == city]
    melted = selected_city_table.melt(id_vars = ['City', 'model'], var_name = 'Scenario', value_name = 'Percent Change')
    melted['Scenarios'] = melted['Scenario'].str.slice(-6)
    melted['Category'] = melted['Scenario'].str.slice(0, -14)
    melted['Percent Change'] = melted['Percent Change'] * 100
    limit = max(max(melted['Percent Change']), abs(min(melted['Percent Change'])))+2

    fig = px.box(melted, x = 'Category', y = 'Percent Change', color = 'Scenarios', points='all', width=1000, height = 600)
    fig.update_traces(boxpoints='all', jitter=0)
    fig.update_yaxes(range = [-limit, limit])
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = ['tmean', 'tmax', 'tmin', 'pr', 'heavy_pr', 'wind'],
            ticktext = ['Mean Temp', 'Max Temp (Fahrenheit)', 'Min Temp (Fahrenheit)', 'Precip', 'Heavy Precip', 'Wind']
        )
    )
    fig.update_layout(
        title_text = f'<b>{city} Box Plots</b>',
        title_x = .5,
        title_y = .95,
    )
    return fig

# layout = pn.Column(select_1, create_plot)
# layout.servable()
# layout

@pn.depends(select_1.param.value)
def create_barplot(city):
    sns.set_style("whitegrid")
    sns.set_context("notebook",font_scale=1.9)
    fig, ax = plt.subplots(figsize=(25, 13))

    df2 = table[table['City']==city]
    df2 = df2.iloc[:,2:]
    columns = df2.columns
    counter = 0
    colors = ['r', 'r', 'orange', 'orange', 'b', 'b', 'lightgreen', 'lightgreen', 'darkgreen', 'darkgreen', 'violet', 'violet'
              # ,'darkviolet', 'darkviolet', 'steelblue', 'steelblue', 'y', 'y'
             ]
    counters = [0.95, 1.3, 1.95, 2.3, 2.95, 3.3, 3.95, 4.3, 4.95, 5.3, 5.95, 6.3
                # , 7, 7.25, 8, 8.25, 9, 9.25
               ]
    for column, color, counter in zip(columns, colors, counters):
        plt.bar(counter, height = (df2[column].max() - df2[column].min()),
                  bottom = df2[column].min(), color = mcolors.to_rgba(color,  .4), linewidth=1, width=.25,
                  edgecolor = mcolors.to_rgba('k',  1))
        plt.scatter(np.repeat(counter, len(df2[column])), df2[column], linewidth=.5,
                    edgecolor = mcolors.to_rgba('k',  1), s = 150, color = mcolors.to_rgba(color,  .6), )

    # Middle section between 10 and -10
    ax.fill_between(x=np.arange(0, 12), y1=.1, y2=-.1, color='gray',  interpolate=True, alpha=.1, zorder = 1)
    ax.axhline(.1, linestyle='--', color='gray')
    ax.axhline(-.1, linestyle='--', color='gray')
    ax.axhline(0, linestyle='--', color='k')


    ax.fill_between(x=np.arange(0, 12), y1=.3, y2=.1, color='y',  interpolate=True, alpha=.2, zorder = 1)
    ax.fill_between(x=np.arange(0, 12), y1=-.1, y2=-.3, color='y',  interpolate=True, alpha=.2, zorder = 1)
    ax.fill_between(x=np.arange(0, 18), y1=.6, y2=.3, color='orange',  interpolate=True, alpha=.2, zorder = 1)
    ax.fill_between(x=np.arange(0, 18), y1=-.3, y2=-.6, color='orange',  interpolate=True, alpha=.2, zorder = 1)


    plt.xlim(0.625, 6.625)
    plt.ylim(-.6, .6)
    plt.yticks(np.arange(-.5, .6, 0.1), labels = np.arange(-50, 60, 10))

    plt.xticks(counters, labels = ['SSP126', 'SSP370','SSP126', 'SSP370','SSP126', 'SSP370','SSP126', 'SSP370',
                                   'SSP126', 'SSP370','SSP126', 'SSP370'
                                   # , 'SSP126', 'SSP370','SSP126', 'SSP370','SSP126', 'SSP370'
                                  ], rotation = 45)

    # xlabel_alt = ['SSP126', 'SSP370', 'SSP126', 'SSP370', 'SSP126', 'SSP370', 'SSP126', 'SSP370', 'SSP126', 'SSP370', 'SSP126', 'SSP370']
    # ax.set_xticklabels(labels=xlabel_alt, rotation=90, horizontalalignment='center')
    plt.grid(axis = 'x', which = 'major',color = 'k', linestyle = '-', linewidth = 1.5, alpha = 0)

    text = ['Tmax', 'Tmean', 'Tmin', 'Rain', 'Heavy Rain', 'Snow'
            # , 'Heavy Snow', ' Wind', 'Solar'
           ]
    counters2 = np.arange(.625, 10)

    for box, text_r in zip(counters2, text):
        ax.text(x=.5+box, y=0.65, s=text_r, fontsize=24, fontweight='bold', horizontalalignment='center', verticalalignment='center')
        ax.add_patch(plt.Rectangle((box,-.6), 1, 1.3, clip_on=False, fill=False, edgecolor='k', linewidth = 1.5, zorder=100000)) #(left, bottom, width, height)

    plt.xlabel('Climate Model')
    plt.ylabel('Percent Change by 2050')
    plt.title(city, y=1.1, fontweight = 'bold', fontsize = 28)
    plt.ioff()
    return fig


# layout = pn.Column(select_1, create_barplot)
# layout.servable()
# layout

#Creating widget to select the city
# select_2 = pn.widgets.Select(name='Select City', options= table['City'].unique().tolist())
# select_2
#Function to create heatmap
@pn.depends(select_1.param.value)
def create_heatmap(city):
    updated = table[table['City'] == city]
    cleaned = updated.loc[:, 'tmean_change_ssp126' : 'wind_change_ssp370']
    cleaned = cleaned.multiply(100)

    mean_values = np.mean(cleaned, axis=0)
    range_row = cleaned.max() - cleaned.min()
    cleaned = pd.DataFrame(np.vstack([cleaned.values, mean_values.values, range_row.values]), columns=cleaned.columns)
    cleaned['Index Title'] = ['GFDL', 'IPSL', 'MPI', 'MRI', 'UKESM', 'Ens. Mean', 'Model Range']
    cleaned.index = cleaned['Index Title']
    del cleaned['Index Title']
    arr = cleaned.values
    arr = np.round(arr, 0)

    rows = ['GFDL', 'IPSL', 'MPI', 'MRI', 'UKESM', 'Model Mean', 'Model Range']
    columns_unique = ['ssp126_tmean','ssp370_tmean','ssp126_tmax','ssp370_tmax','ssp126_tmin','ssp370_tmin','ssp126_pr','ssp370_pr','ssp126_hpr','ssp370_hpr','ssp126_wind','ssp370_wind']
    columns = ['ssp126', 'ssp370', 'ssp126 ', 'ssp370 ', 'ssp126  ', 'ssp370  ', 'ssp126   ', 'ssp370   ', 'ssp126    ', 'ssp370    ', 'ssp126     ', 'ssp370     ']
    fig = px.imshow(arr, text_auto=True, color_continuous_scale="RdBu_r", zmin = -15, zmax = 15, y=rows, x=columns)

    fig.add_annotation(x=0.5,y=-.75,text='<b>Tmean</b>', showarrow=False)
    fig.add_annotation(x=2.5,y=-.75,text='<b>Tmax</b>', showarrow=False)
    fig.add_annotation(x=4.5,y=-.75,text='<b>Tmin</b>', showarrow=False)
    fig.add_annotation(x=6.5,y=-.75,text='<b>Precip</b>', showarrow=False)
    fig.add_annotation(x=8.5,y=-.75,text='<b>Heavy Precip</b>', showarrow=False)
    fig.add_annotation(x=10.5,y=-.75,text='<b>Wind</b>', showarrow=False)
    fig.add_shape(type = 'line', x0 = -.5, x1 = -.5, y0 = -1, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = 1.5, x1 = 1.5, y0 = -1, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = 3.5, x1 = 3.5, y0 = -1, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = 5.5, x1 = 5.5, y0 = -1, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = 7.5, x1 = 7.5, y0 = -1, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = 9.5, x1 = 9.5, y0 = -1, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = 11.5, x1 = 11.5, y0 = -1, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = -.5, x1 = 11.5, y0 = 4.5, y1 = 4.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = -.5, x1 = 11.5, y0 = -1, y1 = -1, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = -.5, x1 = 11.5, y0 = -.5, y1 = -.5, line = dict(color = 'black', width = 4))
    fig.add_shape(type = 'line', x0 = -.5, x1 = 11.5, y0 = 6.5, y1 = 6.5, line = dict(color = 'black', width = 4))
    fig.update_layout(
        coloraxis_colorbar=dict(title='Percent Change by 2050', titleside = 'bottom'),
        title_text = f'<b>{city} Heatmap</b>',
        title_x = .5,
        title_y = .95,
        autosize=False,
        width=800,
        height=480
    )
    fig.update_xaxes(tickvals=np.arange(len(columns)), ticktext=columns)
    fig.update_yaxes(tickvals=np.arange(len(rows)), ticktext=rows)
    return fig

# layout = pn.Column(select_1, pn.Row(create_plot, create_heatmap))
# layout = pn.Column(select_1, create_heatmap)
# layout.servable()
# layout

row_23 = pn.Column(select_month, pn.Row(create_box, create_hist))
# row_23

# plots 1,2,3
col_1 = pn.Column(select_city, max_or_min, numyears, create_line)
col_1_row_23 = pn.Column(select_city, max_or_min, numyears, create_line, pn.Column(select_month, pn.Row(create_box, create_hist)))

# plots 1,2,3,4,5
row_45 = pn.Row(min_plot, max_plot)
col_1_row_23_col_45 = pn.Column(col_1_row_23, row_45)
# col_1_row_23_col_45

# plots 1,2,3,4,5,5,6,7
row_67 = pn.Column(select_1, pn.Row(create_barplot, create_heatmap))
dashboard = pn.Column(col_1_row_23_col_45, row_67)

template = pn.template.FastListTemplate(
    main=[dashboard]
)
template.show()