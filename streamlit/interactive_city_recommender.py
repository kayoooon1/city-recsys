"""
This script is for a streamlit app that visualizes a city recommendation system
based on feature values and weights input by the User.
This is part of the evaluated project component of the course "Data Literacy"
in WS 2023-24, at the University of Tuebingen, by Prof. Phillip Hennig.
Project Title: "Finding Home: Which is the best city?"
Contributors: Kayoon Kim, Jenny Lang, Joseph Wan Wang, Ashutosh Jha
"""

import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np
from tueplots import bundles
from tueplots.constants.color import rgb

plt.rcParams.update(bundles.beamer_moml())

st.set_page_config(
    page_title="Finding Your Best City",
    #figure out how to get this icon, preferably something different
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Philipp Hennig, 2023"},
)

st.sidebar.title("What is important for your best city?")

# Markdown - Quick text explainer on what we are doing in the graphs below.
#st.markdown(
#    """    
#    Evaluating the best cities for you based on feature importance specified in
#    the left!
#    """
#)

# ---------------------------------------------------------------------------------------
# Sidebar: parameters

avg_temp_user = st.sidebar.slider("What is a good average temperature (°C)?", 
        -10, 40, 15, help="What is your ideal average temperature for your ideal city year round?")

std_dev_temp_user = st.sidebar.slider("What is an acceptable range variance from above good average temperature (°C)?", 
        0, 30, 10, help="How much deviation from your average temperature is tolerable for you?")

# should I say very low cost of living instead of just low, I feel all of the
# below will entice a very important response from most people, but does it
# matter?

avg_temp_w = st.sidebar.slider("How important is the above average temperature for you?",
        0, 5, 2, help="(0 - Not at all, 1 - Most important)")
std_temp_w = st.sidebar.slider("How important is the above temperature variance range for you?",
        0, 5, 2, help="(0 - Not at all, 1 - Most important)")
cost_living_w = st.sidebar.slider("How important is low Cost of Living?", 
        0, 5, 2, help="(0 - Not at all, 1 - Most important)")
purchase_pow_w = st.sidebar.slider("How important is high Purchasing Power of the city currency?",
        0, 5, 2, help="(0 - Not at all, 1 - Most important)")
safety_w = st.sidebar.slider("How important is high standards of Safety?", 
        0, 5, 2, help="(0 - Not at all, 1 - Most important)")
pollution_w = st.sidebar.slider("How important is low levels of Pollution?", 
        0, 5, 2, help="(0 - Not at all, 1 - Most important)")
traffic_w = st.sidebar.slider("How important is low traffic wait times?", 
        0, 5, 2, help="(0 - Not at all, 1 - Most important)")

wht_dict = {"wt_mean_tmp":avg_temp_w,"wt_std_dev":std_temp_w,"wt_cost_living":cost_living_w,
        "wt_purchase_pow":purchase_pow_w,"wt_safety":safety_w,"wt_pollution":pollution_w,
        "wt_traffic":traffic_w}


# ---------------------------------------------------------------------------------------
# Get Data
all_data_df = pd.read_csv('all-data-best-city.csv')
if "Unnamed: 0" in list(all_data_df.columns):
    all_data_df = all_data_df.drop(['Unnamed: 0'], axis=1)
#print(list(all_data_df.columns))
#print(all_data_df.head())

# ---------------------------------------------------------------------------------------
# Functions?
def rank_eval(all_data_df, wht_dict, avg_temp_user, std_dev_temp_user):
    best_mean_tmp = avg_temp_user
    best_std_dev_tmp = std_dev_temp_user

    all_data_df['mean_tmp_diff_from_best'] = abs(best_mean_tmp - all_data_df['mean_tmp'])
    all_data_df['tmp_std_dev_diff_from_best'] = abs(best_std_dev_tmp - all_data_df['std_dev_temp'])

    #two ranking, one for average temperature, one for stand. dev.
    all_data_df['rank_mean_tmp'] = all_data_df['mean_tmp_diff_from_best'].rank(ascending=False)
    all_data_df['rank_std_dev_tmp'] = all_data_df['tmp_std_dev_diff_from_best'].rank(ascending=False)

    all_data_df['rank_cost_live_rent'] = all_data_df['cost_live_rent_index'].rank(ascending=False) #the lowest index has the higher score
    all_data_df['rank_purchase_pow'] = all_data_df['purchase_pow_index'].rank(ascending=True) #the lowest index has the lowest score
    all_data_df['rank_safety'] = all_data_df['safety_index'].rank(ascending=True) #the lowest index has the lowest score
    all_data_df['rank_pollution'] = all_data_df['pollution_index'].rank(ascending=False) #the lowest index has the higher score
    all_data_df['rank_traffic'] = all_data_df['trffic_min_index'].rank(ascending=False)
    
    # Calculating final score
    scores = []
    for index, row in all_data_df.iterrows():
        s = (( row['rank_cost_live_rent']*wht_dict['wt_cost_living'] + \
            row['rank_mean_tmp']*wht_dict['wt_mean_tmp'] + \
                row['rank_std_dev_tmp']*wht_dict['wt_std_dev'] + \
                    row['rank_safety']*wht_dict['wt_safety'] +\
                        row['rank_pollution']*wht_dict['wt_pollution'] +\
                            row['rank_traffic']*wht_dict['wt_traffic'] +\
                                row['rank_purchase_pow']*wht_dict['wt_purchase_pow']) / sum(wht_dict.values()))
        scores.append(s)

    finding_home_df = all_data_df[['city_ascii','country','lat','lng','mean_tmp','std_dev_temp']].copy()

    finding_home_df['final_score'] = scores

    finding_home_df = finding_home_df.sort_values(by=['final_score'], ascending=False)

    rnk = [i for i in range(1,finding_home_df.shape[0] + 1)]

    finding_home_df['Rank'] = rnk

    return finding_home_df

finding_home_df = rank_eval(all_data_df, wht_dict, avg_temp_user,std_dev_temp_user)
#print(finding_home_df.columns)
#print(finding_home_df.head())

finding_home_top_df = finding_home_df.head(10)


# ---------------------------------------------------------------------------------------
# Plotting
#fig = px.scatter_geo(finding_home_top5_df, lat='lat', lon='lng',color='city_ascii', title='Best Cities For You!')

# Making the plot fancy
#fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")

import plotly.graph_objects as go

# Create a numerical column 'city_id' that maps to 'city_ascii'
#finding_home_top_df.loc[:, 'city_id'] = pd.Categorical(finding_home_top_df['city_ascii']).codes + 1

fig = go.Figure(data=go.Scattergeo(
    lon = finding_home_top_df['lng'],
    lat = finding_home_top_df['lat'],
    # Add the index to the text
    text = 'Rank ' + finding_home_top_df['Rank'].astype(str) + ': ' + finding_home_top_df['city_ascii'] + ', ' + finding_home_top_df['country'],
    mode = 'markers',
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = False,
        symbol = 'circle',
        line = dict(
            width=1,
            color='rgba(102, 102, 102)'
        ),
        color = finding_home_top_df['Rank'],  # Use 'city_id' for color
        colorscale = 'Viridis',  # Use a predefined colorscale
        colorbar_title="Cities"
    )))


# Update geos
fig.update_geos(
    resolution=50,
    showcoastlines=True, coastlinecolor="RebeccaPurple",
    showland=True, landcolor="Brown",
    showocean=True, oceancolor="Azure",
    showlakes=True, lakecolor="LightBlue",
    showrivers=True, rivercolor="LightBlue"
)

fig.update_layout(
    title_text = 'Top 10 Cities For the Parameters Set By You.',
    geo = dict(
        scope='world',
        projection_type='equirectangular',
        showland = True,
        landcolor = "rgb(250, 250, 250)",
        countrycolor = "rgb(200, 200, 200)",
    ),
    autosize=False,
    width=950,  # Adjust the width
    height=700,  # Adjust the height
    uirevision='constant'
)


st.markdown("### Finding Home : Locating Your Best Cities!")
st.plotly_chart(fig)

# ---------------------------------------------------------------------------------------
# Top 10 Table
top_20_df_to_show = finding_home_top_df
top_20_df_to_show = top_20_df_to_show.drop(['lat','lng','final_score'],axis=1)
top_20_df_to_show['mean_tmp'] = top_20_df_to_show['mean_tmp'].round(1)
top_20_df_to_show['std_dev_temp'] = top_20_df_to_show['std_dev_temp'].round(1)
top_20_df_to_show = top_20_df_to_show.reindex(columns = ['Rank','city_ascii','country','mean_tmp','std_dev_temp'])
st.dataframe(top_20_df_to_show,
        hide_index=True,
        column_config={
            "Rank": "Rank",
            "city_ascii": "City Name",
            "country": "Country",
            "mean_tmp": st.column_config.NumberColumn(
                "Mean Temperature",
                format='%f °C'
                ),
            "std_dev_temp": st.column_config.NumberColumn(
                "Temerature Variance (From Average)",
                format='%f °C'
                )
            },
        width = 900
        )

# ---------------------------------------------------------------------------------------
# Math Explainer
