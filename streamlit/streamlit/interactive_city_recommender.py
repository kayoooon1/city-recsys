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
#from matplotlib import pyplot as plt
import plotly.express as px
import numpy as np
from pathlib import Path
from joblib import load
#from sklearn.decomposition import PCA
import plotly.graph_objects as go
#from tueplots import bundles
#from tueplots.constants.color import rgb

#plt.rcParams.update(bundles.beamer_moml())

st.set_page_config(
    page_title="Finding Your Best City",
    #figure out how to get this icon, preferably something different
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "(c) Ashutosh Jha, 2023"},
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

avg_temp_w = st.sidebar.slider("What is a good average temperature (°C)?", 
        0, 28, 15, help="What is your ideal average temperature for your ideal city year round?")

std_dev_temp_w = st.sidebar.slider("What is an acceptable range variance from above good average temperature (°C)?", 
        0, 10, 5, help="How much deviation from your average temperature is tolerable for you?")


# Markdown - Quick text explainer on what the scales below mean.
st.sidebar.markdown(
"""
For the scales below, indicate from 0.0 - 1.0 (worst to best), how high would you
want your ideal city to score. This could be based on how impotant the feature
in question is for you.
"""
)

cost_living_w = st.sidebar.slider("Cost of Living?", 
        0.0, 1.0, 0.5, help="(0 - Worst, 1 - Best. \nThe cost of living \
                index, includes general cost of living including rent! A low \
                score means, the city is less affordable, while a high score \
                means the city is more affordable. ")
purchase_pow_w = st.sidebar.slider("Purchasing Power of the city's currency?",
        0.0, 1.0, 0.5, help="(0 - Worst, 1 - Best) Purchasing power \
                of a currency implies how much one can buy with one unit of \
                that currency, e.g. The purchasing power of USD and EUR is \
                almost same, but Japanese YEN is weaker than these two, \
                meaning once can purchase less from a unit of japanese \
                YEN than they can with EURO or USD. A high score mean high \
                purchasing power, while a low score means low purchasing \
                power.")
safety_w = st.sidebar.slider("Standards of Safety?", 
        0.0, 1.0, 0.5, help="(0 - Worst, 1 - Best) A low score means low \
                standards of safety, and a high score means a high standard \
                of safety.")
pollution_w = st.sidebar.slider("Pollution?", 
        0.0, 1.0, 0.5, help="(0 - Worst, 1 - Best) A low score means worse \
                pollution, while a high score means better pollution control.")
traffic_w = st.sidebar.slider("Traffic wait times?", 
        0.0, 1.0, 0.5, help="(0 - Worst, 1 - Best) A low score mean worst \
                traffic wait times, while a higher score indicates better \
                traffi conditions.")

wht_dict = {"wt_mean_temp":avg_temp_w,"wt_std_dev":std_dev_temp_w,"wt_cost_living":cost_living_w,
        "wt_purchase_pow":purchase_pow_w,"wt_safety":safety_w,"wt_pollution":pollution_w,
        "wt_traffic":traffic_w}


# ---------------------------------------------------------------------------------------
# Get Data

# all_data_df : dataframe with all unscaled data
all_data_best_city_csv = Path(__file__).parents[1] / 'streamlit/all-data-best-city.csv'

all_data_df = pd.read_csv(all_data_best_city_csv)
if "Unnamed: 0" in list(all_data_df.columns):
    all_data_df = all_data_df.drop(['Unnamed: 0'], axis=1)
#print(list(all_data_df.columns))
#print(all_data_df.head())

# train_df_pca : Dataframe with rows converted to three pprincipal components
# for all 124 data points
train_df_pca_csv = Path(__file__).parents[1] / 'streamlit/train_df_pca.csv'

train_df_pca = pd.read_csv(train_df_pca_csv)
if "Unnamed: 0" in list(all_data_df.columns):
    all_data_df = all_data_df.drop(['Unnamed: 0'], axis=1)


# ---------------------------------------------------------------------------------------
# Functions?
def rank_eval(all_data_df, train_df_pca, wht_dict):
    
    mean_tmp_range = all_data_df['mean_tmp'].max() - all_data_df['mean_tmp'].min()
    std_dev_temp_range = all_data_df['std_dev_temp'].max() - all_data_df['std_dev_temp'].min()
    
    scaled_best_mean_tmp = (wht_dict['wt_mean_temp'] - all_data_df['mean_tmp'].min()) / mean_tmp_range
    scaled_best_std_dev_tmp = (wht_dict['wt_std_dev'] - all_data_df['std_dev_temp'].min()) / std_dev_temp_range

    new_point = np.array([[wht_dict['wt_cost_living'],wht_dict['wt_purchase_pow'],
        wht_dict['wt_safety'],wht_dict['wt_pollution'],wht_dict['wt_traffic'],
        scaled_best_mean_tmp,scaled_best_std_dev_tmp]])

    city_ascii = train_df_pca['city_ascii'].to_list()

    train_numpy_pca = train_df_pca.drop(['city_ascii'], axis=1).values

    # Load the trained PCA model
    pca_loaded_joblib = Path(__file__).parents[1] / 'streamlit/pca_model.joblib'
    pca_loaded = load(pca_loaded_joblib)

    #print(pca_loaded)

    # Convert new_point to a 1-D array
    new_point_loc = new_point.flatten()

    #print(new_point,new_point_loc)

    feature_names = ['scaled_cost_live_rent_index', 'scaled_purchase_pow_index',
            'scaled_safety_index', 'scaled_pollution_index','scaled_trffic_min_index', 
            'scaled_mean_tmp', 'scaled_tmp_std_dev']

    # Convert new point to dataframe with same column names as the dataframe on
    # which PCA was trained, these are obtained from the Jupyter notebook under
    # modelling section.
    new_point_df = pd.DataFrame(new_point_loc.reshape(1, -1), columns=feature_names)

    # Transform the new_point using the same PCA object
    new_point_pca = pca_loaded.transform(new_point_df)

    # Initialize an empty list to store the distances
    distances = []
    # Calculate Euclidean distance for each point in the dataframe
    for point in train_numpy_pca:
        distance = np.sqrt(np.sum((point - new_point_pca) ** 2))
        distances.append(distance)

    # Create a new dataframe with distances
    df_distances = pd.DataFrame(distances, columns=['Distance'], index=city_ascii)

    df_distances['city_ascii'] = df_distances.index

    #print(df_distances.head())

    finding_home_df = all_data_df[['city_ascii','country','lat','lng','mean_tmp','std_dev_temp']].copy()

    finding_home_df = finding_home_df.merge(df_distances, left_on='city_ascii',
            right_on='city_ascii', how='inner')

    finding_home_df = finding_home_df.sort_values(by=['Distance'],
            ascending=True)
    
    rnk = [i for i in range(1,finding_home_df.shape[0] + 1)]
    
    finding_home_df['Rank'] = rnk
    
    return finding_home_df, new_point_pca

finding_home_df, new_point_pca = rank_eval(all_data_df, train_df_pca, wht_dict)
#print(finding_home_df.columns)
#print(finding_home_df.head())

finding_home_top_df = finding_home_df.head(10)


# ---------------------------------------------------------------------------------------
# Plotting
#fig = px.scatter_geo(finding_home_top5_df, lat='lat', lon='lng',color='city_ascii', title='Best Cities For You!')

# Making the plot fancy
#fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")

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
    showrivers=True, rivercolor="LightBlue",
    bgcolor = 'black'
)

fig.update_layout(
    title_text = 'Top 10 Cities For the Parameters Set By You.',
    geo = dict(
        scope='world',
        #projection_type='equirectangular',
        projection_type='natural earth',
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

## PCA components 3D plot showing new city designed by user with real cities.

# Create a trace for the original PCA transformed points
trace1 = go.Scatter3d(
    x=train_df_pca.iloc[:, 0],
    y=train_df_pca.iloc[:, 1],
    z=train_df_pca.iloc[:, 2],
    mode='markers',
    marker=dict(
        size=6,
        color='white',                # set color to white
        opacity=0.5                   # set opacity to make points faint
    ),
    text=train_df_pca['city_ascii'],  # add labels
    hoverinfo='text',
    name='Dataset Cities'
)

# Create a trace for the new point
trace2 = go.Scatter3d(
    x=[new_point_pca[0][0]],
    y=[new_point_pca[0][1]],
    z=[new_point_pca[0][2]],
    mode='markers',
    marker=dict(
        size=10,
        color='red',                 # set color to red
    ),
    name='Ideal City'
)

# Define the layout
layout = go.Layout(
    width = 950,
    height = 600,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3',
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    )
)



# Create the figure and add traces
fig = go.Figure(data=[trace1, trace2], layout=layout)

# Show the plot
st.write("\n")
st.markdown("### Your Best City plotted against Real cities in Dataset,\
        Interact with the graph and find the closest city visually!")
st.plotly_chart(fig)

# ---------------------------------------------------------------------------------------
# Top 10 Table
top_20_df_to_show = finding_home_top_df.copy()
top_20_df_to_show['mean_tmp'] = top_20_df_to_show['mean_tmp'].round(1)
top_20_df_to_show['std_dev_temp'] = top_20_df_to_show['std_dev_temp'].round(1)
top_20_df_to_show = top_20_df_to_show[['Rank','city_ascii','mean_tmp','std_dev_temp']]
st.write("\n")
st.markdown("### Tabular List With Your Best Cities!")
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
