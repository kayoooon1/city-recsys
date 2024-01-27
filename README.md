# Finding Home: Exploring the Best Cities to Live In    
## Project Overview
Which city is the best to live in? This project delves into the exploration of various features about different cities and the impact on their liveability. Some of those features are the average temperature, cost of living and average commuting time. Based on the most desirable values regarding those features, we define the likelihood of a city being ”good” to live in. Based on a unsupervised learning model, we classify the cities in order of their overall liveability based on user preferences regarding the beforementioned features. This is done through customizable parameters, enabling the user to interact with the model and therefore, creating personalized rankings.

## Repository Structure

eda: Exploratory Data Analysis files. Here the descriptive statistics and other preliminary tests are stored.

dat: Contains the combined dataset and the code used for the merging process. We collected the data from various openly available sources. Those datasets were merged and subsequently standardized, obtaining a final sample of 154 cities with 7 features. The data is stored in a CSV. file.

exp: Contains Jupyter notebooks detailing the analysis we have tested to construct our model and the visualizations of our data. The candidate models we have tested are the following: Bayesian propability model, K-means, 

viz: Stored visualizations for our reports
  streamlit: Here we have stored all the files necessary to run our interactive stramlit plots. The link to the Streamlit app can be found here: https://finding-     home-app-ezqbrpvhtiqzyrv3drqedq.streamlit.app/

## Requirements
Python Version: 3.10

Required Libraries:

<pre>
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Dotenv
Tueplot
Stream
Plotly
Pathlib
Joblib
Cartopy
MPL_Toolkits
</pre>

## Team Members 
Ashutosh Jha (ashutosh.jha@student.uni-tuebingen.de)
Jenny Lang (jenny.lang@student.uni-tuebingen.de)  
Joseph L. Wan-Wang (joseph.wan-wang@student.uni-tuebingen.de)   
Kayoon Kim (ka.kim@student.uni-tuebingen.de)   

## Purpose of the Project
This project was done in the context of the "Data Literacy WS23/24" lecture in the University of Tübingen

