# Finding Home: Exploring the Best Cities to Live In    
&nbsp;
## Project Overview
Which city is the best to live in? This project delves into the exploration of various features about different cities and the impact on their liveability. Some of those features are the average temperature, cost of living and average commuting time. Based on the most desirable values regarding those features, we define the likelihood of a city being ”good” to live in. Based on a unsupervised learning model, we classify the cities in order of their overall liveability based on user preferences regarding the beforementioned features. This is done through customizable parameters, enabling the user to interact with the model and therefore, creating personalized rankings.   
&nbsp;
## Repository Structure

**eda**: Exploratory Data Analysis files.

**dat**: Contains the combined dataset and the code used for the merging process. We collected the data from various openly available sources. Those datasets were merged and subsequently standardized, obtaining a final sample of 124 cities with 7 features. The final dataset is stored in a CSV. The file named "all-data-best-city.csv".

**exp**: Contains Jupyter notebooks detailing the analysis we have done to construct our model. 

**viz**: Stored visualizations for our Report.   
- streamlit: Here we have stored all the files necessary to run our interactive streamlit plots. The link to the Streamlit Web-Hosted app can be found here: https://finding-home-app-ezqbrpvhtiqzyrv3drqedq.streamlit.app/
- The app can be run locally using the following command, given all required libraries and python version 3.10 are present.
```
streamlit run viz/streamlit/streamlit/interactive_city_recommender.py
```
&nbsp;
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
&nbsp;
## Team Members 
Ashutosh Jha (ashutosh.jha@student.uni-tuebingen.de)    
Jenny Lang (jenny.lang@student.uni-tuebingen.de)  
Joseph L. Wan-Wang (joseph.wan-wang@student.uni-tuebingen.de)   
Kayoon Kim (ka.kim@student.uni-tuebingen.de)   
&nbsp;
## Purpose of the Project
This project was done in the context of the "Data Literacy WS23/24" lecture in the University of Tübingen

