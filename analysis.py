##################
## Data sources ##
##################

## Covid19 Data ##
#1 - https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series

## Alternate data source (not used here) ##
#1'- https://www.worldometers.info/covid19virus/

## US County-level population estimate for 2019 ##
#2 - https://www.census.gov/data/tables/time-series/demo/popest/2010s-counties-total.html

## US County area data ##
#3 - http://data.sagepub.com/sagestats/document.php?id=7604

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

##############################
## Data import and cleaning ##
##############################

#### import data ########
data_covid19 = pd.read_csv("data/covid19_data_US_2020_04_02.csv") #1
data_county_pop = pd.read_csv("data/county_population_data_US_2010_2019.csv") #2
data_county_area = pd.read_csv("data/county_area_data_US.csv") #3

#### Create dictionary of state abbreviations ########
state_abbrev = pd.read_csv("data/US_state_abbrev.csv")
abbrev_to_state = state_abbrev.set_index('Abbrev').T.to_dict('list')
for key in abbrev_to_state:
    abbrev_to_state[key] = abbrev_to_state[key][0]

#### replace state abbreviations by names to be consistent with other data ####
data_county_area['State']=data_county_area['ST'].replace(abbrev_to_state,regex=True)

#### Remove state info from County field ####
data_county_area['County']=data_county_area['County'].str.split(',').str[0]

#### Make county and column names consistent with other datasets ####
data_covid19['County'] = data_covid19['Admin2'] + ' County'
data_covid19 = data_covid19.rename(columns={'Province_State': 'State'})

#### Remove whitespace ######
data_county_pop['State'] = data_county_pop['State'].str.lstrip()

#################################
## Population Density analysis ##
#################################

#### Merge Population and Area datasets, and calculate population densities in number of individuals per sq. km ####
data_county_pop_area=pd.merge(data_county_pop,data_county_area,on=['County','State'])
data_county_pop_area['area_sq_km'] = data_county_pop_area['SQUARE MILES']*2.58999
data_county_pop_area['Pop_Density']=data_county_pop_area['2019']/data_county_pop_area['area_sq_km']

#### Merge Covid19 and population density datasets ####
data = pd.merge(data_county_pop_area,data_covid19,on=['County','State'])
data['County_State'] = data['County']+str(' ')+data['State']
col = data.pop('County_State')
data.insert(0, col.name, col)

#### Time series length desired (currently set to accept last ten days) ####
time_series_length=10
qualifying_County_State_list=list(data[np.all(data.iloc[:,-time_series_length:]!=0,axis=1)].County_State)

#### Linear weight regression on log(number of cases) vs time (days), weighted by sqrt(number of cases) ####
t_axis=np.arange(time_series_length).reshape(-1,1)
regression = {}
for i in qualifying_County_State_list:
    time_series_data=np.asarray(data[data['County_State']==i].iloc[:,-time_series_length:]).flatten()
    regression[i] = linear_model.LinearRegression();
    regression[i].fit(t_axis,np.log(time_series_data).reshape(-1,1),np.sqrt(time_series_data))

#### Make a list of coefficients
list_of_population_and_regression_coefficients = []
for i in regression.keys():
    list_entry=np.append(np.asarray(list(data[data['County_State']==i].Pop_Density)),regression[i].coef_.flatten())
    list_of_population_and_regression_coefficients.append(list_entry)

list_of_population_and_regression_coefficients=np.asarray(list_of_population_and_regression_coefficients)
#### Number of cases growth rate vs. Population density ######
plt.scatter(np.log(list_of_population_and_regression_coefficients[:,0]),list_of_population_and_regression_coefficients[:,1])
plt.show()


