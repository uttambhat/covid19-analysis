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

## US County average temperature data ##
#4 - ftp://ftp.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpccy-v1.0.0-20200304

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

##############################
## Data import and cleaning ##
##############################

#### import data ########
data_covid19 = pd.read_csv("data/covid19_data_US_2020_04_03.csv", dtype={'FIPS':object}) #1
data_county_pop = pd.read_csv("data/county_population_data_US_2010_2019.csv") #2
data_county_area = pd.read_csv("data/county_area_data_US.csv") #3
data_temperature = pd.read_csv('data/climdiv-tmpccy-v1.0.0-20200304', delim_whitespace=True)

#### Make State, County and Year columns in data_temperature #####
data_temperature['State_code']=data_temperature['State_County_Code_Year'].astype(str).str.slice(start=0,stop=2)
data_temperature['County_code']=data_temperature['State_County_Code_Year'].astype(str).str.slice(start=2,stop=5)
data_temperature['Climate_code']=data_temperature['State_County_Code_Year'].astype(str).str.slice(start=5,stop=7)
data_temperature['Year']=data_temperature['State_County_Code_Year'].astype(str).str.slice(start=7,stop=11)

#### Create dictionary of state abbreviations (for data_county_area) and state codes (for data_temperature) and county FIPS codes (for data_temperature) ########
state_abbrev = pd.read_csv("data/US_state_abbrev.csv")
abbrev_to_state = state_abbrev.set_index('Abbrev').T.to_dict('list')
for key in abbrev_to_state:
    abbrev_to_state[key] = abbrev_to_state[key][0]

state_code = pd.read_csv("data/climate_state_codes.csv", dtype=object)
code_to_state = state_code.set_index('Code').T.to_dict('list')
for key in code_to_state:
    code_to_state[key] = code_to_state[key][0]

state_fips_code = pd.read_csv("data/state_fips_codes.csv", dtype=object)
state_to_fips_code = state_fips_code.set_index('State').T.to_dict('list')
for key in state_to_fips_code:
    state_to_fips_code[key] = state_to_fips_code[key][0]


data_covid19['FIPS'] = data_covid19['FIPS'].str.zfill(5)

#### replace state abbreviations and codes by names to be consistent with other data ####
data_county_area['State']=data_county_area['ST'].replace(abbrev_to_state)
data_temperature['State']=data_temperature['State_code'].replace(code_to_state)
data_temperature['State_FIPS_code']=data_temperature['State'].replace(state_to_fips_code)
data_temperature['FIPS']=data_temperature['State_FIPS_code']+data_temperature['County_code']

#### Remove state info from County field ####
data_county_area['County']=data_county_area['County'].str.split(',').str[0]

#### Make county and column names consistent with other datasets ####
data_covid19['County'] = data_covid19['Admin2'] + ' County'
data_covid19 = data_covid19.rename(columns={'Province_State': 'State'})

#### Remove whitespace ######
data_county_pop['State'] = data_county_pop['State'].str.lstrip()

########################################
## Extract relevant temperature data ###
########################################

temp_list = ['2015','2016','2017','2018','2019']
data_temperature_last_n_years = data_temperature[data_temperature.Year.isin(temp_list)].loc[:,['FIPS','Mar']]
data_relevant_temperature = data_temperature[data_temperature['Year']=='2019'].loc[:,['FIPS','Mar']]
data_relevant_temperature = data_temperature_last_n_years.groupby('FIPS').mean().rename(columns = {'Mar':'Temperature'})

#################################
## Population Density analysis ##
#################################

#### Merge Population and Area datasets, and calculate population densities in number of individuals per sq. km ####
data_county_pop_area=pd.merge(data_county_pop,data_county_area,on=['County','State'])
data_county_pop_area['area_sq_km'] = data_county_pop_area['SQUARE MILES']*2.58999
data_county_pop_area['Pop_Density']=data_county_pop_area['2019']/data_county_pop_area['area_sq_km']

#### Merge Covid19 and population density datasets ####
data = pd.merge(data_county_pop_area,data_covid19,on=['County','State'])
data = pd.merge(data_relevant_temperature,data,on=['FIPS'])
data['County_State'] = data['County']+str(' ')+data['State']
col = data.pop('County_State')
data.insert(0, col.name, col)

#### Time series length desired (currently set to accept last ten days) ####
time_series_length=10
time_series_offset=10
qualifying_County_State_list=list(data[np.all(data.iloc[:,-time_series_length-time_series_offset:-time_series_offset]>0,axis=1)].County_State)

#### Linear weight regression on log(number of cases) vs time (days), weighted by sqrt(number of cases) ####
t_axis=np.arange(time_series_length).reshape(-1,1)
regression = {}
regression_R2 = {}
for i in qualifying_County_State_list:
    time_series_data=np.asarray(data[data['County_State']==i].iloc[:,-time_series_length-time_series_offset:-time_series_offset]).flatten()
    regression[i] = linear_model.LinearRegression();
    regression[i].fit(t_axis,np.log(time_series_data).reshape(-1,1),np.sqrt(time_series_data))
    regression_R2[i] = metrics.r2_score(np.log(time_series_data),regression[i].predict(t_axis).flatten(),np.sqrt(time_series_data))

#### Make a list of coefficients
list_of_population_reg_coefs_r2_score = []
for i in regression.keys():
    list_entry=np.append(np.append(np.asarray(list(data[data['County_State']==i].Pop_Density)),regression[i].coef_.flatten()),regression_R2[i])
    list_of_population_reg_coefs_r2_score.append(list_entry)

list_of_temperature_reg_coefs_r2_score = []
for i in regression.keys():
    list_entry=np.append(np.append(np.asarray(list(data[data['County_State']==i].Temperature)),regression[i].coef_.flatten()),regression_R2[i])
    list_of_temperature_reg_coefs_r2_score.append(list_entry)

list_of_population_reg_coefs_r2_score=np.asarray(list_of_population_reg_coefs_r2_score)
list_of_temperature_reg_coefs_r2_score=np.asarray(list_of_temperature_reg_coefs_r2_score)

#### Plot histogram of R2 values #####
plt.hist(list_of_population_reg_coefs_r2_score[:,2],20)
plt.show()

#### Number of cases growth rate vs. Population density ######
plt.scatter(np.log(list_of_population_reg_coefs_r2_score[:,0]),list_of_population_reg_coefs_r2_score[:,1],(list_of_population_reg_coefs_r2_score[:,2]-0.8)*200)
plt.show()

plt.scatter(list_of_temperature_reg_coefs_r2_score[:,0],list_of_temperature_reg_coefs_r2_score[:,1],(list_of_population_reg_coefs_r2_score[:,2]-0.8)*200)
plt.show()


