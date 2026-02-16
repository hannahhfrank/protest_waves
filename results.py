import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import cm 
from shapely.geometry import Point
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
from sklearn import preprocessing
import json
from scipy.stats import ttest_1samp
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as patches
import random
from dtaidistance import dtw
from scipy.cluster.hierarchy import linkage, fcluster
from tslearn.clustering import silhouette_score
from scipy.spatial.distance import squareform
import matplotlib.patches as mpatches
import os 
from tslearn.barycenters import dtw_barycenter_averaging
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'
random.seed(1)
np.random.seed(1)
plot_params = {"text.usetex":True,"font.family":"serif","font.size":15,"xtick.labelsize":15,"ytick.labelsize":15,"axes.labelsize":15,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
plt.rcParams.update(plot_params)

#############
### India ###
#############

### Indian farmers’ protest—August 2020 until December 2021 ###

# Load and subset
acled = pd.read_csv("data/acled/acled_all_events.csv")
acled=acled.loc[(acled["event_type"]=="Protests")&(acled["country"]=="India")]

# Select most frequent actor
most_frequent = acled['assoc_actor_1'].value_counts()
most_frequent[:10]         
acled['assoc_actor_1'] = acled['assoc_actor_1'].astype(str)
acled_farmers = acled[acled['assoc_actor_1'].str.contains("Farmers",na=False)]

# Obtain dd variable
acled_farmers["dd"] = pd.to_datetime(acled_farmers['event_date'],format='%d %B %Y')
acled_farmers["dd"] = acled_farmers["dd"].dt.strftime('%Y-%m')

# Add grid ids 
acled_farmers["gid"] = 0
acled_farmers.reset_index(inplace=True, drop=True)
def add_value_if_no_or_one_decimal(x, value_to_add):
    if x % 1 == 0 or round(x, 1) == x:  
        return x + value_to_add
    else:
        return x
value_to_add = 0.0001  
acled_farmers['longitude'] = acled_farmers['longitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)
acled_farmers['latitude'] = acled_farmers['latitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)
prio_shp = gpd.read_file('data/acled/priogrid_cell.shp')
prio_shp.reset_index(inplace=True, drop=True)
for i in range(len(acled_farmers)):
    prio_shp_s = prio_shp[(prio_shp["ycoord"]>acled_farmers.latitude.iloc[i]-1) &
                              (prio_shp["ycoord"]<acled_farmers.latitude.iloc[i]+1) &
                              (prio_shp["xcoord"]>acled_farmers.longitude.iloc[i]-1) &
                              (prio_shp["xcoord"]<acled_farmers.longitude.iloc[i]+1)]
    prio_shp_s.reset_index(inplace=True, drop=True)

    for x in range(len(prio_shp_s)):
        if prio_shp_s.geometry.iloc[x].contains(Point(acled_farmers.longitude.iloc[i], acled_farmers.latitude.iloc[i])) == True:
            acled_farmers.gid.iloc[i] = prio_shp_s.gid.iloc[x]
            break   
        
# Aggregate to gid-month        
acled_farmers = pd.DataFrame(acled_farmers.groupby(["dd","gid"]).size())
acled_farmers = acled_farmers.reset_index()
acled_farmers.rename(columns={0:"n_protest_events"}, inplace=True)    
  
# Add missing observation to time series, those have zero events
prio_help = gpd.read_file("data/acled/prio_time_2014.csv")
prio_help_s=prio_help[["gid"]].loc[(prio_help["gwno"]=="750")].reset_index(drop=True)
prio_help_s["gid"]=prio_help_s["gid"].astype(int)
gid = list(prio_help_s.gid.unique())
date = ['2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
        '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12',
        '2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12',
        '2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12',
        '2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
        '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
        '2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12',
        '2023-01','2023-02','2023-03','2023-04','2023-05','2023-06','2023-07','2023-08','2023-09','2023-10','2023-11','2023-12']
for i in range(0, len(gid)):
    for x in range(0, len(date)):        
        if ((acled_farmers['dd']==date[x]) 
            & (acled_farmers['gid']==gid[i])).any()==False:
                s = {'dd':date[x],'gid':gid[i],'n_protest_events':0}
                s = pd.DataFrame(data=s,index=[0])
                acled_farmers = pd.concat([acled_farmers,s])  
acled_farmers = acled_farmers[acled_farmers["gid"].isin(gid)]
acled_farmers = acled_farmers.sort_values(by=["gid","dd"])
acled_farmers=acled_farmers.reset_index(drop=True)
  
# Find gids with high intensity
df_n_country_month = {}
for i in acled_farmers["gid"].unique():
    ts = acled_farmers["n_protest_events"].loc[acled_farmers["gid"]==i]
    df_n_country_month[i] = ts.max()
df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True)     
df_n_country_month=df_n_country_month.sort_values(by=['avg'])

# Min-max normalize
def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        df_s = df.loc[df[group] == i]
        scaler = preprocessing.MinMaxScaler().fit(df_s[x].values.reshape(-1, 1))
        norm = scaler.transform(df_s[x].values.reshape(-1, 1))
        out = pd.concat([out, pd.DataFrame(norm)], ignore_index=True)
    df[f"{x}_norm"] = out  
preprocess_min_max_group(acled_farmers,"n_protest_events","gid")

# Plot high intensity gids
fig, ax = plt.subplots(figsize = (12,8))
for i,l in zip([171155,173312,171875,172591],["solid","dashed","dotted","dashdot"]):
    plt.plot(acled_farmers["dd"].loc[(acled_farmers["gid"]==i)& (acled_farmers["dd"]>="2019-10") & (acled_farmers["dd"]<="2022-02")],acled_farmers["n_protest_events_norm"].loc[(acled_farmers["gid"]==i)& (acled_farmers["dd"]>="2019-10") & (acled_farmers["dd"]<="2022-02")],label=i,linestyle=l,color="black",linewidth=2)
plt.legend(fontsize=20)
plt.xticks(["2019-10","2020-01","2020-04","2020-07","2020-10","2021-01","2021-04","2021-07","2021-10","2022-01"],["10-19","01-20","04-20","07-20","10-20","01-21","04-21","07-21","10-21","01-22"],size=26)
plt.yticks([0,0.2,0.4,0.6,0.8,1],size=25)
plt.savefig("results/acled_india_farmers_protest.eps",dpi=400,bbox_inches="tight")
plt.show()

################
### Pakistan ###
################

### Pakistan Tehreek-e-Insaf (PTI) protest—March until May 2023 ###

# Load and subset
acled = pd.read_csv("data/acled/acled_all_events.csv")
acled=acled.loc[(acled["event_type"]=="Protests")&(acled["country"]=="Pakistan")]

# Select most frequent actor
most_frequent = acled['assoc_actor_1'].value_counts()
most_frequent[:10]         
acled['assoc_actor_1'] = acled['assoc_actor_1'].astype(str)
acled_farmers = acled.loc[acled['assoc_actor_1']=="PTI: Pakistan Tehreek-i-Insaf"]

# Obtain dd variable
acled_farmers["dd"] = pd.to_datetime(acled_farmers['event_date'],format='%d %B %Y')
acled_farmers["dd"] = acled_farmers["dd"].dt.strftime('%Y-%m')

# Add grid ids 
acled_farmers["gid"] = 0
acled_farmers.reset_index(inplace=True, drop=True)
def add_value_if_no_or_one_decimal(x, value_to_add):
    if x % 1 == 0 or round(x, 1) == x:  
        return x + value_to_add
    else:
        return x
value_to_add = 0.0001  
acled_farmers['longitude'] = acled_farmers['longitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)
acled_farmers['latitude'] = acled_farmers['latitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)
prio_shp = gpd.read_file('data/acled/priogrid_cell.shp')
prio_shp.reset_index(inplace=True, drop=True)
for i in range(len(acled_farmers)):
    prio_shp_s = prio_shp[(prio_shp["ycoord"]>acled_farmers.latitude.iloc[i]-1) &
                              (prio_shp["ycoord"]<acled_farmers.latitude.iloc[i]+1) &
                              (prio_shp["xcoord"]>acled_farmers.longitude.iloc[i]-1) &
                              (prio_shp["xcoord"]<acled_farmers.longitude.iloc[i]+1)]
    prio_shp_s.reset_index(inplace=True, drop=True)
    
    for x in range(len(prio_shp_s)):
        if prio_shp_s.geometry.iloc[x].contains(Point(acled_farmers.longitude.iloc[i], acled_farmers.latitude.iloc[i])) == True:
            acled_farmers.gid.iloc[i] = prio_shp_s.gid.iloc[x]
            break        
        
# Aggregate to gid-month               
acled_farmers = pd.DataFrame(acled_farmers.groupby(["dd","gid"]).size())
acled_farmers = acled_farmers.reset_index()
acled_farmers.rename(columns={0:"n_protest_events"}, inplace=True)    

# Add missing observation to time series, those have zero events 
prio_help = gpd.read_file("data/acled/prio_time_2014.csv")
prio_help_s=prio_help[["gid"]].loc[(prio_help["gwno"]=="770")].reset_index(drop=True)
prio_help_s["gid"]=prio_help_s["gid"].astype(int)
gid = list(prio_help_s.gid.unique())
date = ['2010-01','2010-02','2010-03','2010-04','2010-05','2010-06','2010-07','2010-08','2010-09','2010-10','2010-11','2010-12',
        '2011-01','2011-02','2011-03','2011-04','2011-05','2011-06','2011-07','2011-08','2011-09','2011-10','2011-11','2011-12',
        '2012-01','2012-02','2012-03','2012-04','2012-05','2012-06','2012-07','2012-08','2012-09','2012-10','2012-11','2012-12',
        '2013-01','2013-02','2013-03','2013-04','2013-05','2013-06','2013-07','2013-08','2013-09','2013-10','2013-11','2013-12',
        '2014-01','2014-02','2014-03','2014-04','2014-05','2014-06','2014-07','2014-08','2014-09','2014-10','2014-11','2014-12',
        '2015-01','2015-02','2015-03','2015-04','2015-05','2015-06','2015-07','2015-08','2015-09','2015-10','2015-11','2015-12',
        '2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
        '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12',
        '2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12',
        '2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12',
        '2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
        '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
        '2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12',
        '2023-01','2023-02','2023-03','2023-04','2023-05','2023-06','2023-07','2023-08','2023-09','2023-10','2023-11','2023-12']
for i in range(0, len(gid)):
    for x in range(0, len(date)):        
        if ((acled_farmers['dd']==date[x]) 
            & (acled_farmers['gid']==gid[i])).any()==False:
                s = {'dd':date[x],'gid':gid[i],'n_protest_events':0}
                s = pd.DataFrame(data=s,index=[0])
                acled_farmers = pd.concat([acled_farmers,s])  
acled_farmers = acled_farmers[acled_farmers["gid"].isin(gid)]
acled_farmers = acled_farmers.sort_values(by=["gid","dd"])
acled_farmers=acled_farmers.reset_index(drop=True)
  
# Find countries with high intensity
df_n_country_month = {}
for i in acled_farmers["gid"].unique():
    ts = acled_farmers["n_protest_events"].loc[acled_farmers["gid"]==i]
    df_n_country_month[i] = ts.max()
df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True)     
df_n_country_month=df_n_country_month.sort_values(by=['avg'])

# Min-max normalize
def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        df_s = df.loc[df[group] == i]
        scaler = preprocessing.MinMaxScaler().fit(
            df_s[x].values.reshape(-1, 1))
        norm = scaler.transform(df_s[x].values.reshape(-1, 1))
        out = pd.concat([out, pd.DataFrame(norm)], ignore_index=True)
    df[f"{x}_norm"] = out    
preprocess_min_max_group(acled_farmers,"n_protest_events","gid")

# Plot high intensity gids
fig, ax = plt.subplots(figsize = (12,8))
for i,l in zip([179786,165375,179784,178347],["solid","dashed","dotted","dashdot"]):
    plt.plot(acled_farmers["dd"].loc[(acled_farmers["gid"]==i)& (acled_farmers["dd"]>="2022-05") & (acled_farmers["dd"]<="2023-12")],acled_farmers["n_protest_events_norm"].loc[(acled_farmers["gid"]==i)& (acled_farmers["dd"]>="2022-05") & (acled_farmers["dd"]<="2023-12")],label=i,linestyle=l,color="black",linewidth=2)
plt.legend(fontsize=20)
plt.xticks(["2022-05","2022-07","2022-09","2022-11","2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["05-22","07-22","09-22","11-22","01-23","03-23","05-23","07-23","09-23","11-23"],size=26)
plt.yticks([0,0.2,0.4,0.6,0.8,1],size=25)
plt.savefig("results/acled_pakistan_protest.eps",dpi=400,bbox_inches="tight")
plt.show()

#####################
### United States ###
#####################

### United States George Floyd protests—May 2020 until May 2021 ###

# Load and subset
acled = pd.read_csv("data/acled/acled_all_events.csv")
acled=acled.loc[(acled["event_type"]=="Protests")&(acled["country"]=="United States")]

# Select most frequent actor
most_frequent = acled['assoc_actor_1'].value_counts()
most_frequent[:10]
acled['assoc_actor_1'] = acled['assoc_actor_1'].astype(str)
acled_farmers = acled.loc[acled['assoc_actor_1']=="BLM: Black Lives Matter"]

# Obtain dd variable
acled_farmers["dd"] = pd.to_datetime(acled_farmers['event_date'],format='%d %B %Y')
acled_farmers["dd"] = acled_farmers["dd"].dt.strftime('%Y-%m')

# Add grid ids 
acled_farmers["gid"] = 0
acled_farmers.reset_index(inplace=True, drop=True)
def add_value_if_no_or_one_decimal(x, value_to_add):
    if x % 1 == 0 or round(x, 1) == x:  
        return x + value_to_add
    else:
        return x
value_to_add = 0.0001  
acled_farmers['longitude'] = acled_farmers['longitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)
acled_farmers['latitude'] = acled_farmers['latitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)
prio_shp = gpd.read_file('data/acled/priogrid_cell.shp')
prio_shp.reset_index(inplace=True, drop=True)
for i in range(len(acled_farmers)):
    prio_shp_s = prio_shp[(prio_shp["ycoord"]>acled_farmers.latitude.iloc[i]-1) &
                              (prio_shp["ycoord"]<acled_farmers.latitude.iloc[i]+1) &
                              (prio_shp["xcoord"]>acled_farmers.longitude.iloc[i]-1) &
                              (prio_shp["xcoord"]<acled_farmers.longitude.iloc[i]+1)]
    prio_shp_s.reset_index(inplace=True, drop=True)

    for x in range(len(prio_shp_s)):
        if prio_shp_s.geometry.iloc[x].contains(Point(acled_farmers.longitude.iloc[i], acled_farmers.latitude.iloc[i])) == True:
            acled_farmers.gid.iloc[i] = prio_shp_s.gid.iloc[x]
            break       
        
# Aggregate to gid-month                    
acled_farmers = pd.DataFrame(acled_farmers.groupby(["dd","gid"]).size())
acled_farmers = acled_farmers.reset_index()
acled_farmers.rename(columns={0:"n_protest_events"}, inplace=True)      

# Add missing observation to time series, those have zero events 
prio_help = gpd.read_file("data/acled/prio_time_2014.csv")
prio_help_s=prio_help[["gid"]].loc[(prio_help["gwno"]=="2")].reset_index(drop=True)
prio_help_s["gid"]=prio_help_s["gid"].astype(int)
gid = list(prio_help_s.gid.unique())
date = ['2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
        '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
        '2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12',
        '2023-01','2023-02','2023-03','2023-04','2023-05','2023-06','2023-07','2023-08','2023-09','2023-10','2023-11','2023-12']
for i in range(0, len(gid)):
    for x in range(0, len(date)):        
        if ((acled_farmers['dd']==date[x]) 
            & (acled_farmers['gid']==gid[i])).any()==False:
                s = {'dd':date[x],'gid':gid[i],'n_protest_events':0}
                s = pd.DataFrame(data=s,index=[0])
                acled_farmers = pd.concat([acled_farmers,s])  
acled_farmers = acled_farmers[acled_farmers["gid"].isin(gid)]
acled_farmers = acled_farmers.sort_values(by=["gid","dd"])
acled_farmers=acled_farmers.reset_index(drop=True)
  
# Find countries with high intensity
df_n_country_month = {}
for i in acled_farmers["gid"].unique():
    ts = acled_farmers["n_protest_events"].loc[acled_farmers["gid"]==i]
    df_n_country_month[i] = ts.max()
df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True)     
df_n_country_month=df_n_country_month.sort_values(by=['avg'])

# Min-max normalize
def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        df_s = df.loc[df[group] == i]
        scaler = preprocessing.MinMaxScaler().fit(
            df_s[x].values.reshape(-1, 1))
        norm = scaler.transform(df_s[x].values.reshape(-1, 1))
        out = pd.concat([out, pd.DataFrame(norm)], ignore_index=True)
    df[f"{x}_norm"] = out  
preprocess_min_max_group(acled_farmers,"n_protest_events","gid")

# Plot high intensity gids
fig, ax = plt.subplots(figsize = (12,8))
for i,l in zip([188133,198116,188132,183716],["solid","dashed","dotted","dashdot"]):
    plt.plot(acled_farmers["dd"].loc[(acled_farmers["gid"]==i)& (acled_farmers["dd"]>="2020-01") & (acled_farmers["dd"]<="2021-07")],acled_farmers["n_protest_events_norm"].loc[(acled_farmers["gid"]==i)& (acled_farmers["dd"]>="2020-01") & (acled_farmers["dd"]<="2021-07")],label=i,linestyle=l,color="black",linewidth=2)
plt.legend(fontsize=20)
plt.xticks(["2020-01","2020-03","2020-05","2020-07","2020-09","2020-11","2021-01","2021-03","2021-05","2021-07"],["01-20","03-20","05-20","07-20","09-20","11-20","01-21","03-21","05-21","07-21"],size=26)
plt.yticks([0,0.2,0.4,0.6,0.8,1],size=25)
plt.savefig("results/acled_US_protest.eps",dpi=400,bbox_inches="tight")
plt.show()

############################
### Actuals protest maps ###
############################

# Function to min-max normalize
def preprocess_min_max_group(df, x, group):
    out = pd.DataFrame()
    for i in df[group].unique():
        df_s = df.loc[df[group] == i]
        scaler = preprocessing.MinMaxScaler().fit(
            df_s[x].values.reshape(-1, 1))
        norm = scaler.transform(df_s[x].values.reshape(-1, 1))
        out = pd.concat([out, pd.DataFrame(norm)], ignore_index=True)
    df[f"{x}_norm"] = out

# Load acled data
acled = pd.read_csv("data/acled/acled_grid_India_2023.csv")

# Find countries with high intensity
thres=0.5
df_n_country_month = {}
for i in acled["gid"].unique():
    ts = acled["n_protest_events"].loc[acled["gid"]==i][:int(0.7*len(acled["n_protest_events"].loc[acled["gid"]==i]))]
    df_n_country_month[i] = np.mean(np.abs(ts - np.mean(ts)))
df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True) 
country_keep = df_n_country_month.loc[df_n_country_month["avg"]>thres].gid.unique()

# Merge with shape file
prio_shp = gpd.read_file('data/acled/priogrid_cell.shp')
prio_shp.reset_index(inplace=True, drop=True)
india = acled.merge(prio_shp[["gid", "geometry"]],left_on='gid',right_on='gid',how="left")
india = gpd.GeoDataFrame(india, geometry="geometry")

# Min-max normalize and differentiate between high and low intensity
preprocess_min_max_group(india,"n_protest_events","gid")
countries=india.gid.unique()
countries_drop = [item for item in countries if item not in country_keep]
india_drop = india[india['gid'].isin(countries_drop)]
india_keep = india[india['gid'].isin(country_keep)]

# Plot
for i in ['2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
          '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12',
          '2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12',
          '2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12',
          '2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
          '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
          '2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12',
          '2023-01','2023-02','2023-03','2023-04','2023-05','2023-06','2023-07','2023-08','2023-09','2023-10','2023-11','2023-12']:
    fig, ax = plt.subplots(figsize = (10, 10))
    plt.axis('off')
    india_keep.loc[(india_keep["dd"]==i)].plot(column='n_protest_events_norm',ax=ax,cmap="Greys",edgecolor='gray',linewidth=0.1)
    india_drop.loc[(india_drop["dd"]==i)].plot(column='n_protest_events_norm',ax=ax,color="lightblue",edgecolor='gray',linewidth=0.1,alpha=0.1)
    cmap = plt.cm.get_cmap('Greys')
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.01)   
    colorbar=plt.colorbar(cm.ScalarMappable(cmap=cmap,norm=norm),ax=ax,cax=cax)
    colorbar.ax.tick_params(axis='y', labelsize=25)  
    if i=="2016-12" or i=="2018-12":
        plt.savefig(f"results/acled_grid_{i}.eps",dpi=400,bbox_inches="tight")
    plt.show()
      
####################
### Main results ###
####################

# Function to calculate standard errors for WMSE
def wmse_se(y_true, y_pred, weights):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    w = np.asarray(weights)

    z = (y_true - y_pred) ** 2
    w_sum = np.sum(w)
    z_bar_w = np.sum(w * z) / w_sum

    var_wmse = np.sum((w ** 2) * (z - z_bar_w) ** 2) / (w_sum ** 2)
    return np.sqrt(var_wmse)

# Linear model 
final_arima = pd.read_csv("data/predictions/df_linear.csv",index_col=0)

# (1) WMSE #
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_arima,sample_weight=final_arima.n_protest_events),5))
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_darima,sample_weight=final_arima.n_protest_events),5))
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_arimax,sample_weight=final_arima.n_protest_events),5))
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_darimax,sample_weight=final_arima.n_protest_events),5))

arima_wmse=[mean_squared_error(final_arima.n_protest_events, final_arima.preds_arima,sample_weight=final_arima.n_protest_events),
            mean_squared_error(final_arima.n_protest_events, final_arima.preds_darima,sample_weight=final_arima.n_protest_events),
            mean_squared_error(final_arima.n_protest_events, final_arima.preds_arimax,sample_weight=final_arima.n_protest_events),
            mean_squared_error(final_arima.n_protest_events, final_arima.preds_darimax,sample_weight=final_arima.n_protest_events)]

arima_wmse_std=[wmse_se(final_arima.n_protest_events, final_arima.preds_arima,final_arima.n_protest_events),
                wmse_se(final_arima.n_protest_events, final_arima.preds_darima,final_arima.n_protest_events),
            wmse_se(final_arima.n_protest_events, final_arima.preds_arimax,final_arima.n_protest_events),
            wmse_se(final_arima.n_protest_events, final_arima.preds_darimax,final_arima.n_protest_events)]

arima_wmse_std_r = [round(x, 5) for x in arima_wmse_std]
print(arima_wmse_std_r)

final_arima["mse_arima"]=final_arima["n_protest_events"]*(final_arima["n_protest_events"] - final_arima["preds_arima"]) ** 2  
final_arima["mse_darima"]=final_arima["n_protest_events"]*(final_arima["n_protest_events"] - final_arima["preds_darima"]) ** 2  
final_arima["mse_arimax"]=final_arima["n_protest_events"]*(final_arima["n_protest_events"] - final_arima["preds_arimax"]) ** 2  
final_arima["mse_darimax"]=final_arima["n_protest_events"]*(final_arima["n_protest_events"] - final_arima["preds_darimax"]) ** 2  

# t test for whether difference is not zero
print(round(ttest_1samp((final_arima["mse_arima"]-final_arima["mse_arimax"]), 0)[1],5))
print(round(ttest_1samp((final_arima["mse_arima"]-final_arima["mse_darima"]), 0)[1],5))
print(round(ttest_1samp((final_arima["mse_arimax"]-final_arima["mse_darimax"]), 0)[1],5))

# Nonlinear model 
final_rf =  pd.read_csv("data/predictions/df_nonlinear.csv",index_col=0)

print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_rf,sample_weight=final_rf.n_protest_events),5))
print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_drf,sample_weight=final_rf.n_protest_events),5))
print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_rfx,sample_weight=final_rf.n_protest_events),5))
print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_drfx,sample_weight=final_rf.n_protest_events),5))

rf_wmse=[mean_squared_error(final_rf.n_protest_events, final_rf.preds_rf,sample_weight=final_rf.n_protest_events),
            mean_squared_error(final_rf.n_protest_events, final_rf.preds_drf,sample_weight=final_rf.n_protest_events),
            mean_squared_error(final_rf.n_protest_events, final_rf.preds_rfx,sample_weight=final_rf.n_protest_events),
            mean_squared_error(final_rf.n_protest_events, final_rf.preds_drfx,sample_weight=final_rf.n_protest_events)]

rf_wmse_std=[wmse_se(final_rf.n_protest_events, final_rf.preds_rf,final_rf.n_protest_events),
             wmse_se(final_rf.n_protest_events, final_rf.preds_drf,final_rf.n_protest_events),
            wmse_se(final_rf.n_protest_events, final_rf.preds_rfx,final_rf.n_protest_events),
            wmse_se(final_rf.n_protest_events, final_rf.preds_drfx,final_rf.n_protest_events)]

rf_wmse_std_f = [round(x, 5) for x in rf_wmse_std]
print(rf_wmse_std_f)

final_rf["mse_rf"]=final_rf["n_protest_events"]*(final_rf["n_protest_events"] - final_rf["preds_rf"]) ** 2  
final_rf["mse_drf"]=final_rf["n_protest_events"]*(final_rf["n_protest_events"] - final_rf["preds_drf"]) ** 2  
final_rf["mse_rfx"]=final_rf["n_protest_events"]*(final_rf["n_protest_events"] - final_rf["preds_rfx"]) ** 2  
final_rf["mse_drfx"]=final_rf["n_protest_events"]*(final_rf["n_protest_events"] - final_rf["preds_drfx"]) ** 2  

# t test for whether difference is not zero
print(round(ttest_1samp((final_rf["mse_rf"]-final_rf["mse_rfx"]), 0)[1],5))
print(round(ttest_1samp((final_rf["mse_rf"]-final_rf["mse_drf"]), 0)[1],5))
print(round(ttest_1samp((final_rf["mse_rfx"]-final_rf["mse_drfx"]), 0)[1],5))

# Main plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
conf=[1.65*arima_wmse_std[0],1.65*arima_wmse_std[1],1.65*arima_wmse_std[2],1.65*arima_wmse_std[3]]
ax1.scatter([0,1,2,3],arima_wmse,color="black",marker='o',s=80)
ax1.errorbar([0,1,2,3],arima_wmse,yerr=conf,color="black",linewidth=2,fmt='none')

conf2=[1.65*rf_wmse_std[0],1.65*rf_wmse_std[1],1.65*rf_wmse_std[2],1.65*rf_wmse_std[3]]
ax2.scatter([0,1,2,3],rf_wmse,color="black",marker='o',s=80)
ax2.errorbar([0,1,2,3],rf_wmse,yerr=conf2,color="black",linewidth=2,fmt='none')

ax1.grid(False)
ax2.grid(False)
ax1.set_ylim(0.19,0.26)
ax2.set_ylim(0.27,0.34)

ax1.plot([0,1],[0.213,0.213],linewidth=0.5,color="black")
ax1.plot([0,0],[0.213,0.214],linewidth=0.5,color="black")
ax1.plot([1,1],[0.213,0.214],linewidth=0.5,color="black")
ax1.text(0.45, 0.211, "***", fontsize=12)

ax1.plot([0,2],[0.249,0.249],linewidth=0.5,color="black")
ax1.plot([0,0],[0.249,0.248],linewidth=0.5,color="black")
ax1.plot([2,2],[0.249,0.248],linewidth=0.5,color="black")
ax1.text(1, 0.2493, "o", fontsize=12)

ax1.plot([2,3],[0.213,0.213],linewidth=0.5,color="black")
ax1.plot([2,2],[0.213,0.214],linewidth=0.5,color="black")
ax1.plot([3,3],[0.213,0.214],linewidth=0.5,color="black")
ax1.text(2.45, 0.211, "***", fontsize=12)

# ------

ax2.plot([0,1],[0.292,0.292],linewidth=0.5,color="black")
ax2.plot([0,0],[0.292,0.293],linewidth=0.5,color="black")
ax2.plot([1,1],[0.292,0.293],linewidth=0.5,color="black")
ax2.text(0.45, 0.29, "***", fontsize=12)

ax2.plot([0,2],[0.333,0.333],linewidth=0.5,color="black")
ax2.plot([0,0],[0.333,0.332],linewidth=0.5,color="black")
ax2.plot([2,2],[0.333,0.332],linewidth=0.5,color="black")
ax2.text(1, 0.3333, "o", fontsize=12)

ax2.plot([2,3],[0.292,0.292],linewidth=0.5,color="black")
ax2.plot([2,2],[0.292,0.293],linewidth=0.5,color="black")
ax2.plot([3,3],[0.292,0.293],linewidth=0.5,color="black")
ax2.text(2.45, 0.29, "***", fontsize=12)

ax1.set_ylabel("Weighted mean squared error (WMSE)",size=22)
ax2.yaxis.set_ticks_position('right')
ax1.set_xticks([*range(4)],['ARIMA','DARIMA','ARIMAX','DARIMAX'],fontsize=18)
ax2.set_xticks([*range(4)],['RF','DRF','DRX','DRFX'],fontsize=18)
plt.subplots_adjust(wspace=0.1)
plt.savefig("results/results_main_plot.eps",dpi=400,bbox_inches="tight")

# (2) MSE #
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_arima),5))
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_darima),5))
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_arimax),5))
print(round(mean_squared_error(final_arima.n_protest_events, final_arima.preds_darimax),5))

final_arima["mse_arima"]=(final_arima["n_protest_events"] - final_arima["preds_arima"]) ** 2  
final_arima["mse_darima"]=(final_arima["n_protest_events"] - final_arima["preds_darima"]) ** 2  
final_arima["mse_arimax"]=(final_arima["n_protest_events"] - final_arima["preds_arimax"]) ** 2  
final_arima["mse_darimax"]=(final_arima["n_protest_events"] - final_arima["preds_darimax"]) ** 2  

# Print mean error
print(final_arima["mse_arima"].mean())
print(final_arima["mse_darima"].mean())
print(final_arima["mse_arimax"].mean())
print(final_arima["mse_darimax"].mean())

# Print std for error
print(round(final_arima["mse_arima"].std()/np.sqrt(len(final_arima)),5))
print(round(final_arima["mse_darima"].std()/np.sqrt(len(final_arima)),5))
print(round(final_arima["mse_arimax"].std()/np.sqrt(len(final_arima)),5))
print(round(final_arima["mse_darimax"].std()/np.sqrt(len(final_arima)),5))

# t test for whether difference is not zero
print(round(ttest_1samp((final_arima["mse_arima"]-final_arima["mse_arimax"]), 0)[1],5))
print(round(ttest_1samp((final_arima["mse_arima"]-final_arima["mse_darima"]), 0)[1],5))
print(round(ttest_1samp((final_arima["mse_arimax"]-final_arima["mse_darimax"]), 0)[1],5))

# (2) MSE #
print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_rf),5))
print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_drf),5))
print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_rfx),5))
print(round(mean_squared_error(final_rf.n_protest_events, final_rf.preds_drfx),5))

final_rf["mse_rf"]=(final_rf["n_protest_events"] - final_rf["preds_rf"]) ** 2  
final_rf["mse_drf"]=(final_rf["n_protest_events"] - final_rf["preds_drf"]) ** 2  
final_rf["mse_rfx"]=(final_rf["n_protest_events"] - final_rf["preds_rfx"]) ** 2  
final_rf["mse_drfx"]=(final_rf["n_protest_events"] - final_rf["preds_drfx"]) ** 2  

# Print mean error
print(final_rf["mse_rf"].mean())
print(final_rf["mse_drf"].mean())
print(final_rf["mse_rfx"].mean())
print(final_rf["mse_drfx"].mean())

# Print std for error
print(round(final_rf["mse_rf"].std()/np.sqrt(len(final_rf)),5))
print(round(final_rf["mse_drf"].std()/np.sqrt(len(final_rf)),5))
print(round(final_rf["mse_rfx"].std()/np.sqrt(len(final_rf)),5))
print(round(final_rf["mse_drfx"].std()/np.sqrt(len(final_rf)),5))

# t test for whether difference is not zero
print(round(ttest_1samp((final_rf["mse_rf"]-final_rf["mse_rfx"]), 0)[1],5))
print(round(ttest_1samp((final_rf["mse_rf"]-final_rf["mse_drf"]), 0)[1],5))
print(round(ttest_1samp((final_rf["mse_rfx"]-final_rf["mse_drfx"]), 0)[1],5))

#####################
### Density plots ###
#####################

results={"country":[],"WMSE_preds_arima":[],"WMSE_preds_darima":[],"WMSE_preds_rf":[],"WMSE_preds_drf":[]}
for i in range(len(country_keep)): 
    results["country"].append(country_keep[i])
    results["WMSE_preds_arima"].append(mean_squared_error(final_arima.loc[final_arima["country"]==country_keep[i]].n_protest_events,final_arima.loc[final_arima["country"]==country_keep[i]].preds_arima,sample_weight=final_arima.loc[final_arima["country"]==country_keep[i]].n_protest_events))  
    results["WMSE_preds_darima"].append(mean_squared_error(final_arima.loc[final_arima["country"]==country_keep[i]].n_protest_events, final_arima.loc[final_arima["country"]==country_keep[i]].preds_darima,sample_weight=final_arima.loc[final_arima["country"]==country_keep[i]].n_protest_events))
    results["WMSE_preds_rf"].append(mean_squared_error(final_rf.loc[final_rf["country"]==country_keep[i]].n_protest_events,final_rf.loc[final_rf["country"]==country_keep[i]].preds_rf,sample_weight=final_rf.loc[final_rf["country"]==country_keep[i]].n_protest_events))   
    results["WMSE_preds_drf"].append(mean_squared_error(final_rf.loc[final_rf["country"]==country_keep[i]].n_protest_events, final_rf.loc[final_rf["country"]==country_keep[i]].preds_drf,sample_weight=final_rf.loc[final_rf["country"]==country_keep[i]].n_protest_events))
results=pd.DataFrame(results)
results['ARIMA_change'] = (results["WMSE_preds_arima"]-results["WMSE_preds_darima"])
results['RF_change'] = (results["WMSE_preds_rf"]-results["WMSE_preds_drf"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sns.kdeplot(results['ARIMA_change'],ax=ax1,color='black',linewidth=2)
ax1.axvline(0, color='black', linestyle='--', linewidth=1)
sns.kdeplot(results['RF_change'],ax=ax2,color='black',linewidth=2)
ax2.axvline(0, color='black', linestyle='--', linewidth=1)
plt.subplots_adjust(wspace=0.05)
ax1.set_ylabel("Density values",fontsize=18)
ax2.set_ylabel("")
ax1.set_xlabel("ARIMA",fontsize=18)
ax2.set_xlabel("RF",fontsize=18)
ax1.set_yticks([0,5,10,15,20,25,30,35],[0,5,10,15,20,25,30,35],fontsize=18)
ax1.set_ylim(-1, 35)
ax2.set_ylim(-0.6, 18)

ax1.set_xlim(-0.05, 0.18)
ax1.set_xticks([-0.05,0,0.05,0.1,0.15],[-0.05,0,0.05,0.1,0.15],fontsize=18)
ax2.set_xlim(-0.07, 0.22)
ax2.set_xticks([-0.05,0,0.05,0.1,0.15,0.2],[-0.05,0,0.05,0.1,0.15,0.2],fontsize=18)
ax2.set_yticks([0,2,4,6,8,10,12,14,16,18],[0,2,4,6,8,10,12,14,16,18],fontsize=18)
ax2.yaxis.set_ticks_position('right')

results.sort_values(by=['ARIMA_change'], ascending=True, inplace=True)

selected_points = [(results['ARIMA_change'].iloc[0], "161816")]
for value, label in selected_points:
    ax1.text(value, 0.6, label, fontsize=12, color='black', ha='center')
ax1.plot(results['ARIMA_change'].iloc[0], 0, marker='x', color='black',markersize=10)

selected_points = [(results['ARIMA_change'].iloc[1], "160373")]
for value, label in selected_points:
    ax1.text(value, 1.4, label, fontsize=12, color='black', ha='center')
ax1.plot(results['ARIMA_change'].iloc[1], 0, marker='x', color='black',markersize=10)

 # --------
 
selected_points = [(results['ARIMA_change'].iloc[-1], "157475")]
for value, label in selected_points:
    ax1.text(value, 0.6, label, fontsize=12, color='black', ha='center')
ax1.plot(results['ARIMA_change'].iloc[-1], 0, marker='x', color='black',markersize=10)

selected_points = [(results['ARIMA_change'].iloc[-2], "153161")]
for value, label in selected_points:
    ax1.text(value, 1.4, label, fontsize=12, color='black', ha='center')  
ax1.plot(results['ARIMA_change'].iloc[-2], 0, marker='x', color='black',markersize=10)

selected_points = [(results['ARIMA_change'].iloc[-3], "163970")]
for value, label in selected_points:
    ax1.text(value, 2.2, label, fontsize=12, color='black', ha='center')
ax1.plot(results['ARIMA_change'].iloc[-3], 0, marker='x', color='black',markersize=10)

selected_points = [(results['ARIMA_change'].iloc[-4], "175473")]
for value, label in selected_points:
    ax1.text(value, 3, label, fontsize=12, color='black', ha='center')
ax1.plot(results['ARIMA_change'].iloc[-4], 0, marker='x', color='black',markersize=10)

results.sort_values(by=['RF_change'], ascending=True, inplace=True)

selected_points = [(results['RF_change'].iloc[-1], "175473")]
for value, label in selected_points:
    ax2.text(value, 0.2, label, fontsize=12, color='black', ha='center')
ax2.plot(results['RF_change'].iloc[-1], -0.1, marker='x', color='black',markersize=10)

selected_points = [(results['RF_change'].iloc[-2], "153883")]
for value, label in selected_points:   
    ax2.text(value, 0.6, label, fontsize=12, color='black', ha='center')
ax2.plot(results['RF_change'].iloc[-2], -0.1, marker='x', color='black',markersize=10)

selected_points = [(results['RF_change'].iloc[-3], "171878")]
for value, label in selected_points:
    ax2.text(value, 1, label, fontsize=12, color='black', ha='center')
ax2.plot(results['RF_change'].iloc[-3], -0.1, marker='x', color='black',markersize=10)

selected_points = [(results['RF_change'].iloc[-4], "166856")]
for value, label in selected_points:
    ax2.text(value, 1.4, label, fontsize=12, color='black', ha='center')
ax2.plot(results['RF_change'].iloc[-4], -0.1, marker='x', color='black',markersize=10)

selected_points = [(results['RF_change'].iloc[-5], "171877")]
for value, label in selected_points:
    ax2.text(value, 1.8, label, fontsize=12, color='black', ha='center')
ax2.plot(results['RF_change'].iloc[-5], -0.1, marker='x', color='black',markersize=10)

# ----

selected_points = [(results['RF_change'].iloc[0], "150278")]
for value, label in selected_points:
    ax2.text(value, 0.2, label, fontsize=12, color='black', ha='center')     
ax2.plot(results['RF_change'].iloc[0], -0.1, marker='x', color='black',markersize=10)

selected_points = [(results['RF_change'].iloc[1], "166868")]
for value, label in selected_points:   
    ax2.text(value, 0.6, label, fontsize=12, color='black', ha='center')
ax2.plot(results['RF_change'].iloc[1], -0.1, marker='x', color='black',markersize=10)


plt.savefig("results/results_main_dist.eps",dpi=400,bbox_inches="tight")


############################
### Directional accuracy ###
############################

final_arima = pd.read_csv("data/predictions/df_linear.csv",index_col=0)

final_arima['actual_lag'] = final_arima.groupby('country')['n_protest_events'].shift(1)
final_arima['preds_arima_lag'] = final_arima.groupby('country')['preds_arima'].shift(1)
final_arima['preds_darima_lag'] = final_arima.groupby('country')['preds_darima'].shift(1)

final_arima["actual_change"] = final_arima["n_protest_events"] - final_arima["actual_lag"]
final_arima["pred_arima_change"] = final_arima["preds_arima"] - final_arima["preds_arima_lag"]
final_arima["pred_darima_change"] = final_arima["preds_darima"] - final_arima["preds_darima_lag"]

# ARIMA # 
final_arima["correct_no_change_arima"] = ((final_arima.actual_change == 0) & (final_arima.pred_arima_change == 0)).astype(int)
final_arima["correct_direction_arima"] = ((np.sign(final_arima.actual_change) == np.sign(final_arima.pred_arima_change)) &(final_arima.actual_change != 0)).astype(int)
final_arima["incorrect_arima"]=0
final_arima.loc[((final_arima["correct_no_change_arima"]!=1) & (final_arima["correct_direction_arima"]!=1)), "incorrect_arima"] = 1

# DARIMA # 
final_arima["correct_no_change_darima"] = ((final_arima.actual_change == 0) & (final_arima.pred_darima_change == 0)).astype(int)
final_arima["correct_direction_darima"] = ((np.sign(final_arima.actual_change) == np.sign(final_arima.pred_darima_change)) &(final_arima.actual_change != 0)).astype(int)
final_arima["incorrect_darima"]=0
final_arima.loc[((final_arima["correct_no_change_darima"]!=1) & (final_arima["correct_direction_darima"]!=1)), "incorrect_darima"] = 1

final_arima=final_arima.dropna()

da_arima=[(final_arima['correct_direction_arima'].sum()/len(final_arima))*100,
          (final_arima['correct_no_change_arima'].sum()/len(final_arima))*100,
          (final_arima['incorrect_arima'].sum()/len(final_arima))*100]
print(da_arima)

da_darima=[(final_arima['correct_direction_darima'].sum()/len(final_arima))*100,
           (final_arima['correct_no_change_darima'].sum()/len(final_arima))*100,
           (final_arima['incorrect_darima'].sum()/len(final_arima))*100]
print(da_darima)


final_rf =  pd.read_csv("data/predictions/df_nonlinear.csv",index_col=0)

final_rf['actual_lag'] = final_rf.groupby('country')['n_protest_events'].shift(1)
final_rf['preds_rf_lag'] = final_rf.groupby('country')['preds_rf'].shift(1)
final_rf['preds_drf_lag'] = final_rf.groupby('country')['preds_drf'].shift(1)

final_rf["actual_change"] = final_rf["n_protest_events"] - final_rf["actual_lag"]
final_rf["pred_rf_change"] = final_rf["preds_rf"] - final_rf["preds_rf_lag"]
final_rf["pred_drf_change"] = final_rf["preds_drf"] - final_rf["preds_drf_lag"]

# RF # 
final_rf["correct_no_change_rf"] = ((final_rf.actual_change == 0) & (final_rf.pred_rf_change == 0)).astype(int)
final_rf["correct_direction_rf"] = ((np.sign(final_rf.actual_change) == np.sign(final_rf.pred_rf_change)) &(final_rf.actual_change != 0)).astype(int)
final_rf["incorrect_rf"]=0
final_rf.loc[((final_rf["correct_no_change_rf"]!=1) & (final_rf["correct_direction_rf"]!=1)), "incorrect_rf"] = 1

# DARIMA # 
final_rf["correct_no_change_drf"] = ((final_rf.actual_change == 0) & (final_rf.pred_drf_change == 0)).astype(int)
final_rf["correct_direction_drf"] = ((np.sign(final_rf.actual_change) == np.sign(final_rf.pred_drf_change)) &(final_rf.actual_change != 0)).astype(int)
final_rf["incorrect_drf"]=0
final_rf.loc[((final_rf["correct_no_change_drf"]!=1) & (final_rf["correct_direction_drf"]!=1)), "incorrect_drf"] = 1

final_rf=final_rf.dropna()

da_rf=[(final_rf['correct_direction_rf'].sum()/len(final_rf))*100,
          (final_rf['correct_no_change_rf'].sum()/len(final_rf))*100,
          (final_rf['incorrect_rf'].sum()/len(final_rf))*100]
print(da_rf)

da_drf=[(final_rf['correct_direction_drf'].sum()/len(final_rf))*100,
          (final_rf['correct_no_change_drf'].sum()/len(final_rf))*100,
          (final_rf['incorrect_drf'].sum()/len(final_rf))*100]
print(da_drf)

# Main plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
ax1.bar([0.3,0.6],[da_arima[2],da_darima[2]],0.2,label="Incorrect",color="gainsboro")
ax1.bar([0.3,0.6],[da_arima[1],da_darima[1]],0.2,label="No change",color="darkgray",bottom=[da_arima[2],da_darima[2]])
ax1.bar([0.3,0.6],[da_arima[0],da_darima[0]],0.2,label="Correct",color="dimgray",bottom=[da_arima[2]+da_arima[1],da_darima[2]+da_darima[1]])
ax1.set_xticks([0.3, 0.6])
ax1.set_xticklabels(["ARIMA", "DARIMA"],size=25)
ax1.text(0.27, 36, round(da_arima[2],2), fontsize=25, color='black')
ax1.text(0.27, 76, round(da_arima[1],2), fontsize=25, color='black')
ax1.text(0.272, 93, round(da_arima[0],2), fontsize=25, color='black')
ax1.text(0.57, 34, round(da_darima[2],2), fontsize=25, color='black')
ax1.text(0.57, 73, round(da_darima[1],2), fontsize=25, color='black')
ax1.text(0.57, 92, round(da_darima[0],2), fontsize=25, color='black')
ax1.set_yticks([0,20,40,60,80,100],[0,20,40,60,80,100],size=27)

ax2.bar([0.3,0.6],[da_rf[2],da_drf[2]],0.2,label="Incorrect",color="gainsboro")
ax2.bar([0.3,0.6],[da_rf[1],da_drf[1]],0.2,label="No change",color="darkgray",bottom=[da_rf[2],da_drf[2]])
ax2.bar([0.3,0.6],[da_rf[0],da_drf[0]],0.2,label="Correct",color="dimgray",bottom=[da_rf[2]+da_rf[1],da_drf[2]+da_drf[1]])
ax2.set_xticks([0.3, 0.6])
ax2.set_xticklabels(["RF", "DRF"],size=25)
ax2.text(0.27, 19, round(da_rf[2],2), fontsize=25, color='black')
ax2.text(0.27, 62, round(da_rf[1],2), fontsize=25, color='black')
ax2.text(0.272, 91, round(da_rf[0],2), fontsize=25, color='black')
ax2.text(0.57, 18, round(da_drf[2],2), fontsize=25, color='black')
ax2.text(0.57, 60, round(da_drf[1],2), fontsize=25, color='black')
ax2.text(0.572, 90, round(da_drf[0],2), fontsize=25, color='black')
ax2.set_yticks([0,20,40,60,80,100],[0,20,40,60,80,100],size=27)

patch3 = mpatches.Patch(color='dimgray', label='Correct direction')   
patch2 = mpatches.Patch(color='darkgray', label='Correct no change')        
patch1 = mpatches.Patch(color='gainsboro', label='Incorrect')        
fig.legend(handles=[patch3,patch2,patch1],fontsize=25,ncol=1,loc='right',bbox_to_anchor=(1.285,0.88),frameon=False,labelspacing=0.1)
plt.tight_layout()

plt.savefig("results/directional_accuracy.eps",dpi=400,bbox_inches="tight")

#################
### Centroids ### 
#################

with open("data/predictions/arima_shapes_thres0.5.json", 'r') as json_file:
    shapes_arima = json.load(json_file)
shapes_arima = {key: value for key, value in shapes_arima.items() if not key.startswith("darimax_")}  
  
with open("data/predictions/rf_shapes_thres0.5.json", 'r') as json_file:
    shapes_rf = json.load(json_file)
shapes_rf = {key: value for key, value in shapes_rf.items() if not key.startswith("drfx_")}  
  
# Random selection
shapes_arima_strip = {key.replace("darima_", ''): value for key, value in shapes_arima.items()}
shapes_rf_strip = {key.replace("drf_", ''): value for key, value in shapes_rf.items()}

# Plot
fig = plt.figure(figsize=(22, 12))
outer_grid = GridSpec(1, 2, figure=fig, wspace=0.05)
inner_grid_1 = GridSpecFromSubplotSpec(6, 7, subplot_spec=outer_grid[0])  
for i,n in zip(range(6),random.choices(list(shapes_arima_strip.keys()), k=6)):
    for j, seq in enumerate(shapes_arima[f"darima_{n}"][1]):
        ax = fig.add_subplot(inner_grid_1[i, j])
        ax.plot(seq,linewidth=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(y=-0.1,linewidth=0.5)
        plt.subplots_adjust(wspace=0.01,hspace=0.5)
        if j == 1:
            ax.set_title(f"Grid {n}", fontsize=25)

inner_grid_2 = GridSpecFromSubplotSpec(6, 7, subplot_spec=outer_grid[1])  
for i,n in zip(range(6),random.choices(list(shapes_rf_strip.keys()), k=6)):
    for j, seq in enumerate(shapes_rf[f"drf_{n}"][1]):
        ax = fig.add_subplot(inner_grid_2[i, j])
        ax.plot(seq,linewidth=2)
        if j == 1:
            ax.set_title(f"Grid {n}", fontsize=25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(y=-0.1,linewidth=0.5)
        plt.subplots_adjust(wspace=0.01,hspace=0.5)

left_box = patches.Rectangle((0.119, 0.1), 0.389, 0.825, linewidth=0.5, edgecolor='black', facecolor='none')
fig.add_artist(left_box)
fig.text(0.31,0.07, "ARIMA", fontsize=30, color='black', ha='center')
right_box = patches.Rectangle((0.517, 0.1), 0.389, 0.825, linewidth=0.5, edgecolor='black', facecolor='none')
fig.add_artist(right_box)
fig.text(0.7,0.07, "RF", fontsize=30, color='black', ha='center')
    
plt.savefig("results/results_main_centroids_random.eps",dpi=400,bbox_inches="tight")

########################
### Prediction plots ###
########################

final_arima = pd.read_csv("data/predictions/df_linear.csv",index_col=0)
final_rf =  pd.read_csv("data/predictions/df_nonlinear.csv",index_col=0)

# Good examples
fig = plt.figure(figsize=(12, 12))
inner_grid_1 = GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_grid[0], wspace=0.1, hspace=0.35)  # 2x2 grid for first subplot
for n,y,i,j,m in zip([157475,171877,171877,171878],[2022,2022,2023,2023],[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1],["a","r","r","r"]):
    ax = fig.add_subplot(inner_grid_1[i, j])
    if m=="a":
        plt.plot(final_arima["dd"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],final_arima["n_protest_events"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],label="Actuals",linestyle="solid",color="black",linewidth=1)
        plt.plot(final_arima["dd"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],final_arima["preds_arima"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],label="ARIMA",linestyle="dotted",color="black",linewidth=1)
        plt.plot(final_arima["dd"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],final_arima["preds_darima"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],label="DARIMA",linestyle="dashed",color="black",linewidth=1)
        ax.set_ylim(-0.1, 1)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1],size=10)
        ax.set_title(f"Grid {n}", fontsize=15)
    if m=="r":
        plt.plot(final_rf["dd"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],final_rf["n_protest_events"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],label="Actuals",linestyle="solid",color="black",linewidth=1)
        plt.plot(final_rf["dd"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],final_rf["preds_rf"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],label="ARIMA",linestyle="dotted",color="black",linewidth=1)
        plt.plot(final_rf["dd"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],final_rf["preds_drf"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],label="DARIMA",linestyle="dashed",color="black",linewidth=1)
        ax.set_ylim(-0.1, 1)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1],size=10)
        ax.set_title(f"Grid {n}", fontsize=15)     
    if y==2022:
        plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],size=10)   
    if y==2023:
        plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],size=10)   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if j==1:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
plt.savefig("results/results_main_preds_best_select_short.eps",dpi=400,bbox_inches="tight")

# Bad examples 
fig = plt.figure(figsize=(12, 12))
outer_grid = GridSpec(1, 2, figure=fig, wspace=0.05)
inner_grid_1 = GridSpecFromSubplotSpec(4, 2, subplot_spec=outer_grid[0], wspace=0.1, hspace=0.35)  # 2x2 grid for first subplot
for n,y,i,j,m in zip([161816,160373,150278,166868],[2023,2023,2022,2022],[0,0,1,1,2,2,3,3],[0,1,0,1,0,1,0,1],["a","a","r","r"]):
    ax = fig.add_subplot(inner_grid_1[i, j])
    if m=="a":
        plt.plot(final_arima["dd"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],final_arima["n_protest_events"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],label="Actuals",linestyle="solid",color="black",linewidth=1)
        plt.plot(final_arima["dd"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],final_arima["preds_arima"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],label="ARIMA",linestyle="dotted",color="black",linewidth=1)
        plt.plot(final_arima["dd"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],final_arima["preds_darima"].loc[(final_arima["country"]==n)&(final_arima["dd"]<=f"{y}-12")&(final_arima["dd"]>=f"{y}-01")],label="DARIMA",linestyle="dashed",color="black",linewidth=1)
        ax.set_ylim(-0.1, 1)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1],size=10)
        ax.set_title(f"Grid {n}", fontsize=15)
    if m=="r":
        plt.plot(final_rf["dd"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],final_rf["n_protest_events"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],label="Actuals",linestyle="solid",color="black",linewidth=1)
        plt.plot(final_rf["dd"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],final_rf["preds_rf"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],label="ARIMA",linestyle="dotted",color="black",linewidth=1)
        plt.plot(final_rf["dd"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],final_rf["preds_drf"].loc[(final_rf["country"]==n)&(final_rf["dd"]<=f"{y}-12")&(final_rf["dd"]>=f"{y}-01")],label="DARIMA",linestyle="dashed",color="black",linewidth=1)
        ax.set_ylim(-0.1, 1)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1],size=10)
        ax.set_title(f"Grid {n}", fontsize=15)
    if y==2022:
        plt.xticks(["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],["01-22","03-22","05-22","07-22","09-22","11-22"],size=10)   
    if y==2023:
        plt.xticks(["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],["01-23","03-23","05-23","07-23","09-23","11-23"],size=10)   
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if j==1:
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
            
plt.savefig("results/results_main_preds_bad_select_short.eps",dpi=400,bbox_inches="tight")

##########################
### Validate threshold ###
##########################

final_arima_static = pd.read_csv("data/predictions/linear_static.csv",index_col=0)

# Find countries with high intensity for different thresholds
for thres in [0.5,0.55,0.6,0.65,0.7]:
    print(f"Threshold {thres}")
    df_n_country_month = {}
    for i in acled["gid"].unique():
        ts = acled["n_protest_events"].loc[acled["gid"]==i][:int(0.7*len(acled["n_protest_events"].loc[acled["gid"]==i]))]
        df_n_country_month[i] = np.mean(np.abs(ts - np.mean(ts)))
    df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
    df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True) 
    country_keeps = df_n_country_month.loc[df_n_country_month["avg"]>thres].gid.unique()
    
    # Dynamic models
    final_arima_dynamic = pd.read_csv("data/predictions/linear_dynamic_thres0.5.csv",index_col=0)
    final_arima_dynamic = final_arima_dynamic[final_arima_dynamic['country'].isin(country_keeps)]
    
    # Merge
    final_arima=pd.merge(final_arima_static,final_arima_dynamic[["dd","country",'preds_darima','preds_darimax']],on=["dd","country"],how="left")
    final_arima=final_arima.sort_values(by=["country","dd"])
    final_arima=final_arima.reset_index(drop=True)
    final_arima['preds_darima'] = final_arima['preds_darima'].fillna(final_arima['preds_arima'])
    final_arima['preds_darimax'] = final_arima['preds_darimax'].fillna(final_arima['preds_arimax'])

    # Evaluate
    final_arima["mse_arima"]=final_arima["n_protest_events"]*(final_arima["n_protest_events"] - final_arima["preds_arima"]) ** 2  
    final_arima["mse_darima"]=final_arima["n_protest_events"]*(final_arima["n_protest_events"] - final_arima["preds_darima"]) ** 2   
    print(round((mean_squared_error(final_arima.n_protest_events,final_arima.preds_arima,sample_weight=final_arima.n_protest_events)-mean_squared_error(final_arima.n_protest_events, final_arima.preds_darima,sample_weight=final_arima.n_protest_events)),5))
    print(round(ttest_1samp((final_arima["mse_arima"]-final_arima["mse_darima"]), 0)[1],5))
    
# Validate threshold
final_rf_static = pd.read_csv("data/predictions/nonlinear_static.csv",index_col=0)

# Find countries with high intensity for different thresholds
for thres in [0.5,0.55,0.6,0.65,0.7]:
    print(f"Threshold {thres}")
    df_n_country_month = {}
    for i in acled["gid"].unique():
        ts = acled["n_protest_events"].loc[acled["gid"]==i][:int(0.7*len(acled["n_protest_events"].loc[acled["gid"]==i]))]
        df_n_country_month[i] = np.mean(np.abs(ts - np.mean(ts)))
    df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
    df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True) 
    country_keeps = df_n_country_month.loc[df_n_country_month["avg"]>thres].gid.unique()
    
    # Dynamic models
    final_rf_dynamic = pd.read_csv("data/predictions/nonlinear_dynamic_thres0.5.csv",index_col=0)
    final_rf_dynamic = final_rf_dynamic[final_rf_dynamic['country'].isin(country_keeps)]
    
    # Merge
    final_rf=pd.merge(final_rf_static,final_rf_dynamic[["dd","country",'preds_drf','preds_drfx']],on=["dd","country"],how="left")
    final_rf=final_rf.sort_values(by=["country","dd"])
    final_rf=final_rf.reset_index(drop=True)
    final_rf['preds_drf'] = final_rf['preds_drf'].fillna(final_rf['preds_rf'])
    final_rf['preds_drfx'] = final_rf['preds_drfx'].fillna(final_rf['preds_rfx'])

    # Evaluate
    final_rf["mse_rf"]=final_rf["n_protest_events"]*(final_rf["n_protest_events"] - final_rf["preds_rf"]) ** 2  
    final_rf["mse_drf"]=final_rf["n_protest_events"]*(final_rf["n_protest_events"] - final_rf["preds_drf"]) ** 2   
    print(round((mean_squared_error(final_rf.n_protest_events,final_rf.preds_rf,sample_weight=final_rf.n_protest_events)-mean_squared_error(final_rf.n_protest_events, final_rf.preds_drf,sample_weight=final_rf.n_protest_events)),5))
    print(round(ttest_1samp((final_rf["mse_rf"]-final_rf["mse_drf"]), 0)[1],5))
    
###################################
### Clustering of the centroids ###
###################################
  
with open("data/predictions/rf_shapes_thres0.5.json", 'r') as json_file:
    shapes_rf = json.load(json_file)
shapes_rf = {key: value for key, value in shapes_rf.items() if not key.startswith("drfx_")}  
  
score_test=-1
for k in [3,5,7]:
    df_cen=pd.DataFrame()
    # For each country
    for d in shapes_rf.keys():
        # For each centroid
        for i in range(len(shapes_rf[d][1])):
            # save country (d[2:]) and cluster number (i) --> needed for merging later
            row = pd.DataFrame([[d[4:], i]], columns=['country', 'clusters'])
            # Obtain corresponding centroid, by converting list of lists into list
            cen=[]
            for x in range(len(shapes_rf[d][1][i])):
                cen.append(shapes_rf[d][1][i][x][0])
            # Convert list with centroid into df, so that each column is one point                    
            centro=pd.DataFrame(cen).T
            # Add centroid to country and cluster number
            row=pd.concat([row,centro],axis=1)
            # Add row to out df
            df_cen=pd.concat([df_cen,row])
    
    arr=df_cen[[0,1,2,3,4,5,6,7,8]].values
    rows_without_nan = []
    # Iterate over each row in the dataset
    for row in arr:
        row=row.astype(float)
        # Remove missing values from the row and append to the list
        rows_without_nan.append(row[~np.isnan(row)])
    
    distance_matrix = dtw.distance_matrix_fast(rows_without_nan)  
    condensed_dist_matrix = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist_matrix, method='complete')
    clusters = fcluster(linkage_matrix, t=k, criterion='maxclust')
    df_cen["clusters_cen"]=clusters

    score = silhouette_score(rows_without_nan, clusters,metric="dtw")
    print(round(score,5))
    
    if score>score_test: 
        score_test=score
        df_cen_final_rf=df_cen
        unique_clusters = np.unique(clusters)

        # Define out list
        representatives = []
        # Loop over clusters
        for k_num in unique_clusters:
            # Get all centroids assigned to the specific cluster
            cluster_seq=[]
            # Loop over each case
            for i, cluster in enumerate(clusters): 
                # and save centroid if the case belongs to cluster k
                if cluster == k_num:
                    cluster_seq.append(rows_without_nan[i])
                    
            # Then calculate the centroid using DTW Barycenter Averaging (DBA)
            # takes the mean for time series
            cen = dtw_barycenter_averaging(cluster_seq, barycenter_size=9)
            # Save centroid
            representatives.append(cen.ravel())

        n_obs=np.unique(clusters,return_counts=True)
        n_clusters = len(representatives)
        cols = 3
        rows = n_clusters // cols + (n_clusters % cols > 0)
            
        # Plot centroids
        plt.figure(figsize=(12.5, 4 * rows))
        for i, seq, n in zip(range(n_clusters),representatives, n_obs[1]):
            plt.subplot(rows, cols, i+1)
            plt.plot(seq,linewidth=3,color="black")
            plt.title(f'Cluster {i+1}, n={n}',size=37)
            plt.ylim(-0.05,1.1)
            plt.yticks([],[])
            plt.xticks([],[])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        plt.savefig("results/clusters_India_rf.eps",dpi=400,bbox_inches="tight")
        plt.show()
        
        
        n_clusters = len(representatives)
        n_member_plots = 16 
        member_cols = 8    
        
        # Plot centroids and representatives
        fig = plt.figure(figsize=(17, 3 * n_clusters))
        outer = fig.add_gridspec(n_clusters, 1, hspace=0.2)
        
        for i in range(n_clusters):
            inner = outer[i].subgridspec(1, 2, width_ratios=[0.9, 3.1], wspace=0.1)
            ax_c = fig.add_subplot(inner[0, 0])
            
            # Centroids
            ax_c.plot(representatives[i], color="black", linewidth=3)
            ax_c.set_title(f"Cluster {i+1}, n={n_obs[1][i]}", fontsize=27)
            ax_c.set_ylim(-0.05,1.1)
            ax_c.set_xticks([])
            ax_c.set_yticks([])
        
            # Representatives
            member_idx = np.where(clusters == i + 1)[0]
            members = [rows_without_nan[j] for j in member_idx]
            samples = random.sample(members, min(n_member_plots, len(members)))
        
            n_rows = int(np.ceil(len(samples) / member_cols))
            member_grid = inner[0, 1].subgridspec(n_rows, member_cols, hspace=0.35, wspace=0.3)
        
            for j, s in enumerate(samples):
                ax = fig.add_subplot(member_grid[j])
                ax.plot(s, color="gray", linewidth=2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(-0.05,1.1)
                ax.axis("off")
        
        plt.tight_layout()
        plt.savefig("results/dendogram_India_rf.eps", dpi=400, bbox_inches="tight")
        plt.show()
 
# Performance for each cluster

# Get data set with predictions and within-grid cluster assignments 
flattened = [item for v in shapes_rf.values() for item in v[2]]
final_rf_dynamic = pd.read_csv("data/predictions/nonlinear_dynamic_thres0.5.csv",index_col=0)
final_rf_static = pd.read_csv("data/predictions/nonlinear_static.csv",index_col=0)
final_rf=pd.merge(final_rf_dynamic,final_rf_static[["dd","country",'preds_rf','preds_rfx']],on=["dd","country"],how="left")
final_rf["clusters"]=flattened

# Merge observations over grid and within-cluster id
df_cen_final_rf["country"] = df_cen_final_rf["country"].astype(int)
df_final_cen_rf=pd.merge(final_rf, df_cen_final_rf[["country","clusters","clusters_cen"]],on=["clusters","country"])

# For each cluster, obtain wmse, standard error, and t-test
rf_clu_wmse=[]
rf_clu_wmse_std=[]

for x in [1,2,3,4,5]:
    print(f"Cluster {x}")
    print(round(mean_squared_error(df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].preds_rf,sample_weight=df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events),5))
    print(round(mean_squared_error(df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].preds_drf,sample_weight=df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events),5))
    
    rf_clu_wmse.append([mean_squared_error(df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].preds_rf,sample_weight=df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events),mean_squared_error(df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].preds_drf,sample_weight=df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events)])
    rf_clu_wmse_std.append([wmse_se(df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].preds_rf, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events),wmse_se(df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].preds_drf, df_final_cen_rf[df_final_cen_rf["clusters_cen"]==x].n_protest_events)])
    
    # t test for whether difference is not zero
    df_final_cen_rf["mse_rf"]=df_final_cen_rf["n_protest_events"]*(df_final_cen_rf["n_protest_events"] - df_final_cen_rf["preds_rf"]) ** 2  
    df_final_cen_rf["mse_drf"]=df_final_cen_rf["n_protest_events"]*(df_final_cen_rf["n_protest_events"] - df_final_cen_rf["preds_drf"]) ** 2  
    print(round(ttest_1samp((df_final_cen_rf["mse_rf"][df_final_cen_rf["clusters_cen"]==x]-df_final_cen_rf["mse_drf"][df_final_cen_rf["clusters_cen"]==x]), 0)[1],5))


         
with open("data/predictions/arima_shapes_thres0.5.json", 'r') as json_file:
    shapes_arima = json.load(json_file)
shapes_arima = {key: value for key, value in shapes_arima.items() if not key.startswith("darimax_")}  

score_test=-1
for k in [3,5,7]:
    df_cen=pd.DataFrame()
    # For each country
    for d in shapes_arima.keys():
        # For each centroid
        for i in range(len(shapes_arima[d][1])):
            # save country (d[2:]) and cluster number (i) --> needed for merging later
            row = pd.DataFrame([[d[7:], i]], columns=['country', 'clusters'])
            # Obtain corresponding centroid, by converting list of lists into list
            cen=[]
            for x in range(len(shapes_arima[d][1][i])):
                cen.append(shapes_arima[d][1][i][x][0])
            # Convert list with centroid into df, so that each column is one point                    
            centro=pd.DataFrame(cen).T
            # Add centroid to country and cluster number
            row=pd.concat([row,centro],axis=1)
            # Add row to out df
            df_cen=pd.concat([df_cen,row])
    
    arr=df_cen[[0,1,2,3,4,5,6,7,8]].values
    rows_without_nan = []
    # Iterate over each row in the dataset
    for row in arr:
        row=row.astype(float)
        # Remove missing values from the row and append to the list
        rows_without_nan.append(row[~np.isnan(row)])
    
    distance_matrix = dtw.distance_matrix_fast(rows_without_nan)  
    condensed_dist_matrix = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist_matrix, method='complete')
    clusters = fcluster(linkage_matrix, t=k, criterion='maxclust')
    df_cen["clusters_cen"]=clusters

    score = silhouette_score(rows_without_nan, clusters,metric="dtw")
    print(round(score,5))
    
    if score>score_test: 
        score_test=score
        df_cen_final_arima=df_cen
        unique_clusters = np.unique(clusters)

        # Define out list
        representatives = []
        # Loop over clusters
        for k_num in unique_clusters:
            # Get all centroids assigned to the specific cluster
            cluster_seq=[]
            # Loop over each case
            for i, cluster in enumerate(clusters): 
                # and save centroid if the case belongs to cluster k
                if cluster == k_num:
                    cluster_seq.append(rows_without_nan[i])
                    
            # Then calculate the centroid using DTW Barycenter Averaging (DBA)
            # takes the mean for time series
            cen = dtw_barycenter_averaging(cluster_seq, barycenter_size=9)
            # Save centroid
            representatives.append(cen.ravel())

        n_obs=np.unique(clusters,return_counts=True)
        n_clusters = len(representatives)
        cols = 3
        rows = n_clusters // cols + (n_clusters % cols > 0)
            
        # Plot centroids
        plt.figure(figsize=(12.5, 4 * rows))
        for i, seq, n in zip(range(n_clusters),representatives, n_obs[1]):
            plt.subplot(rows, cols, i+1)
            plt.plot(seq,linewidth=3,color="black")
            plt.title(f'Cluster {i+1}, n={n}',size=37)
            plt.ylim(-0.05,1.1)
            plt.yticks([],[])
            plt.xticks([],[])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        plt.savefig("results/clusters_India_arima.eps",dpi=400,bbox_inches="tight")
        plt.show()
        
        
        n_clusters = len(representatives)
        n_member_plots = 16 
        member_cols = 8    
        
        # Plot centroids and representatives
        fig = plt.figure(figsize=(17, 3 * n_clusters))
        outer = fig.add_gridspec(n_clusters, 1, hspace=0.2)
        
        for i in range(n_clusters):
            inner = outer[i].subgridspec(1, 2, width_ratios=[0.9, 3.1], wspace=0.1)
            ax_c = fig.add_subplot(inner[0, 0])
            
            # Centroids
            ax_c.plot(representatives[i], color="black", linewidth=3)
            ax_c.set_title(f"Cluster {i+1}, n={n_obs[1][i]}", fontsize=27)
            ax_c.set_ylim(-0.05,1.1)
            ax_c.set_xticks([])
            ax_c.set_yticks([])
        
            # Representatives
            member_idx = np.where(clusters == i + 1)[0]
            members = [rows_without_nan[j] for j in member_idx]
            samples = random.sample(members, min(n_member_plots, len(members)))
        
            n_rows = int(np.ceil(len(samples) / member_cols))
            member_grid = inner[0, 1].subgridspec(n_rows, member_cols, hspace=0.35, wspace=0.3)
        
            for j, s in enumerate(samples):
                ax = fig.add_subplot(member_grid[j])
                ax.plot(s, color="gray", linewidth=2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylim(-0.05,1.1)
                ax.axis("off")
        
        plt.tight_layout()
        plt.savefig("results/dendogram_India_arima.eps", dpi=400, bbox_inches="tight")
        plt.show()
 
# Performance for each cluster


# Get data set with predictions and within-grid cluster assignments 
flattened = [item for v in shapes_arima.values() for item in v[2]]
final_arima_dynamic = pd.read_csv("data/predictions/linear_dynamic_thres0.5.csv",index_col=0)
final_arima_static = pd.read_csv("data/predictions/linear_static.csv",index_col=0)
final_arima=pd.merge(final_arima_dynamic,final_arima_static[["dd","country",'preds_arima','preds_arimax']],on=["dd","country"],how="left")
final_arima["clusters"]=flattened

# Merge observations over grid and within-cluster id
df_cen_final_arima["country"] = df_cen_final_arima["country"].astype(int)
df_final_cen_arima=pd.merge(final_arima, df_cen_final_arima[["country","clusters","clusters_cen"]],on=["clusters","country"])

# For each cluster, obtain wmse, standard error, and t-test
arima_clu_wmse=[]
arima_clu_wmse_std=[]

for x in [1,2,3,4,5]:
    print(f"Cluster {x}")
    print(round(mean_squared_error(df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].preds_arima,sample_weight=df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events),5))
    print(round(mean_squared_error(df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].preds_darima,sample_weight=df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events),5))

    arima_clu_wmse.append([mean_squared_error(df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].preds_arima,sample_weight=df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events), mean_squared_error(df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].preds_darima,sample_weight=df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events)])
    arima_clu_wmse_std.append([wmse_se(df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].preds_arima, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events),wmse_se(df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].preds_darima, df_final_cen_arima[df_final_cen_arima["clusters_cen"]==x].n_protest_events)])

    # t test for whether difference is not zero
    df_final_cen_arima["mse_arima"]=df_final_cen_arima["n_protest_events"]*(df_final_cen_arima["n_protest_events"] - df_final_cen_arima["preds_arima"]) ** 2  
    df_final_cen_arima["mse_darima"]=df_final_cen_arima["n_protest_events"]*(df_final_cen_arima["n_protest_events"] - df_final_cen_arima["preds_darima"]) ** 2  
    print(round(ttest_1samp((df_final_cen_arima["mse_arima"][df_final_cen_arima["clusters_cen"]==x]-df_final_cen_arima["mse_darima"][df_final_cen_arima["clusters_cen"]==x]), 0)[1],5))


# Main plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
conf=[1.65*arima_wmse_std[0],1.65*arima_wmse_std[1]]
axes[0,0].scatter([0,1],arima_wmse[:2],color="black",marker='o',s=80)
axes[0,0].errorbar([0,1],arima_wmse[:2],yerr=conf,color="black",linewidth=2,fmt='none')

ax2 = axes[0,0].twinx()
conf2=[1.65*arima_wmse_std[0],1.65*arima_wmse_std[1]]
ax2.scatter([2,3],rf_wmse[:2],color="black",marker='o',s=80)
ax2.errorbar([2,3],rf_wmse[:2],yerr=conf2,color="black",linewidth=2,fmt='none')
axes[0,0].set_xticks([*range(4)],['ARIMA','DARIMA','RF','DRF'],fontsize=15)

# Cluster 1
conf=[1.65*arima_clu_wmse_std[0][0],1.65*arima_clu_wmse_std[0][1]]
axes[0,1].scatter([0,1],arima_clu_wmse[0],color="black",marker='o',s=80)
axes[0,1].errorbar([0,1],arima_clu_wmse[0],yerr=conf,color="black",linewidth=2,fmt='none')

ax2 = axes[0,1].twinx()
conf2=[1.65*rf_clu_wmse_std[0][0],1.65*rf_clu_wmse_std[0][1]]
ax2.scatter([2,3],rf_clu_wmse[0],color="gray",marker='o',s=80)
ax2.errorbar([2,3],rf_clu_wmse[0],yerr=conf2,color="gray",linewidth=2,fmt='none')
axes[0,1].set_title("Cluster 1",size=22)
axes[0,1].set_xticks([*range(4)],['ARIMA','DARIMA','RF','DRF'],fontsize=15)

# Cluster 2
conf=[1.65*arima_clu_wmse_std[1][0],1.65*arima_clu_wmse_std[1][1]]
axes[0,2].scatter([0,1],arima_clu_wmse[1],color="black",marker='o',s=80)
axes[0,2].errorbar([0,1],arima_clu_wmse[1],yerr=conf,color="black",linewidth=2,fmt='none')

ax2 = axes[0,2].twinx()
conf2=[1.65*rf_clu_wmse_std[1][0],1.65*rf_clu_wmse_std[1][1]]
ax2.scatter([2,3],rf_clu_wmse[1],color="black",marker='o',s=80)
ax2.errorbar([2,3],rf_clu_wmse[1],yerr=conf2,color="black",linewidth=2,fmt='none')
axes[0,2].set_title("Cluster 2",size=22)
axes[0,2].set_xticks([*range(4)],['ARIMA','DARIMA','RF','DRF'],fontsize=15)

# Cluster 3
conf=[1.65*arima_clu_wmse_std[2][0],1.65*arima_clu_wmse_std[2][1]]
axes[1,0].scatter([0,1],arima_clu_wmse[2],color="black",marker='o',s=80)
axes[1,0].errorbar([0,1],arima_clu_wmse[2],yerr=conf,color="black",linewidth=2,fmt='none')

ax2 = axes[1,0].twinx()
conf2=[1.65*rf_clu_wmse_std[2][0],1.65*rf_clu_wmse_std[2][1]]
ax2.scatter([2,3],rf_clu_wmse[2],color="black",marker='o',s=80)
ax2.errorbar([2,3],rf_clu_wmse[2],yerr=conf2,color="black",linewidth=2,fmt='none')
axes[1,0].set_title("Cluster 3",size=22)
axes[1,0].set_xticks([*range(4)],['ARIMA','DARIMA','RF','DRF'],fontsize=15)

# Cluster 4
conf=[1.65*arima_clu_wmse_std[3][0],1.65*arima_clu_wmse_std[3][1]]
axes[1,1].scatter([0,1],arima_clu_wmse[3],color="gray",marker='o',s=80)
axes[1,1].errorbar([0,1],arima_clu_wmse[3],yerr=conf,color="gray",linewidth=2,fmt='none')

ax2 = axes[1,1].twinx()
conf2=[1.65*rf_clu_wmse_std[3][0],1.65*rf_clu_wmse_std[3][1]]
ax2.scatter([2,3],rf_clu_wmse[3],color="black",marker='o',s=80)
ax2.errorbar([2,3],rf_clu_wmse[3],yerr=conf2,color="black",linewidth=2,fmt='none')
axes[1,1].set_title("Cluster 4",size=22)
axes[1,1].set_xticks([*range(4)],['ARIMA','DARIMA','RF','DRF'],fontsize=15)

# Cluster 5
conf=[1.65*arima_clu_wmse_std[4][0],1.65*arima_clu_wmse_std[4][1]]
axes[1,2].scatter([0,1],arima_clu_wmse[4],color="black",marker='o',s=80)
axes[1,2].errorbar([0,1],arima_clu_wmse[4],yerr=conf,color="black",linewidth=2,fmt='none')

ax2 = axes[1,2].twinx()
conf2=[1.65*rf_clu_wmse_std[4][0],1.65*rf_clu_wmse_std[4][1]]
ax2.scatter([2,3],rf_clu_wmse[4],color="black",marker='o',s=80)
ax2.errorbar([2,3],rf_clu_wmse[4],yerr=conf2,color="black",linewidth=2,fmt='none')
axes[1,2].set_title("Cluster 5",size=22)
axes[1,2].set_xticks([*range(4)],['ARIMA','DARIMA','RF','DRF'],fontsize=15)

plt.tight_layout()
plt.savefig("results/results_per_clu.eps",dpi=400,bbox_inches="tight")
plt.show()

