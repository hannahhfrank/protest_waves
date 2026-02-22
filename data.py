import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load ACLED data from local folder 
acled = pd.read_csv("data/acled/acled_all_events.csv",low_memory=False,index_col=[0]) 

##########################
### Get data for India ###
##########################

# Only include certain events 
df_s = acled.loc[(acled['event_type']=="Protests") |
                (acled['event_type']=="Explosions/Remote violence") |
                (acled['event_type']=="Riots") |
                (acled['event_type']=="Strategic developments") |
                (acled['event_type']=="Battles") |
                (acled['event_type']=="Violence against civilians") ].copy(deep=True)

# Subset for India 
df_protest = df_s.loc[(df_s["country"] == "India") & (df_s["year"]==2016) | 
                      (df_s["country"] == "India") & (df_s["year"]==2017) | 
                      (df_s["country"] == "India") & (df_s["year"]==2018) | 
                      (df_s["country"] == "India") & (df_s["year"]==2019) | 
                      (df_s["country"] == "India") & (df_s["year"]==2020) |
                      (df_s["country"] == "India") & (df_s["year"]==2021) |
                      (df_s["country"] == "India") & (df_s["year"]==2022) |
                      (df_s["country"] == "India") & (df_s["year"]==2023)].copy(deep=True)

# Check frequency of sources for India 
counts = df_protest["source"].value_counts()

# Add dates 
df_protest["dd"] = pd.to_datetime(df_protest['event_date'],format='%d %B %Y')
df_protest["dd"] = df_protest["dd"].dt.strftime('%Y-%m')

# Add grid ids 
df_protest["gid"] = 0
df_protest.reset_index(inplace=True, drop=True)

# Check if value has no or only one decimal
def add_value_if_no_or_one_decimal(x, value_to_add):
    if x % 1 == 0 or round(x, 1) == x:  
        return x + value_to_add
    else:
        return x

# Apply the function to the column
value_to_add = 0.0001  
df_protest['longitude'] = df_protest['longitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)
df_protest['latitude'] = df_protest['latitude'].apply(add_value_if_no_or_one_decimal, value_to_add=value_to_add)

# Prio shape file 
prio_shp = gpd.read_file('data/acled/priogrid_cell.shp')
prio_shp.reset_index(inplace=True, drop=True)

# Assign events to grid id 
for i in range(len(df_protest)):
    prio_shp_s = prio_shp[(prio_shp["ycoord"]>df_protest.latitude.iloc[i]-1) &
                              (prio_shp["ycoord"]<df_protest.latitude.iloc[i]+1) &
                              (prio_shp["xcoord"]>df_protest.longitude.iloc[i]-1) &
                              (prio_shp["xcoord"]<df_protest.longitude.iloc[i]+1)]
    prio_shp_s.reset_index(inplace=True, drop=True)

    for x in range(len(prio_shp_s)):
        if prio_shp_s.geometry.iloc[x].contains(Point(df_protest.longitude.iloc[i], df_protest.latitude.iloc[i])) == True:
            df_protest.gid.iloc[i] = prio_shp_s.gid.iloc[x]
            break
            
###############
### Protest ###
###############

df_s = df_protest.loc[df_protest['event_type']=="Protests"]
agg_month = pd.DataFrame(df_s.groupby(["dd","year","gid","iso","country"]).size())
agg_protest = agg_month.reset_index()
agg_protest.rename(columns={0:"n_protest_events"}, inplace=True)

##################################
### Explosions/remote violence ###
##################################

df_s = df_protest.loc[df_protest['event_type']=="Explosions/Remote violence"]
agg_month = pd.DataFrame(df_s.groupby(["dd","year","gid","iso","country"]).size())
agg_remote = agg_month.reset_index()
agg_remote.rename(columns={0:"n_remote_events"}, inplace=True)

# Merge
df=pd.merge(agg_protest, agg_remote[["dd","gid","n_remote_events"]],on=["dd","gid"],how="outer")

#############
### Riots ###
#############

df_s = df_protest.loc[df_protest['event_type']=="Riots"]
agg_month = pd.DataFrame(df_s.groupby(["dd","year","gid","iso","country"]).size())
agg_riot = agg_month.reset_index()
agg_riot.rename(columns={0:"n_riot_events"}, inplace=True)

# Merge
df=pd.merge(df, agg_riot[["dd","gid","n_riot_events"]],on=["dd","gid"],how="outer")

#############################
### Strategic developmens ###
#############################

df_s = df_protest.loc[df_protest['event_type']=="Strategic developments"]
agg_month = pd.DataFrame(df_s.groupby(["dd","year","gid","iso","country"]).size())
agg_dev = agg_month.reset_index()
agg_dev.rename(columns={0:"n_stratdev_events"}, inplace=True)

# Merge
df=pd.merge(df, agg_dev[["dd","gid","n_stratdev_events"]],on=["dd","gid"],how="outer")     

###############
### Battles ###
###############

df_s = df_protest.loc[df_protest['event_type']=="Battles"]
agg_month = pd.DataFrame(df_s.groupby(["dd","year","gid","iso","country"]).size())
agg_dev = agg_month.reset_index()
agg_dev.rename(columns={0:"n_battles_events"}, inplace=True)

# Merge
df=pd.merge(df, agg_dev[["dd","gid","n_battles_events"]],on=["dd","gid"],how="outer")     

##################################
### Violence against civilians ###
##################################

df_s = df_protest.loc[df_protest['event_type']=="Violence against civilians"]
agg_month = pd.DataFrame(df_s.groupby(["dd","year","gid","iso","country"]).size())
agg_dev = agg_month.reset_index()
agg_dev.rename(columns={0:"n_civilvio_events"}, inplace=True)

# Merge
df=pd.merge(df, agg_dev[["dd","gid","n_civilvio_events"]],on=["dd","gid"],how="outer")     

# Fix missing values for observations with no protest events
df.loc[df["dd"]=="2016-01","year"]=2016
df.loc[df["dd"]=="2016-02","year"]=2016
df.loc[df["dd"]=="2016-03","year"]=2016
df.loc[df["dd"]=="2016-04","year"]=2016
df.loc[df["dd"]=="2016-05","year"]=2016
df.loc[df["dd"]=="2016-06","year"]=2016
df.loc[df["dd"]=="2016-07","year"]=2016
df.loc[df["dd"]=="2016-08","year"]=2016
df.loc[df["dd"]=="2016-09","year"]=2016
df.loc[df["dd"]=="2016-10","year"]=2016
df.loc[df["dd"]=="2016-11","year"]=2016
df.loc[df["dd"]=="2016-12","year"]=2016
df.loc[df["dd"]=="2017-01","year"]=2017
df.loc[df["dd"]=="2017-02","year"]=2017
df.loc[df["dd"]=="2017-03","year"]=2017
df.loc[df["dd"]=="2017-04","year"]=2017
df.loc[df["dd"]=="2017-05","year"]=2017
df.loc[df["dd"]=="2017-06","year"]=2017
df.loc[df["dd"]=="2017-07","year"]=2017
df.loc[df["dd"]=="2017-08","year"]=2017
df.loc[df["dd"]=="2017-09","year"]=2017
df.loc[df["dd"]=="2017-10","year"]=2017
df.loc[df["dd"]=="2017-11","year"]=2017
df.loc[df["dd"]=="2017-12","year"]=2017
df.loc[df["dd"]=="2018-01","year"]=2018
df.loc[df["dd"]=="2018-02","year"]=2018
df.loc[df["dd"]=="2018-03","year"]=2018
df.loc[df["dd"]=="2018-04","year"]=2018
df.loc[df["dd"]=="2018-05","year"]=2018
df.loc[df["dd"]=="2018-06","year"]=2018
df.loc[df["dd"]=="2018-07","year"]=2018
df.loc[df["dd"]=="2018-08","year"]=2018
df.loc[df["dd"]=="2018-09","year"]=2018
df.loc[df["dd"]=="2018-10","year"]=2018
df.loc[df["dd"]=="2018-11","year"]=2018
df.loc[df["dd"]=="2018-12","year"]=2018
df.loc[df["dd"]=="2019-01","year"]=2019
df.loc[df["dd"]=="2019-02","year"]=2019
df.loc[df["dd"]=="2019-03","year"]=2019
df.loc[df["dd"]=="2019-04","year"]=2019
df.loc[df["dd"]=="2019-05","year"]=2019
df.loc[df["dd"]=="2019-06","year"]=2019
df.loc[df["dd"]=="2019-07","year"]=2019
df.loc[df["dd"]=="2019-08","year"]=2019
df.loc[df["dd"]=="2019-09","year"]=2019
df.loc[df["dd"]=="2019-10","year"]=2019
df.loc[df["dd"]=="2019-11","year"]=2019
df.loc[df["dd"]=="2019-12","year"]=2019
df.loc[df["dd"]=="2020-01","year"]=2020
df.loc[df["dd"]=="2020-02","year"]=2020
df.loc[df["dd"]=="2020-03","year"]=2020
df.loc[df["dd"]=="2020-04","year"]=2020
df.loc[df["dd"]=="2020-05","year"]=2020
df.loc[df["dd"]=="2020-06","year"]=2020
df.loc[df["dd"]=="2020-07","year"]=2020
df.loc[df["dd"]=="2020-08","year"]=2020
df.loc[df["dd"]=="2020-09","year"]=2020
df.loc[df["dd"]=="2020-10","year"]=2020
df.loc[df["dd"]=="2020-11","year"]=2020
df.loc[df["dd"]=="2020-12","year"]=2020
df.loc[df["dd"]=="2021-01","year"]=2021
df.loc[df["dd"]=="2021-02","year"]=2021
df.loc[df["dd"]=="2021-03","year"]=2021
df.loc[df["dd"]=="2021-04","year"]=2021
df.loc[df["dd"]=="2021-05","year"]=2021
df.loc[df["dd"]=="2021-06","year"]=2021
df.loc[df["dd"]=="2021-07","year"]=2021
df.loc[df["dd"]=="2021-08","year"]=2021
df.loc[df["dd"]=="2021-09","year"]=2021
df.loc[df["dd"]=="2021-10","year"]=2021
df.loc[df["dd"]=="2021-11","year"]=2021
df.loc[df["dd"]=="2021-12","year"]=2021
df.loc[df["dd"]=="2022-01","year"]=2022
df.loc[df["dd"]=="2022-02","year"]=2022
df.loc[df["dd"]=="2022-03","year"]=2022
df.loc[df["dd"]=="2022-04","year"]=2022
df.loc[df["dd"]=="2022-05","year"]=2022
df.loc[df["dd"]=="2022-06","year"]=2022
df.loc[df["dd"]=="2022-07","year"]=2022
df.loc[df["dd"]=="2022-08","year"]=2022
df.loc[df["dd"]=="2022-09","year"]=2022
df.loc[df["dd"]=="2022-10","year"]=2022
df.loc[df["dd"]=="2022-11","year"]=2022
df.loc[df["dd"]=="2022-12","year"]=2022
df.loc[df["dd"]=="2023-01","year"]=2023
df.loc[df["dd"]=="2023-02","year"]=2023
df.loc[df["dd"]=="2023-03","year"]=2023
df.loc[df["dd"]=="2023-04","year"]=2023
df.loc[df["dd"]=="2023-05","year"]=2023
df.loc[df["dd"]=="2023-06","year"]=2023
df.loc[df["dd"]=="2023-07","year"]=2023
df.loc[df["dd"]=="2023-08","year"]=2023
df.loc[df["dd"]=="2023-09","year"]=2023
df.loc[df["dd"]=="2023-10","year"]=2023
df.loc[df["dd"]=="2023-11","year"]=2023
df.loc[df["dd"]=="2023-12","year"]=2023
df["iso"]=356
df["country"]="India"

# Fill missing values in event columns with 0, those have zero events
df.isnull().any()
df=df.fillna(0)

# Add observations for gid-months which are completely missing

# Get all gid ids for india
prio_help = pd.read_csv("data/acled/prio_time_2014.csv")
prio_help_s=prio_help.loc[prio_help["gwno"]==750].reset_index(drop=True)
gid = list(prio_help_s.gid.unique())

# Make range of time stamps to add missing observations 
date = ['2016-01','2016-02','2016-03','2016-04','2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12',
        '2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12',
        '2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09','2018-10','2018-11','2018-12',
        '2019-01','2019-02','2019-03','2019-04','2019-05','2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12',
        '2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09','2020-10','2020-11','2020-12',
        '2021-01','2021-02','2021-03','2021-04','2021-05','2021-06','2021-07','2021-08','2021-09','2021-10','2021-11','2021-12',
        '2022-01','2022-02','2022-03','2022-04','2022-05','2022-06','2022-07','2022-08','2022-09','2022-10','2022-11','2022-12',
        '2023-01','2023-02','2023-03','2023-04','2023-05','2023-06','2023-07','2023-08','2023-09','2023-10','2023-11','2023-12']

# Loop through every gid-month
for i in range(0, len(gid)):
    for x in range(0, len(date)):
        
        # Check if gid-month in data, if False add
        if ((df['dd']==date[x]) 
            & (df['gid']==gid[i])).any()==False:
                
                # Subset data to add
                s = {'dd':date[x],'year':int(date[x][0:4]),'gid':gid[i],'country':"India",'iso':356,'n_protest_events':0,"n_remote_events":0,"n_riot_events":0,"n_stratdev_events":0,"n_battles_events":0,"n_civilvio_events":0}
                s = pd.DataFrame(data=s,index=[0])
                df = pd.concat([df,s])  

# Some events that were assigned to India in ACLED data,
# do not fall within India based on prio grid 
# these are not terrestrial grid cells based on the "Static Table"
miss = df[~df["gid"].isin(gid)]
miss = miss.sort_values(by=["gid","dd"])
print(miss.gid.unique())

# Compare some with the Prio grid
# these are grids close to the coast lines or borders with other countries. 
# It seems like these were incorrectly assigned to India, and should be removed.
# https://grid.prio.org/#/

# --> remove
df = df[df["gid"].isin(gid)] 

# Sort and checks
df = df.sort_values(by=["gid","dd"])
df=df.reset_index(drop=True)
print(df.isnull().any())
print(df.duplicated(subset=["dd","gid"]).any())
df.dtypes

#####################
### Add PRIO data ###
#####################

# landarea: total area covered by land in the grid cell in square kilometers 
# ttime_: travel time to the nearest major city
# mountain_mean:  proportion of mountainous terrain within the cell
# imr_: measures infant mortality rate
# cmr_: prevalence of child malnutrition
# petroleum_s: dummy variable for whether onshore petroleum deposits have been found within the given grid cell
# diamsec_s: dummy variable for whether secondary (alluvial) diamond deposits have been found within the given grid cell
# diamprim_s: dummy variable for whether primary (kimberlite) diamond deposits have been found within the given grid cell
# goldplacer_s: dummy variable for whether placer gold deposits have been found within the given grid cell
# goldsurface_s: dummy variable for whether surface gold deposits have been found within the given grid cell
# goldvein_s: dummy variable for whether vein gold deposits have been found within the given grid cell
# gem_s: dummy variable for whether gem deposits have been found within the given grid cell
# urban_gc: coverage of urban areas in each cell
# agri_gc: coverage of agricultural areas in each cell
# forest_gc: coverage of forest areas in each cell
# shrub_gc: coverage of shrubland in each cell
# herb_gc: coverage of herbaceous vegetation and lichens/mosses in each cel
# aquaveg_gc: coverage of aquatic vegetation in each cell
# barren_gc: coverage of barren areas in each cell
# water_gc: coverage of water areas in each cell
# maincrop: indicates the main crop code for the cell
# harvarea: sum of the harvested area (given in hectares) for the cell’s main crop 
# rainseas: initial month of the rainy season in the cell
# growstart: starting month of the growing season for the cell’s main crop
# growend: final month of the growing season for the cell’s main crop

prio_static = pd.read_csv('data/acled/prio_static.csv')
prio_static = prio_static.drop(columns=['row','col',"xcoord","ycoord",'cmr_mean','cmr_max','cmr_min','cmr_sd','growend','growstart','diamsec_s','diamprim_s','gem_s','goldplacer_s','goldvein_s','goldsurface_s','harvarea','imr_mean','mountains_mean','petroleum_s','imr_max','imr_min','imr_sd','maincrop','rainseas','ttime_min',"ttime_max",'ttime_sd'])

# Merge 
df=pd.merge(df,prio_static, how='left', on='gid')

# Save 
print(df.isnull().any())
counts = df.groupby("gid").size()
print(counts.nunique()==1)
print(df.duplicated(subset=["dd","gid"]).any())
df.to_csv("data/acled/acled_grid_India_2023.csv",index=False, sep=',')
df.dtypes
