import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from functions import ARIMA_opti_pred,DARIMA_opti_pred,general_model,general_dynamic_model
import json

# Load data
acled = pd.read_csv("data/acled/acled_grid_India_2023.csv")

#######################################
### Get static models for all cases ###
#######################################

# Initate
final_arima=pd.DataFrame()
final_rf=pd.DataFrame()
countries = acled["gid"].unique()
norm=True
for c in countries:                 
    print(c)
    ts=acled["n_protest_events"].loc[acled["gid"]==c]
    X=acled[['n_riot_events','n_battles_events',"ttime_mean","urban_gc"]].loc[acled["gid"]==c]
    
    # Return 0 if training data is flat
    if ts[:int(0.7*len(ts))].max()==0:
        print("flat")
        
        if norm==True:
            min_val = np.min(ts)
            max_val = np.max(ts)
            ts_norm = (ts - min_val) / (max_val - min_val)
            ts_norm=ts_norm.fillna(0) 
            
        data_lin = {'dd': list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):]),
            'country': [c] * len(list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):])),
            'n_protest_events': list(ts_norm[int(0.7*len(ts)):]),
            'preds_arima': [0] * len(list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):])),
            'preds_arimax': [0] * len(list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):]))}
        preds = pd.DataFrame(data_lin)
        final_arima = pd.concat([final_arima, preds])
        final_arima=final_arima.reset_index(drop=True)
        final_arima.to_csv("data/predictions/linear_static.csv")
        
        if norm==True:
            y_train = ts[:int(0.7*len(ts))]
            mini = np.min(y_train)
            maxi = np.max(y_train)
            y_train = (y_train - mini) / (maxi - mini)
            y_train=y_train.fillna(0) 
            
            y_test = ts[int(0.7*len(ts)):]       
            mini = np.min(y_test)
            maxi = np.max(y_test)
            y_test = (y_test - mini) / (maxi - mini)
            y_test=y_test.fillna(0) 
            ts_norm=pd.concat([y_train,y_test]) 
                    
        data_rf = {'dd': list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):]),
            'country': [c] * len(list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):])),
            'n_protest_events': list(ts_norm[int(0.7*len(ts)):]),
            'preds_rf': [0] * len(list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):])),
            'preds_rfx': [0] * len(list(acled["dd"].loc[acled["gid"]==c][int(0.7*len(ts)):])),}
        preds = pd.DataFrame(data_rf)
        final_rf = pd.concat([final_rf, preds])
        final_rf=final_rf.reset_index(drop=True)
        final_rf.to_csv("data/predictions/nonlinear_static.csv") 
       
    # Fit static models
    else:

        #####################
        ### Linear models ###
        #####################
            
        # ARIMA opti
        arima = ARIMA_opti_pred(ts,norm=True)
        preds = pd.DataFrame(acled["dd"].loc[acled["gid"]==c][-len(arima["actuals"]):])
        preds.columns = ["dd"]  
        preds["country"] = c
        preds["n_protest_events"] = list(arima["actuals"])
        preds["preds_arima"] = list(arima["arima_pred"])
            
        # ARIMAX
        arimax = ARIMA_opti_pred(ts,X=X,norm=True)
        preds["preds_arimax"] = list(arimax["arima_pred"])
        final_arima = pd.concat([final_arima, preds])
        final_arima=final_arima.reset_index(drop=True)
        final_arima.to_csv("data/predictions/linear_static.csv")
            
        #########################
        ### Non-linear models ###
        #########################
        
        # RF
        rf = general_model(ts,norm=True,opti=True) 
        preds = pd.DataFrame(acled["dd"].loc[acled["gid"]==c][-len(rf["actuals"]):])
        preds.columns = ["dd"]  
        preds["country"] = c
        preds["n_protest_events"] = list(rf["actuals"])
        preds["preds_rf"] = list(rf["rf_pred"])
    
        # RFX
        rfx = general_model(ts,X=X,norm=True,opti=True) 
        preds["preds_rfx"] = list(rfx["rf_pred"])
        final_rf = pd.concat([final_rf, preds])
        final_rf=final_rf.reset_index(drop=True)
        final_rf.to_csv("data/predictions/nonlinear_static.csv")  

print(mean_squared_error(final_arima.n_protest_events, final_arima.preds_arima,sample_weight=final_arima.n_protest_events))
print(mean_squared_error(final_arima.n_protest_events, final_arima.preds_arimax,sample_weight=final_arima.n_protest_events))
print(mean_squared_error(final_rf.n_protest_events, final_rf.preds_rf,sample_weight=final_rf.n_protest_events))
print(mean_squared_error(final_rf.n_protest_events, final_rf.preds_rfx,sample_weight=final_rf.n_protest_events))

######################
### Dynamic models ###
######################

thres=0.5
final_darima=pd.DataFrame()
final_drf=pd.DataFrame()
shapes_arima={}
shapes_rf={}

# Find countries with high intensity
df_n_country_month = {}
for i in acled["gid"].unique():
    ts = acled["n_protest_events"].loc[acled["gid"]==i][:int(0.7*len(acled["n_protest_events"].loc[acled["gid"]==i]))]
    df_n_country_month[i] = np.mean(np.abs(ts - np.mean(ts)))
df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True) 
country_keep = df_n_country_month.loc[df_n_country_month["avg"]>thres].gid.unique()
countries = acled["gid"].unique()    

print(f"Out of {len(countries)}, {len(country_keep)} are dynamic")
    
# If high intensity, use dynamic model   
for c in country_keep:
    print(c)
    ts=acled["n_protest_events"].loc[acled["gid"]==c]
    X=acled[['n_riot_events','n_battles_events',"ttime_mean","urban_gc"]].loc[acled["gid"]==c]
        
    #####################
    ### Linear models ###
    #####################

    # DARIMA
    darima = DARIMA_opti_pred(ts,norm=True)
    preds = pd.DataFrame(acled["dd"].loc[acled["gid"]==c][-len(darima["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["n_protest_events"] = list(darima["actuals"])
    preds["preds_darima"] = list(darima["darima_pred"])
    shapes_arima.update({f"darima_{c}":[darima["s"],darima["shapes"].tolist(),darima["clusters"].tolist()]})
    
    # DARIMAX
    darimax = DARIMA_opti_pred(ts,X=X,norm=True)  
    preds["preds_darimax"] = list(darimax["darima_pred"])
    final_darima = pd.concat([final_darima, preds])
    final_darima=final_darima.reset_index(drop=True)
    shapes_arima.update({f"darimax_{c}":[darimax["s"],darimax["shapes"].tolist(),darimax["clusters"].tolist()]})
    final_darima.to_csv(f"data/predictions/linear_dynamic_thres{thres}.csv")
               
    #########################
    ### Non-linear models ###
    #########################
    
    # DRF
    drf = general_dynamic_model(ts,norm=True,opti=True) 
    preds = pd.DataFrame(acled["dd"].loc[acled["gid"]==c][-len(drf["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["n_protest_events"] = list(drf["actuals"])
    preds["preds_drf"] = list(drf["drf_pred"])
    shapes_rf.update({f"drf_{c}":[drf["s"],drf["shapes"].tolist(),drf["clusters"].tolist()]})
    
    # DRFX
    drfx = general_dynamic_model(ts,X=X,norm=True,opti=True)
    preds["preds_drfx"] = list(drfx["drf_pred"])
    final_drf = pd.concat([final_drf, preds])
    final_drf=final_drf.reset_index(drop=True)
    shapes_rf.update({f"drfx_{c}":[drfx["s"],drfx["shapes"].tolist(),drfx["clusters"].tolist()]})
    final_drf.to_csv(f"data/predictions/nonlinear_dynamic_thres{thres}.csv")  

# Save shapes
with open(f'data/predictions/arima_shapes_thres{thres}.json', 'w') as json_file:
    json.dump(shapes_arima, json_file)
with open(f'data/predictions/rf_shapes_thres{thres}.json', 'w') as json_file:
    json.dump(shapes_rf, json_file)

print(mean_squared_error(final_darima.n_protest_events, final_darima.preds_darima,sample_weight=final_darima.n_protest_events))
print(mean_squared_error(final_darima.n_protest_events, final_darima.preds_darimax,sample_weight=final_darima.n_protest_events))
print(mean_squared_error(final_drf.n_protest_events, final_drf.preds_drf,sample_weight=final_drf.n_protest_events))
print(mean_squared_error(final_drf.n_protest_events, final_drf.preds_drfx,sample_weight=final_drf.n_protest_events))

####################           
### Merge cases ####
####################           
           
# A. linear models

# Static models
final_arima_static = pd.read_csv("data/predictions/linear_static.csv",index_col=0)

# Find countries with high intensity
thres=0.5
df_n_country_month = {}
for i in acled["gid"].unique():
    ts = acled["n_protest_events"].loc[acled["gid"]==i][:int(0.7*len(acled["n_protest_events"].loc[acled["gid"]==i]))]
    df_n_country_month[i] = np.mean(np.abs(ts - np.mean(ts)))
df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True) 
country_keep = df_n_country_month.loc[df_n_country_month["avg"]>thres].gid.unique()

# Dynamic models
final_arima_dynamic = pd.read_csv("data/predictions/linear_dynamic_thres0.5.csv",index_col=0)
final_arima_dynamic = final_arima_dynamic[final_arima_dynamic['country'].isin(country_keep)]

# Merge
df_linear=pd.merge(final_arima_static,final_arima_dynamic[["dd","country",'preds_darima','preds_darimax']],on=["dd","country"],how="left")
df_linear=df_linear.sort_values(by=["country","dd"])
df_linear=df_linear.reset_index(drop=True)
df_linear['preds_darima'] = df_linear['preds_darima'].fillna(df_linear['preds_arima'])
df_linear['preds_darimax'] = df_linear['preds_darimax'].fillna(df_linear['preds_arimax'])
df_linear.to_csv("data/predictions/df_linear.csv")  

print(mean_squared_error(df_linear.n_protest_events, df_linear.preds_arima,sample_weight=df_linear.n_protest_events))
print(mean_squared_error(df_linear.n_protest_events, df_linear.preds_arimax,sample_weight=df_linear.n_protest_events))
print(mean_squared_error(df_linear.n_protest_events, df_linear.preds_darima,sample_weight=df_linear.n_protest_events))
print(mean_squared_error(df_linear.n_protest_events, df_linear.preds_darimax,sample_weight=df_linear.n_protest_events))

# A. Nonlinear models

# Static models
final_rf_static = pd.read_csv("data/predictions/nonlinear_static.csv",index_col=0)

# Find countries with high intensity
thres=0.5
df_n_country_month = {}
for i in acled["gid"].unique():
    ts = acled["n_protest_events"].loc[acled["gid"]==i][:int(0.7*len(acled["n_protest_events"].loc[acled["gid"]==i]))]
    df_n_country_month[i] = np.mean(np.abs(ts - np.mean(ts)))
df_n_country_month = pd.DataFrame.from_dict(df_n_country_month, orient="index").reset_index()
df_n_country_month.rename(columns = {'index':'gid', 0:'avg'}, inplace = True) 
country_keep = df_n_country_month.loc[df_n_country_month["avg"]>thres].gid.unique()

# Dynamic models
final_rf_dynamic = pd.read_csv("data/predictions/nonlinear_dynamic_thres0.5.csv",index_col=0)
final_rf_dynamic = final_rf_dynamic[final_rf_dynamic['country'].isin(country_keep)]

# Merge
df_nonlinear=pd.merge(final_rf_static,final_rf_dynamic[["dd","country",'preds_drf','preds_drfx']],on=["dd","country"],how="left")
df_nonlinear=df_nonlinear.sort_values(by=["country","dd"])
df_nonlinear=df_nonlinear.reset_index(drop=True)
df_nonlinear['preds_drf'] = df_nonlinear['preds_drf'].fillna(df_nonlinear['preds_rf'])
df_nonlinear['preds_drfx'] = df_nonlinear['preds_drfx'].fillna(df_nonlinear['preds_rfx'])
df_nonlinear.to_csv("data/predictions/df_nonlinear.csv")  

print(mean_squared_error(df_nonlinear.n_protest_events, df_nonlinear.preds_rf,sample_weight=df_nonlinear.n_protest_events))
print(mean_squared_error(df_nonlinear.n_protest_events, df_nonlinear.preds_rfx,sample_weight=df_nonlinear.n_protest_events))
print(mean_squared_error(df_nonlinear.n_protest_events, df_nonlinear.preds_drf,sample_weight=df_nonlinear.n_protest_events))
print(mean_squared_error(df_nonlinear.n_protest_events, df_nonlinear.preds_drfx,sample_weight=df_nonlinear.n_protest_events))


