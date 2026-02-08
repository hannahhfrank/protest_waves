import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from functions import DARIMA_opti_pred,general_dynamic_model
import json
from dtaidistance import dtw
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from tslearn.clustering import silhouette_score
from tslearn.barycenters import dtw_barycenter_averaging
import matplotlib.pyplot as plt
import random
import os 
import matplotlib as mpl
os.environ['PATH'] = "/Library/TeX/texbin:" + os.environ.get('PATH', '')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']
mpl.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}\usepackage[T1]{fontenc}'

# Load data
acled = pd.read_csv("data/acled/acled_grid_India_2023.csv")

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
    rng = np.random.default_rng(1)
    lambda_g = ts.mean()
    vals = rng.poisson(lam=lambda_g, size=len(ts))
    ts_random=pd.Series(vals, index=ts.index, name=ts.name)
    
    #if c==145234:
    #    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(9, 10))
    #    axes[0,0].plot(ts[:24].index, ts[:24].values,c="black",linewidth=2)
    #    axes[0,0].set_title("Original time series",size=20)
    #    axes[0,0].xaxis.set_visible(False)      
    #    axes[0,0].set_ylim(bottom=-0.05)
    #    axes[0,0].tick_params(axis='both', which='major', labelsize=15)
    #    axes[0,1].plot(ts_random[:24].index, ts_random[:24].values,color="gray",linewidth=2)
    #    axes[0,1].set_title("Poisson time series",size=20)
    #    axes[0,1].xaxis.set_visible(False)        
    #    axes[0,1].set_ylim(bottom=-0.05)
    #    axes[0,1].tick_params(axis='both', which='major', labelsize=15)
        
       
    #if c==165428:
    #    axes[1,0].plot(ts[24:48].index, ts[24:48].values,c="black",linewidth=2)
    #    axes[1,0].xaxis.set_visible(False)      
    #    axes[1,0].set_ylim(bottom=-0.05)
    #    axes[1,0].tick_params(axis='both', which='major', labelsize=15)
    #    axes[1,1].plot(ts_random[24:48].index, ts_random[24:48].values,color="gray",linewidth=2)
    #    axes[1,1].xaxis.set_visible(False)        
    #    axes[1,1].set_ylim(bottom=-0.05)
    #    axes[1,1].tick_params(axis='both', which='major', labelsize=15)

    #if c==165428:
    #    axes[2,0].plot(ts[:24].index, ts[:24].values,c="black",linewidth=2)
    #    axes[2,0].xaxis.set_visible(False)      
    #    axes[2,0].set_ylim(bottom=-0.05)
    #    axes[2,0].tick_params(axis='both', which='major', labelsize=15)
    #    axes[2,1].plot(ts_random[:24].index, ts_random[:24].values,color="gray",linewidth=2)
    #    axes[2,1].xaxis.set_visible(False)        
    #    axes[2,1].set_ylim(bottom=-0.05)
    #    axes[2,1].tick_params(axis='both', which='major', labelsize=15)  
        
    #if c==174030:
    #    axes[3,0].plot(ts[:24].index, ts[:24].values,c="black",linewidth=2)
    #    axes[3,0].xaxis.set_visible(False)      
    #    axes[3,0].set_ylim(bottom=-0.05)
    #    axes[3,0].tick_params(axis='both', which='major', labelsize=15)
    #    axes[3,1].plot(ts_random[:24].index, ts_random[:24].values,color="gray",linewidth=2)
    #    axes[3,1].xaxis.set_visible(False)        
    #    axes[3,1].set_ylim(bottom=-0.05)
    #    axes[3,1].tick_params(axis='both', which='major', labelsize=15)  
        
    #    plt.tight_layout()
    #    fig.subplots_adjust(hspace=0.05)
    #    plt.savefig("results/ts_random_example.eps",dpi=400,bbox_inches="tight")
    #    plt.show()

    #####################
    ### Linear models ###
    #####################

    # DARIMA
    darima = DARIMA_opti_pred(ts_random,norm=True)
    preds = pd.DataFrame(acled["dd"].loc[acled["gid"]==c][-len(darima["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["n_protest_events"] = list(darima["actuals"])
    preds["preds_darima"] = list(darima["darima_pred"])
    final_darima = pd.concat([final_darima, preds])
    final_darima=final_darima.reset_index(drop=True)
    shapes_arima.update({f"darima_{c}":[darima["s"],darima["shapes"].tolist()]})
    final_darima.to_csv(f"data/predictions/linear_dynamic_thres{thres}_random.csv")
               
    #########################
    ### Non-linear models ###
    #########################
    
    # DRF
    drf = general_dynamic_model(ts_random,norm=True,opti=True) 
    preds = pd.DataFrame(acled["dd"].loc[acled["gid"]==c][-len(drf["actuals"]):])
    preds.columns = ["dd"]  
    preds["country"] = c
    preds["n_protest_events"] = list(drf["actuals"])
    preds["preds_drf"] = list(drf["drf_pred"])
    final_drf = pd.concat([final_drf, preds])
    final_drf=final_drf.reset_index(drop=True)
    shapes_rf.update({f"drf_{c}":[drf["s"],drf["shapes"].tolist()]})
    final_drf.to_csv(f"data/predictions/nonlinear_dynamic_thres{thres}_random.csv")  

# Save shapes
with open(f'data/predictions/arima_shapes_thres{thres}_random.json', 'w') as json_file:
    json.dump(shapes_arima, json_file)
with open(f'data/predictions/rf_shapes_thres{thres}_random.json', 'w') as json_file:
    json.dump(shapes_rf, json_file)

print(mean_squared_error(final_darima.n_protest_events, final_darima.preds_darima,sample_weight=final_darima.n_protest_events))
print(mean_squared_error(final_drf.n_protest_events, final_drf.preds_drf,sample_weight=final_drf.n_protest_events))


###################################
### Clustering ot the centroids ###
###################################
  
with open("data/predictions/rf_shapes_thres0.5_random.json", 'r') as json_file:
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
        plt.savefig("results/clusters_India_rf_random.eps",dpi=400,bbox_inches="tight")
        plt.show()
        
        
        n_clusters = len(representatives)
        n_member_plots = 16 
        member_cols = 8    
        
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
        plt.savefig("results/dendogram_India_rf_random.eps", dpi=400, bbox_inches="tight")
        plt.show()
 
        
        
with open("data/predictions/arima_shapes_thres0.5_random.json", 'r') as json_file:
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
        cols = 4
        rows = n_clusters // cols + (n_clusters % cols > 0)
            
        plt.figure(figsize=(17, 4 * rows))
        for i, seq, n in zip(range(n_clusters),representatives, n_obs[1]):
            plt.subplot(rows, cols, i+1)
            plt.plot(seq,linewidth=3,color="black")
            plt.title(f'Cluster {i+1}, n={n}',size=41)
            plt.ylim(-0.05,1.1)
            plt.yticks([],[])
            plt.xticks([],[])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.01)
        plt.savefig("results/clusters_India_arima_random.eps",dpi=400,bbox_inches="tight")
        plt.show()
        
        
        n_clusters = len(representatives)
        n_member_plots = 16 
        member_cols = 8    
        
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
        plt.savefig("results/dendogram_India_arima_random.eps", dpi=400, bbox_inches="tight")
        plt.show()
        
        
        
        
        
        
        
        
        
        
