import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans,silhouette_score
from pmdarima.arima import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import PredefinedSplit,GridSearchCV


def ARIMA_opti_pred(y: pd.Series, 
               X: str=None, 
               norm: bool=False):
           
    if norm==True:
        min_val = np.min(y)
        max_val = np.max(y)
        y = (y - min_val) / (max_val - min_val)
        y=y.fillna(0) 
    
    # Test data
    ex_test=y.iloc[int(0.7*len(y)):]  
            
    # Train and test data for x
    if X is not None:
        x_train=X.iloc[:int(0.7*len(y)),:]
        x_test=X.iloc[int(0.7*len(y)):,:]
            
    if X is None:
        # Fit model and make predictions
        preds_final=[]   
        for i in range(len(ex_test)):
            arima_test=ARIMA(order=(1,1,0),with_intercept=False).fit(y.iloc[:int(0.7*len(y))+i])
            preds_final.append(arima_test.arima_res_.get_forecast(1).predicted_mean.iloc[0])
                                                                  
    elif X is not None:
        # Fit model and make predictions
        preds_final=[]
        history=x_train
        for i in range(len(ex_test)):
            model_test= ARIMA(order=(1,1,0),with_intercept=False).fit(y.iloc[:int(0.7*len(y))+i],history)
            exog = x_test.iloc[i:i+1,:]
            preds_final.append(model_test.arima_res_.get_forecast(1, exog=exog).predicted_mean.iloc[0])
            history=np.concatenate([history,exog]) 
                              
    return({'arima_pred':pd.Series(preds_final),
            'actuals':ex_test.reset_index(drop=True)})
    


def DARIMA_opti_pred(y: pd.Series, 
                X: str=None,
                model=TimeSeriesKMeans(n_clusters=5,metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0),
                norm: bool=False):
    
    # Copy ts for clusterin
    y_clu=y.copy()
    
    if norm==True:
        min_val = np.min(y)
        max_val = np.max(y)
        y = (y - min_val) / (max_val - min_val)
        y=y.fillna(0) 
   
    ###################
    ### Clustering  ###
    ###################

    min_test=np.inf
    
    # For numer of clusters in test_clus
    for n_clu in [3,5,7]:
        # For window length in test_win
        for number_s in [3,5,7,9]:
            model.n_clusters=n_clu
                                 
            # Training data
            ex=y_clu.iloc[:int(0.7*len(y_clu))]
                    
            # Test data
            ex_test=y_clu.iloc[int(0.7*len(y_clu)):]
                            
            ### Training data ###
            ts_seq=[]
                    
            # Get input
            for i in range(number_s,len(ex)):
                ts_seq.append(y_clu.iloc[i-number_s:i])  
            ts_seq=np.array(ts_seq)
                    
            # Sacling
            ts_seq_l= pd.DataFrame(ts_seq).T
            ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
            ts_seq_l=ts_seq_l.fillna(0)
            ts_seq_l=np.array(ts_seq_l.T)
                            
            # Clustering 
            ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
            model.n_clusters=n_clu
            m_dba = model.fit(ts_seq_l)
            cl= m_dba.labels_
            cl=pd.Series(cl)
            cl=pd.get_dummies(cl).astype(int)
                        
            # Make sure that length of dummy set is equal to n_clu
            # If not, add empty column 
            cl_b=pd.DataFrame(columns=range(n_clu))
            cl=pd.concat([cl_b,cl],axis=0)   
            cl=cl.fillna(0)
            
            ### Test data ###
            ts_seq=[]
            
            # Get inout
            for i in range(len(ex),len(y_clu)):
                ts_seq.append(y_clu.iloc[i-number_s:i])    
            ts_seq=np.array(ts_seq)
            
            # Sacling
            ts_seq_l= pd.DataFrame(ts_seq).T
            ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
            ts_seq_l=ts_seq_l.fillna(0) 
            ts_seq_l=np.array(ts_seq_l.T)
                    
            # Use trained model to predict clusters in test data
            ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
            y_test = m_dba.predict(ts_seq_l)
            y_test_seq = m_dba.predict(ts_seq_l)
            y_test=pd.Series(y_test)
            y_test=pd.get_dummies(y_test).astype(int)
                
            # Make sure that length of dummy set is equal to n_clu
            # If not, add empty column 
            y_t=pd.DataFrame(columns=range(n_clu))
            y_test=pd.concat([y_t,y_test],axis=0)   
            y_test=y_test.fillna(0)  
                        
            ###################
            ### Predictions ###
            ###################
                    
            # Training data
            ex=y.iloc[:int(0.7*len(y))]
            ex_test=y.iloc[int(0.7*len(y)):]
                

            if X is None: 
                
                # Fit model and make predictions
                pred=[]
                cl_c=cl.copy()
                for i in range(len(ex_test)):
                    darima_test=ARIMA(order=(1,1,0),with_intercept=False).fit(y.iloc[number_s:int(0.7*len(y))+i],np.array(cl_c))
                    exog = y_test.iloc[i:i+1,:]
                    pred.append(darima_test.arima_res_.get_forecast(1, exog=exog).predicted_mean.iloc[0])
                    cl_c=np.concatenate([cl_c,exog]) 

                ar_met=mean_squared_error(ex_test,pred,sample_weight=ex_test)
                                                        
                # If ar_met is smaller than min_mae
                if ar_met < min_test:
                    min_test=ar_met
                    preds_final=pred
                    para=[n_clu,number_s]
                    shapes=m_dba.cluster_centers_
                    seq_clusters=y_test_seq
                    if np.unique(y_test_seq).size==1:
                        s=np.nan
                    else:
                        s=silhouette_score(ts_seq_l, y_test_seq, metric="dtw")  
            
            elif X is not None:
                x_train=X.iloc[:int(0.7*len(y)),:]
                x_test=X.iloc[int(0.7*len(y)):,:]
                                            
                # Merge clusters and additional X-covariate
                exo_in=np.concatenate([x_train[number_s:],np.array(cl)],axis=1)
                exo_test=np.concatenate([x_test,np.array(y_test)],axis=1)
                                                                                                                                
                # Fit model and make predictions
                pred=[]
                history=exo_in.copy()
                for i in range(len(ex_test)):
                    model_test=ARIMA(order=(1,1,0),with_intercept=False).fit(y.iloc[number_s:int(0.7*len(y))+i],history)         
                    exog = exo_test[i:i+1,:]
                    pred.append(model_test.arima_res_.get_forecast(1, exog=exog).predicted_mean.iloc[0])
                    history=np.concatenate([history,exog])
                                                                           
                ar_met=mean_squared_error(ex_test,pred,sample_weight=ex_test)
                                
                # If ar_met is smaller than min_mae
                if ar_met < min_test:
                    min_test=ar_met
                    preds_final=pred
                    para=[n_clu,number_s]
                    shapes=m_dba.cluster_centers_
                    seq_clusters=y_test_seq
                    if np.unique(y_test_seq).size==1:
                        s=np.nan
                    else:
                        s=silhouette_score(ts_seq_l, y_test_seq, metric="dtw")                                         
                   
    print(f"Final DARIMA {para}: {min_test}")
    

    return({'darima_pred':pd.Series(preds_final),
            'actuals':ex_test.reset_index(drop=True),
            "shapes":shapes,
            "s":s,
            "clusters":seq_clusters})

    
  
def general_model(y,
                  X: bool = None,
                  norm: bool=False,
                  opti: bool=False):
    
    # Normalize 
    if norm==True:
         y_train = y[:int(0.7*len(y))]
         mini = np.min(y_train)
         maxi = np.max(y_train)
         y_train = (y_train - mini) / (maxi - mini)
         y_train=y_train.fillna(0) 
         
         y_test = y[int(0.7*len(y)):]       
         mini = np.min(y_test)
         maxi = np.max(y_test)
         y_test = (y_test - mini) / (maxi - mini)
         y_test=y_test.fillna(0) 
     
         y=pd.concat([y_train,y_test]) 
                
         
    min_test=np.inf
    for ar in [2,3,4,6]:
        
        def lags(series):
            last = series.iloc[-ar:].fillna(0)
            return last.tolist() + [0] * (ar - len(last))
            
        data_matrix = []
        for i in range(ar, len(y) + 1):
            data_matrix.append(lags(y.iloc[:i]))
                
        # Columns names
        cols_name=[]
        for i in range(ar):
            cols_name.append(f"t-{i}")  
        cols_name=cols_name[::-1]
        data=pd.DataFrame(data_matrix,columns=cols_name)
            
        # Get training and test data
        output=data.iloc[:, -1]
        in_put=data.iloc[:, :-1]
            
        if X is not None:
            X=X.iloc[-len(in_put):].reset_index(drop=True)
            in_put=pd.concat([X, in_put],axis=1)
                
        y_train = output[:-(len(y)-int(0.7*len(y)))]
        x_train = in_put[:-(len(y)-int(0.7*len(y)))]
        
        y_test = output[-(len(y)-int(0.7*len(y))):]        
        x_test = in_put[-(len(y)-int(0.7*len(y))):]    
        
        if opti==True: 
        
            # Validation
            val_train_index = list(y_train[:int(0.5*len(y_train))].index)
            val_test_index = list(y_train[int(0.5*len(y_train)):].index)
        
            splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
            ps = PredefinedSplit(test_fold=splits)
            
            opti_grid = {'n_estimators': [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000]}
            
            grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0), param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
            grid_search.fit(x_train, y_train)
            best_params = grid_search.best_params_
            model_fit = RandomForestRegressor(**best_params,random_state=0)
            model_fit.fit(x_train, y_train)
            pred = model_fit.predict(x_test)
        
        else: 
            model_fit = RandomForestRegressor(random_state=0)
            model_fit.fit(x_train, y_train)
            pred = model_fit.predict(x_test)
            
        if y_test.max()==0:
               error=((y_test.reset_index(drop=True)-pd.Series(pred))**2).mean()
        
        else: 
            error=mean_squared_error(y_test,pred,sample_weight=y_test)

        if error<min_test:
            min_test=error
            para=[ar]
            preds_final=pred
        
    print(f"Final RF {para}: {min_test}")
    
    return({'rf_pred':pd.Series(preds_final),
            'actuals':y_test.reset_index(drop=True)})
            
        
def general_dynamic_model(y,
                    X: bool = None,
                    norm: bool=False,
                    model=TimeSeriesKMeans(n_clusters=5,metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0),
                    opti:bool=False):
    
    y_clu=y
    
    # Normalize 
    if norm==True:
         y_train = y[:int(0.7*len(y))]
         mini = np.min(y_train)
         maxi = np.max(y_train)
         y_train = (y_train - mini) / (maxi - mini)
         y_train=y_train.fillna(0) 
         
         y_test = y[int(0.7*len(y)):]       
         mini = np.min(y_test)
         maxi = np.max(y_test)
         y_test = (y_test - mini) / (maxi - mini)
         y_test=y_test.fillna(0) 
     
         y=pd.concat([y_train,y_test]) 
    
    ##################
    ### Clustering ###
    ##################

    # Initiate test metric
    min_test=np.inf
    # For numer of clusters in test_clus
    for n_clu in [3,5,7]:
        # For window length in test_win
        for number_s in [3,5,7,9]:
            # Update number of clusters in model 
            model.n_clusters=n_clu
        
            # Training data
            ex=y_clu.iloc[:int(0.7*len(y_clu))]
            
            ### Training data ###
            ts_seq=[]
            
            # Make list of lists, 
            # each sub-list contains number_s observations
            for i in range(number_s,len(ex)):
                ts_seq.append(y_clu.iloc[i-number_s:i])
                
            # Convert into array,
            # each row is a time series of number_s observations     
            ts_seq=np.array(ts_seq)
            
            # Scaling
            ts_seq_l= pd.DataFrame(ts_seq).T
            ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
            ts_seq_l=ts_seq_l.fillna(0) # if seq uniform 
            ts_seq_l=np.array(ts_seq_l.T)
                        
            # Reshape array,
            # each sub array contains times series of number_s observations
            ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
            
            # Clustering and convert into dummy set
            model.n_clusters=n_clu
            m_dba = model.fit(ts_seq_l)
            cl= m_dba.labels_
            cl=pd.Series(cl)
            cl=pd.get_dummies(cl).astype(int)
            
            # Make sure that length of dummy set is equal to n_clu
            # If not, add empty column 
            cl_b=pd.DataFrame(columns=range(n_clu))
            cl=pd.concat([cl_b,cl],axis=0)   
            cl=cl.fillna(0)
        
            ### Test data ###
            ts_seq=[]
        
            # Make list of lists, 
            # each sub-list contains number_s observations
            for i in range(len(ex),len(y_clu)):
                ts_seq.append(y_clu.iloc[i-number_s:i])
                
            # Convert into array,
            # each row is a time series of number_s observations       
            ts_seq=np.array(ts_seq)
            
            # Sacling
            ts_seq_l= pd.DataFrame(ts_seq).T
            ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
            ts_seq_l=ts_seq_l.fillna(0) # if seq uniform 
            ts_seq_l=np.array(ts_seq_l.T)
                    
            # Reshape array,
            # each sub array contains times series of number_s observations
            ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
            
            # Use trained model to predict clusters in test data
            # and convert into dummy set
            y_test = m_dba.predict(ts_seq_l)
            y_test_seq = m_dba.predict(ts_seq_l)
            y_test=pd.Series(y_test)
            y_test=pd.get_dummies(y_test).astype(int)
            
            # Make sure that length of dummy set is equal to n_clu
            # If not, add empty column 
            y_t=pd.DataFrame(columns=range(n_clu))
            y_test=pd.concat([y_t,y_test],axis=0)   
            y_test=y_test.fillna(0)  
            
            clusters=pd.concat([cl,y_test],axis=0,ignore_index=True)
            index=list(range(len(y)-len(clusters), len(y)))
            clusters.set_index(pd.Index(index),inplace=True)
            
            ###################
            ### Predictions ###
            ###################
            
            for ar in [2,3,4,6]:
                def lags(series):
                    last = series.iloc[-ar:].fillna(0)
                    return last.tolist() + [0] * (ar - len(last))
                
                data_matrix = []
                for i in range(ar, len(y) + 1):
                    data_matrix.append(lags(y.iloc[:i]))
                    
                # Columns names
                cols_name=[]
                for i in range(ar):
                    cols_name.append(f"t-{i}")  
                cols_name=cols_name[::-1]
                data=pd.DataFrame(data_matrix,columns=cols_name)
                
                # Get training and test data
                in_put=pd.DataFrame(data.iloc[:, :-1])
                
                index=list(range(len(y)-len(in_put), len(y)))
                in_put.set_index(pd.Index(index),inplace=True)
                    
                if len(clusters)>=len(in_put):
                    in_put=pd.concat([clusters,in_put],axis=1)
                else: 
                    in_put=pd.concat([in_put,clusters],axis=1)
                    
                in_put=in_put.fillna(0)
                    
                if X is not None:
                    X=X.reset_index(drop=True)
                    in_put=pd.concat([X,in_put],axis=1)
                    in_put = in_put.dropna()
                
                in_put.columns = in_put.columns.map(str)
                
                output=y.reset_index(drop=True)
                output=output[-len(in_put):]
    
                y_train = output[:-(len(y)-int(0.7*len(y)))]
                x_train = in_put[:-(len(y)-int(0.7*len(y)))]
    
                y_test = output[-(len(y)-int(0.7*len(y))):]        
                x_test = in_put[-(len(y)-int(0.7*len(y))):] 
                
                if opti==True: 
                
                    # Validation
                    val_train_index = list(y_train[:int(0.5*len(y_train))].index)
                    val_test_index = list(y_train[int(0.5*len(y_train)):].index)
                
                    splits = np.array([-1] * len(val_train_index) + [0] * len(val_test_index))
                    ps = PredefinedSplit(test_fold=splits)
                    
                    opti_grid = {'n_estimators': [10, 231, 452, 673, 894, 1115, 1336, 1557, 1778, 2000]}
                    
                    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=0), param_grid=opti_grid, cv=ps, verbose=0, n_jobs=-1)
                    grid_search.fit(x_train, y_train)
                    best_params = grid_search.best_params_
                    model_fit = RandomForestRegressor(**best_params,random_state=0)
                    model_fit.fit(x_train, y_train)
                    pred = model_fit.predict(x_test)
                
                else: 
                    model_fit = RandomForestRegressor(random_state=0)
                    model_fit.fit(x_train, y_train)
                    pred = model_fit.predict(x_test)
                        
                error=mean_squared_error(y_test,pred,sample_weight=y_test)
                                    
                # If ar_met smaller than min_test
                if error<min_test:
                    min_test=error
                    para=[ar,n_clu,number_s]
                    preds_final=pred
                    shapes=m_dba.cluster_centers_
                    seq_clusters=y_test_seq
                    if np.unique(y_test_seq).size==1:
                        s=np.nan
                    else: 
                        s=silhouette_score(ts_seq_l, y_test_seq, metric="dtw") 
    
    print(f"Final DRF {para}: {min_test}")
    
    return({'drf_pred':pd.Series(preds_final),
            'actuals':y_test.reset_index(drop=True),
            "shapes":shapes,
            "s":s,
            "clusters":seq_clusters})



