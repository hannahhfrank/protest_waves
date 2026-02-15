This repository contains the replication material for "The Dynamics of Dissent: Patterns in Protest Cycles" (Hannah Frank and Thomas Chadefaux).

## Requirements
- The analysis is run in Python 3.10.13.
- The required python libraries are listed in requirements.txt. 

## Description of files 
- /data/acled contains the dataset (acled_grid_India_2023.csv) used for the analysis.
- /data/predictions contains the outputs from the analysis (predictions for the dynamic models: linear_dynamic_thres0.5.csv and nonlinear_dynamic_thres0.5.csv, predictions for the static models: linear_static.csv and nonlinear_static.csv and two merged versions: df_linear.csv and df_nonlinear.csv), extracted protest patterns, including patterns based on artificial protest data (arima_shapes_thres0.5.json, rf_shapes_thres0.5.json, arima_shapes_thres0.5_random.json, rf_shapes_thres0.5_random.json).
- /out contains the visualizations and tables contained in the paper. 
- data.py creates the dataset used for the analysis data/acled/acled_grid_India_2023.csv. 
- functions.py contains the functions used during the analysis. 
- main.py obtains predictions within-grid. 
- main_random.py obtains random protest patterns. 
- results.py creates the outputs for the prediction model, including the aggregated protest patterns. 

## Replication instructions
First create a virtual environment, activate the environment and install the libraries. 

```
conda create -n protest_waves python=3.10.13
conda activate protest_waves
pip install -r requirements.txt
```

Then run the main files after each other. This will take approximately 24 hours.

```
python main.py
python main_random.py
```

The final results are produced by running the results file. 

 ```
python results.py
```

