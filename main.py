### reading gage height data for reach segments that have an active gage (the duplicates, meaning the reaches that had more than one active gage 
###were dropped and there were some gages that returned nan for reach ID or were disconnected from the graph, that is why the number of datapoints 
###is smaller than 255 (244< 255)


import numpy as np
import pandas as pd
import sys
import geopandas as gpd
import importlib
sys.path.append(r"C:\Users\fsgg7\Downloads\SensorPlacement")
import hclustering as hc
importlib.reload(hc)
import dt_model as dt
importlib.reload(dt)
import preprocessing
#open and save the csv file to a data frame
gageheightuniform=pd.read_csv(r"C:\Users\fsgg7\Downloads\gage_height_data\gage_height_2014_2024_appliedUniform.csv")
#making sure the index is datetime not range
gageheightuniform.set_index('dateTime', inplace=True)
gageheightuniform.index = pd.to_datetime(gageheightuniform.index)
last_available_date = gageheightuniform.index.max()

# Calculate the date 2 years before the last available date
three_years_ago = last_available_date - pd.DateOffset(years=3)

# Filter the DataFrame to only include rows from the last 2 years (for faster testing)
dff =gageheightuniform[gageheightuniform.index >= three_years_ago]
 
bridges=pd.read_csv(r"C:\Users\fsgg7\Downloads\gage_height_data\bridges_on_rivers_9_23_2024.csv", dtype={'cumAveElev': str,'cumSlope':str})
print("does bridges have a comid column?", 'COMID' in bridges.columns)
int_gages=pd.read_excel(r"C:\Users\fsgg7\Downloads\gage_height_data\intersected_gages_9_23_2024.xlsx")


#dl=dataloader(int_gages,bridges.copy(), original_brd=r"C:\Users\fsgg7\Downloads\gage_height_data\updated_bridge2024-11-03_12-17-01.xlsx")
dl=hc.dataloader(int_gages,bridges.copy(), original_brd=r"C:\Users\fsgg7\Downloads\gage_height_data\updated_bridge2024-11-03_12-17-01.xlsx")
# convert gage feature site number column to string to match the time-series column headers
int_gages['site_no'] = int_gages['site_no'].astype(str).str.zfill(8)  # change int values to str and add the preceding zeros to match
print(int_gages.shape)
print(dff.shape)

 #make sure that we only choose the features of gages that have time-series columns available
gage_features = int_gages[int_gages['site_no'].isin(dff.columns)]
gage_features.reset_index(drop=True, inplace=True)
#do the same thing for dff
dff = dff[int_gages['site_no'][int_gages['site_no'].isin(dff.columns)].values]

print('do the number of intersected gages and columns of dff match?', len(dff.columns)==gage_features['site_no'].nunique())

gage_features, bridges=preprocessing.preprocessAndNormGages(gage_features,bridges,one_hot=False)

dt_labels, _,_,_,_,_,_, cop_scores, mdavies_scores,_=dl.runClustering(dff, max_clusters=7,name="DTW", transpose=True, weights=None)
# passing clusters to decision tree as labels for classification and getting the feature importance as output
importance_arrays=dt.treeClassifier(gage_features, bridges,[dt.remove_outliers(lbl) for lbl in dt_labels[:-1]])


dl.modifygeometry()
# running the same function again with adjusted inputs, changing transpose to false and passing the weights
brd_labels, medoids,_,_,_,_,_, cop_scores, mdavies_scores,mtracker=dl.runClustering(bridges, max_clusters=25,name="EUC", transpose=False, weights=importance_arrays)

#brd_labels, medoids,_,_,_,_,_, cop_scores, mdavies_scores,mtracker=runClustering(bridges, max_clusters=250,name="EUC", transpose=False, weights=importance_arrays)

# mtracker has the original indices of pinpointed bridges, now we can load the bridge dataset and pick these locations as sensor locations based on these indices

bridgess=pd.read_csv(r"C:\Users\fsgg7\Downloads\gage_height_data\bridges_on_rivers_9_23_2024.csv", dtype={'cumAveElev': str,'cumSlope':str})
indices=list(mtracker.values())
#choose the bridges that are 
final_bridges = bridgess.loc[sorted(indices)]
final_bridges = final_bridges[['Structure NBI Submittal ID', 'Structure Length', 'Out to Out FT',
       'Out to Out IN', 'Longitude Decimal Degree', 'Latitude Decimal Degree',
       'Service Under']]
#save pinpointed bridges and their ID and geographical information in a csv file
final_bridges.to_csv(r"C:\Users\fsgg7\Downloads\final_bridge_information.csv")
