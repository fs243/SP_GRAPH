
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import ast

def preprocessAndNormGages(gages, bridges, one_hot=True):
    if 'cumDistanc' in bridges.columns:
        bridges.rename(columns={'cumDistanc': 'cumDistance'}, inplace=True)
    if 'cumAveElev' in gages.columns:
    # Convert string representations of lists into actual lists
        gages['cumAveElev'] = gages['cumAveElev'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        gages['cumElevValue'] = gages['cumAveElev'].apply(lambda x: sum(pd.to_numeric(i, errors='coerce') for i in x if pd.notna(i)))
    if 'cumAveElev' in bridges.columns:
    # Convert string representations of lists into actual lists for bridges
        bridges['cumAveElev'] = bridges['cumAveElev'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        bridges['cumElevValue'] = bridges['cumAveElev'].apply(lambda x: sum(pd.to_numeric(i, errors='coerce') for i in x if pd.notna(i)))
    # Convert elements to numeric and sum (handling non-numeric values)
    def convert_to_float_list(value):
        if isinstance(value, str):
            try:
                parsed_list=ast.literal_eval(value)
                return [float(i) for i in parsed_list]
            except(ValueError, SyntaxError, TypeError):
                return None
        return value   


    # Convert elements to numeric and take average (handling non-numeric values)
    
    '''if 'cumSlope' in gages.columns:
    # Convert string representations of lists into actual lists
        gages['cumSlope'] = gages['cumSlope'].apply(convert_to_float_list)
        gages['cumSlope']=gages['cumSlope'].apply(lambda lst: sum([i for i in lst])/len(lst))
    
    if 'cumSlope' in bridges.columns:
    # Convert string representations of lists into actual lists for bridges
        bridges['cumSlope'] = bridges['cumSlope'].apply(convert_to_float_list)
        bridges['cumSlope']=bridges['cumSlope'].apply(lambda lst: sum([i for i in lst])/len(lst))'''
                              
    if 'HydroConn' in gages.columns:
    # Handle missing values in HydroConn
   
        gages['HydroConn'] = gages['HydroConn'].fillna('Missing')
        gages['HydroConn'] = gages['HydroConn'].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)
    # Convert strings in HydroConn to integers if necessary
    if 'HydroConn' in bridges.columns:
        bridges['HydroConn'] = bridges['HydroConn'].fillna('Missing')
        bridges['HydroConn'] = bridges['HydroConn'].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)

    # Apply OrdinalEncoder
    encoder_columns = ['LEVELPATHI', 'TERMINALPA', 'UPLEVELPAT', 'DNLEVELPAT', 'HydroConn','HUC_12', 'HUC_8','HUC_10']

  
 
        
    # Apply One-Hot Encoding
    if one_hot:
        print("Entering the one-hot encoding loop")
        one_hot_encoder = OneHotEncoder(sparse_output=False)

        # Fit the one-hot encoder
        one_hot_encoder.fit(pd.concat([bridges[['HydroConn']], gages[['HydroConn']]], axis=0))

        # Transform the data
        one_hot_encoded_bridges = one_hot_encoder.transform(bridges[['HydroConn']])
        one_hot_encoded_gages = one_hot_encoder.transform(gages[['HydroConn']])

        # Create DataFrames for the one-hot encoded features with appropriate column names
        bridge_columns = [f'HydroConn_{int(cat)}' for cat in one_hot_encoder.categories_[0]]
        gage_columns = [f'HydroConn_{int(cat)}' for cat in one_hot_encoder.categories_[0]]

        # Add one-hot-encoded columns to the DataFrame
        bridges = pd.concat([bridges, pd.DataFrame(one_hot_encoded_bridges, columns=bridge_columns)], axis=1)
        gages = pd.concat([gages, pd.DataFrame(one_hot_encoded_gages, columns=gage_columns)], axis=1)
        should_drop=['cumAveElev', 'cumSlope', 'cumMinElev', 'cumMaxElev', "HydroConn"]
        # Drop original columns
        gages = gages.drop(columns=[col for col in gages.columns if col in should_drop ])
        bridges = bridges.drop(columns=[col for col in bridges.columns if col in should_drop ])


    # Select numeric columns for scaling
   
    mutual_columns=[c for c in bridges.columns if c in gages.columns]
    gagecols,bricols=gages.columns,bridges.columns
    gages.drop(columns=[col for col in gagecols if col not in mutual_columns],inplace=True)
    bridges.drop(columns=[col for col in bricols if col not in mutual_columns],inplace=True)


    unused_cols=['Unnamed: 0', 'COMID', 'GEOMETRY', 'MAXELEVSMO',\
       'MINELEVSMO', 'SLOPE', \
        'cumMinElev', 'cumMaxElev',\
       'cumAveElev', 'cumElev']
                   
    
    gages.drop(columns=unused_cols,inplace=True)
    bridges.drop(columns=unused_cols,inplace=True)
    gages.rename({'HydroConn1':'Hydroconn'}, inplace=True)
    gages.rename({'HydroConn1':'Hydroconn'}, inplace=True)


    scaler=MinMaxScaler(feature_range=(0,1000))

    bridges1 = scaler.fit_transform(bridges)
   
    gages1 = scaler.fit_transform(gages)
    bridges=pd.DataFrame(bridges1, columns=bridges.columns, index=bridges.index)
    gages=pd.DataFrame(gages1, columns=gages.columns, index=gages.index)
    bridges=bridges.apply(pd.to_numeric,errors='coerce')
    gages=gages.apply(pd.to_numeric,errors='coerce')
    return gages, bridges
