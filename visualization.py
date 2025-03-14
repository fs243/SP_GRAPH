import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
from shapely.wkt import loads
from shapely.geometry import LineString
import ast
import folium
import random
import matplotlib.colors as mcolors
from shapely.geometry import Point
import datetime  

def feature_histogram(gage_features):
    

    X=gage_features
    # Initialize the plot
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))  # 3 rows, 5 columns for subplots
    axes = axes.ravel()  # Flatten the 3x5 grid to easily loop through the axes
    
    for i, col in enumerate(X.columns[:24]):  # Assuming the first 24 columns are features
        ax = axes[i]
    
        # Plot histogram of the feature
        sns.histplot(X[col], bins=30, kde=False, stat='density', ax=ax, color='skyblue', edgecolor='black')
    
        # Fit a normal distribution to the feature data
        mean, std = norm.fit(X[col])
    
        # Generate data for normal distribution curve
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mean, std)
    
        # Plot the normal distribution curve
        ax.plot(x, p, 'r', linewidth=2)
    
        # Set the title and labels
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()



def createheatmap_corr(df):
# Calculate the correlation matrix (or use your data matrix)
    corr_matrix = df.corr()
    linkage_matrix= linkage(corr_matrix, method='average')
    dendro_order= leaves_list(linkage_matrix)
    sorted_corr_matrix= corr_matrix.iloc[dendro_order,dendro_order]
    sorted_labels=df.columns[dendro_order]
    # Create a heatmap with the 'jet' colormap
    plt.figure(figsize=(12, 8))
    sns.heatmap(sorted_corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
    
    # Add a title
    plt.title("correlation coefficient of topographical and hydraulical features")
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def visualize():
    sensorlabeldf=pd.read_csv(r"C:\Users\fsgg7\Downloads\gage_height_data\sensorclusterinfo.csv")
    
    
    sensor_locations=[True if x in list(mtracker.values()) else False for x in range(len(bridges)) ]
    bridges2=pd.read_csv(r"C:\Users\fsgg7\Downloads\gage_height_data\bridges_on_rivers_9_23_2024.csv")
    merged=pd.concat([bridges2,pd.DataFrame({'labels':labels, 'Is_sensor_location':sensor_locations})], axis=1)
    cumulative_rows=[]
    merged['GEOMETRY']=merged['GEOMETRY'].apply(loads)
    merged= gpd.GeoDataFrame(merged, geometry='GEOMETRY')
    indices=[]
    
    for i in range(merged['labels'].nunique()):
        indices.append(merged.index[merged['labels']==i].tolist())
        
    
    merged['GEOMETRY']=merged['GEOMETRY'].apply(lambda x:  LineString(list((p[0],p[1]) for p in x.coords)))
    
    
    
    indices = []
    for i in range(merged['labels'].nunique()):
        indices.append(merged.index[merged['labels'] == i].tolist())
    
    # Convert the DataFrame to a GeoDataFrame
    merged = gpd.GeoDataFrame(merged, geometry='GEOMETRY', crs="EPSG:4326")
    
    # Calculate centroid for map centering
    centroid = merged.geometry.centroid.unary_union.centroid
    map_center = [centroid.y, centroid.x]
    
    # Create a base folium map centered at the centroid
    m = folium.Map(location=map_center, zoom_start=6)
    
    # Generate a list of colors using matplotlib's colormap
    colormap = list(mcolors.CSS4_COLORS.values())  # Get a large list of named CSS colors
    
    # If 300 colors are needed and not available, generate random hex colors
    if len(colormap) < 300:
        colormap.extend([f"#{''.join([random.choice('0123456789ABCDEF') for _ in range(6)])}" for _ in range(300 - len(colormap))])
    
    # Ensure exactly 300 unique colors
    random.shuffle(colormap)
    colormap = colormap[:300]
    
    # Transform to a different CRS if needed (use standard syntax)
    merged = merged.to_crs("EPSG:4269")
    
    # Loop through the groups of indices and assign each group a unique color
    for idx, group_indices in enumerate(indices):  # Assuming `indices` is a list of lists
        group_color = colormap[idx]  # Pick a unique color for the group
    
        # Plot each LineString for the group
        for indx in group_indices:
            if merged['GEOMETRY'] is not None and not merged['GEOMETRY'].iloc[indx].is_empty:  # Check that geometry is not empty
                folium.GeoJson(
                    merged['GEOMETRY'].iloc[indx],
                    style_function=lambda x, col=group_color: {'color': col, 'weight': 2}
                ).add_to(m)
    
    # Filter sensor locations based on 'final decision2'
    sensorlocationGeodata = merged[merged['final decision'] == True]
    sensorlocationGeodata['sensor geometry'] = sensorlocationGeodata.apply(
        lambda row: Point(row['Longitude Decimal Degree'], row['Latitude Decimal Degree']), axis=1
    )
    sensorlocationGeodata = gpd.GeoDataFrame(sensorlocationGeodata, geometry='sensor geometry', crs="EPSG:4326")
    
    # Add sensor location geometries to the map with a distinct color
    folium.GeoJson(
        sensorlocationGeodata['sensor geometry'],
        style_function=lambda x: {'color': colormap[-1], 'weight': 6}
    ).add_to(m)
    
    # Save the map as an HTML file to view in the browser
    addon = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    m.save(rf'C:\Users\fsgg7\Downloads\gage_height_data\ReachwithBridgePlusSensorsHDB_{addon}.html')
