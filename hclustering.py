

import numpy as np
import pandas as pd
from tslearn.metrics import cdist_dtw
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, leaves_list
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import LineString
import ast
import folium
import random
import matplotlib.colors as mcolors
from shapely.geometry import Point
import datetime

class dataloader:
    def __init__(self,gagedf, bridgedf, original_brd=r"C:\Users\fsgg7\Downloads\gage_height_data\updated_bridge2024-11-03_12-17-01.xlsx"):
        self.bridges=bridgedf #pd.read_csv(bridgepath, dtype={'cumAveElev': str,'cumSlope':str})
        self.int_gages_int_river=gagedf #pd.read_excel(gagepath)
        self.int_gages=self.int_gages_int_river.copy()
        self.original_bridges=pd.read_excel(original_brd)
        print("is the geometry column of any bridge missing?",self.original_bridges['GEOMETRY'].isnull().sum())
        self.original_bridges=self.original_bridges[self.original_bridges['COMID'].isin(self.bridges['COMID'])]
    
    def load_wkt(self, geometry_str):
        if isinstance(geometry_str, str):
            return loads(geometry_str)
        return geometry_str
    
    def modifygeometry(self):
        
        int_gages_gage=gpd.GeoDataFrame(self.int_gages)
        int_gages_gage['geometry']=int_gages_gage['geometry'].apply(loads)
        int_gages_gage=gpd.GeoDataFrame(int_gages_gage, geometry='geometry')
        int_gages_int_river=gpd.GeoDataFrame(self.int_gages_int_river)
        int_gages_int_river['GEOMETRY'].dropna(inplace=True)
        
        int_gages_int_river['GEOMETRY']=int_gages_int_river['GEOMETRY'].apply(self.load_wkt)
        int_gages_int_river=gpd.GeoDataFrame(int_gages_int_river,geometry='GEOMETRY')
        self.original_bridges['GEOMETRY']=self.original_bridges['GEOMETRY'].apply(loads)
        self.original_bridges=gpd.GeoDataFrame(self.original_bridges, geometry='GEOMETRY')
        self.original_bridges=self.original_bridges[self.original_bridges['GEOMETRY'].notna()]
        
        self.original_bridges['previous sensors']=self.original_bridges['GEOMETRY'].apply(lambda brd: int_gages_int_river.intersects(brd).any())
        #self.original_bridges['previous_sensors']=self.original_bridges['COMID'].apply(lambda row: int_gages_int_river['COMID'].isin(row))
        self.original_bridges['previous sensors'].describe()
       
        return
    
       
    def tracksensors(self,labels,cluster_to_split, mtracker):
     
        
        zipped=zip(mtracker.values(), labels)
        
        sensor_locations=[True if x in  mtracker.values() else False for x in range(len(self.original_bridges)) ]
        labels=[labels[i]  if sensor_locations[i]==True else -1 for i in range(len(sensor_locations))]
        print("len(self.original_bridges), len(labels), len(sensor_locations)",len(self.original_bridges), len(labels), len(sensor_locations))
        print("previous sensor", self.original_bridges['previous sensors'].describe())
        merged=pd.concat([self.original_bridges,pd.DataFrame({'labels':labels, 'Is_new_sensor_location':sensor_locations})], axis=1)
        merged['Is_new_sensor_location'].fillna(False)
        new_ones=merged[(merged['labels']==cluster_to_split) | (merged['labels']==cluster_to_split+1)]
        new_ones['sensor in cluster']=new_ones.groupby(by='labels')['previous sensors'].transform('sum')
        
        new_ones['final decision']=new_ones.apply(lambda x: x['sensor in cluster']==0  ,axis=1)

        print("new_ones", new_ones)
        if len(new_ones)== len(new_ones[new_ones['final decision']==True]):
            print("the new sensor locations are equal to the previous ones")
            return +1
        elif len(new_ones[new_ones['final decision']==True])==0:
            return -1
        return 0


        
    """clustering for time series features where each row is a timestamp and columns are features, this functions processes
    dataframes as inputs
    
    Args:
        -input Dataframe
        -max_clusters: maximum number of clusters
        -name: "DTW" for dynamic time warping distance or "EUC" for Euclidean distance, default is "DTW", "CORR" for corre
        -transpose: default is True, does the matrix requires to be transposed, set True for timeseris, False for gage and bridge
        -weights: defaults to None, only pass it if you want to apply weights to the features (as in feature importance matrix)
    Returns:
        labels for each datapoint, silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores for each number of clusters
    
    
    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    
    Example:
        ```python
        df = runClustering(df, max_clusters=30, name='EUC', transpose=False,weights=None)
        ```
    """   
        
    def runClustering(self, data, max_clusters=60,name="DTW", transpose=True, weights=None):
    
        def heatmap(cdist_matrix):
            # Create a heatmap for the DTW distance matrix
            plt.figure(figsize=(8*3, 6*3))
            sns.heatmap(cdist_matrix, cmap='rainbow', annot=True, fmt=".2f", cbar=True)
            plt.title('DTW Cross-Distance Matrix Heatmap')
            plt.xlabel('Sample Index')
            plt.ylabel('Sample Index')
            plt.show()
    
        def plotscores(scores,names):
          num_of_cols=2
          num_of_rows=int(np.ceil(len(scores)//num_of_cols))
          k=0
        # Plotting the values
          fig, axes=plt.subplots(num_of_rows, num_of_cols,figsize=(20, 6))
          axes=axes.flatten()
          for k in range(len(scores)):
             
                  
              num_clusters = [x+2 for x, _ in scores[k]]
              values = [y for _, y in scores[k]]
              
            
              #  Score plot
              axes[k].plot(num_clusters, values, marker='o', color='b')
            
              # Labeling the plot
              '''axes[i,j].xlabel('Number of Clusters')
              axes[i,j].ylabel('Score Value')'''
              axes[k].set_title(f"{names[k]}")
            
              axes[k].grid(True)
        
          # Show the plot
          plt.tight_layout()
          plt.show()
        def davies_bouldin_score_dtw(dtw_dist_matrix, labels):
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
        
            # Dictionary to store intra-cluster distances (i.e., within-cluster scatter)
            intra_cluster_dists = defaultdict(float)
            cluster_centroids = {}
        
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                cluster_distances = dtw_dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        
                # Calculate the within-cluster distance (average pairwise distance within cluster)
                intra_cluster_dists[label] = np.mean(cluster_distances)
        
                # The centroid is just the "medoid" for k-medoids
                cluster_centroids[label] = cluster_indices[np.argmin(np.sum(cluster_distances, axis=1))]
        
            # Calculate the Davies-Bouldin score
            db_index = 0.0
            for i in unique_labels:
                max_ratio = float('-inf')
                for j in unique_labels:
                    if i != j:
                        # Inter-cluster distance between cluster i and cluster j (between their centroids)
                        inter_cluster_dist = dtw_dist_matrix[cluster_centroids[i], cluster_centroids[j]]
        
                        # Davies-Bouldin ratio: (scatter_i + scatter_j) / distance(i, j)
                        ratio = (intra_cluster_dists[i] + intra_cluster_dists[j]) / inter_cluster_dist
                        max_ratio = max(max_ratio, ratio)
        
                db_index += max_ratio
        
            return db_index / n_clusters
    
        import numpy as np
    
        def compute_cop_index(distance_matrix, labels):
            unique_labels = set(labels)
            n_clusters = len(unique_labels)
            max_intra_cluster_distance = []
            min_inter_cluster_distance = []
        
            # Calculate intra-cluster distances
            for label in unique_labels:
                cluster_points = np.where(np.array(labels) == label)[0]
                if len(cluster_points) > 1:  # Only calculate if cluster has more than one point
                    intra_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
                    max_intra_cluster_distance.append(np.max(intra_distances))
        
            # Calculate inter-cluster distances
            for label_a in unique_labels:
                cluster_a_points = np.where(np.array(labels) == label_a)[0]
                for label_b in unique_labels:
                    if label_a != label_b:
                        cluster_b_points = np.where(np.array(labels) == label_b)[0]
                        inter_distances = distance_matrix[np.ix_(cluster_a_points, cluster_b_points)]
                        min_inter_cluster_distance.append(np.min(inter_distances))
            epsilon=1e-7
            # COP Index: Ratio of max intra-cluster distance to min inter-cluster distance
            cop_index = max(max_intra_cluster_distance) / min(min_inter_cluster_distance)+epsilon
            return cop_index
        
        # Usage
        # cop_index = compute_cop_index(distance_matrix, labels)
        # print("COP Index:", cop_index)
        def compute_modified_davies_bouldin(distance_matrix, labels, epsilon=1e-10):
            unique_labels = set(labels)
            n_clusters = len(unique_labels)
            intra_cluster_scatter = {}
            inter_cluster_distances = np.full((n_clusters, n_clusters), np.inf)
        
            # Calculate intra-cluster scatter (S_i)
            for i, label in enumerate(unique_labels):
                cluster_points = np.where(np.array(labels) == label)[0]
                if len(cluster_points) > 1:
                    intra_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
                    intra_cluster_scatter[label] = np.mean(intra_distances)
                else:
                    intra_cluster_scatter[label] = 0  # Single-point clusters have zero scatter
        
            # Calculate inter-cluster minimum distances (M_ij)
            label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            for label_a in unique_labels:
                for label_b in unique_labels:
                    if label_a != label_b:
                        cluster_a_points = np.where(np.array(labels) == label_a)[0]
                        cluster_b_points = np.where(np.array(labels) == label_b)[0]
                        inter_distances = distance_matrix[np.ix_(cluster_a_points, cluster_b_points)]
                        inter_cluster_distances[label_to_index[label_a], label_to_index[label_b]] = np.min(inter_distances)
        
            # Calculate modified Davies-Bouldin index
            db_index = 0.0
            for label in unique_labels:
                max_ratio = 0.0
                for other_label in unique_labels:
                    if label != other_label:
                        Si = intra_cluster_scatter[label]
                        Sj = intra_cluster_scatter[other_label]
                        Mij = inter_cluster_distances[label_to_index[label], label_to_index[other_label]]
                        max_ratio = max(max_ratio, (Si + Sj) / (Mij + epsilon))
                db_index += max_ratio
        
            db_index /= n_clusters
            return db_index
    
    
        def calinski_harabasz_score_dtw(dtw_dist_matrix, labels):
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            n_samples = len(labels)
        
            # Overall centroid (medoid of all time series)
            overall_centroid = np.argmin(np.sum(dtw_dist_matrix, axis=1))
        
            # Dictionary to store intra-cluster distances and cluster centroids
            intra_cluster_dists = defaultdict(float)
            cluster_centroids = {}
        
            # Total within-cluster variance (scatter)
            Wk = 0.0
        
            for label in unique_labels:
                cluster_indices = np.where(labels == label)[0]
                cluster_distances = dtw_dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        
                # Calculate the within-cluster distance (average pairwise distance within cluster)
                intra_cluster_dists[label] = np.sum(cluster_distances) / len(cluster_indices)
        
                # The centroid (medoid) for k-medoids is the point that minimizes the total distance to all others
                cluster_centroids[label] = cluster_indices[np.argmin(np.sum(cluster_distances, axis=1))]
        
                # Sum of within-cluster distances
                Wk += np.sum(cluster_distances)
        
            # Total between-cluster variance
            Bk = 0.0
        
            for label in unique_labels:
                cluster_size = np.sum(labels == label)
                # Distance between the cluster centroid and the overall centroid
                inter_cluster_dist = dtw_dist_matrix[cluster_centroids[label], overall_centroid]
                Bk += cluster_size * (inter_cluster_dist ** 2)
        
            # Calinski-Harabasz score
            numerator = Bk / (n_clusters - 1)
            denominator = Wk / (n_samples - n_clusters)
        
            ch_score = numerator / denominator
            return ch_score
            
    
    
    
    # Step 2: Function to run K-Medoids and split clusters
        def customClustering(data, num_clusters,name,transpose, weights,bridges=self.bridges):
        #print("shape of input data",data.shape)
            if not transpose:
              data_list=data.tolist()
              corr_coeff=pd.DataFrame(data).corr()
              unique_values={val:idx for idx, val in enumerate(pd.unique(bridges['HydroConn']))}
              value_counts=bridges['HydroConn'].value_counts()
              sorted_counts=value_counts.sort_values(ascending=False)
                
              summ=0
              selected_values=[]
              
              for value, count in sorted_counts.items():
                  
                  if summ <245:
                      summ+=1
                      selected_values.append(unique_values[value])
                  else:
                      break
              labelsss = np.zeros(len(data_list), dtype=int)
            
              labels=[max(selected_values)+1 for lbl in labelsss if lbl not in selected_values] # all datapoints start with their coarse graph values
            else:
              data_list=data.T.tolist()
              corr_coeff=pd.DataFrame(data).T.corr()
              labels = np.zeros(len(data_list), dtype=int)  # All data points start with label 0
              
            original_indices = list(range(len(data_list)))  # Keep track of original indices
            silhouette_scores,davies_bouldin_scores,db_dtw_scores,calinski_harabasz_scores,labelss,all_medoids_indices,ch_dtw_scores = [],[],[],[],[],[],[]
            cop_dtws,modified_daviess=[],[]
            #print("shape of correlation matrix", corr_coeff.shape)
            corr_distance=pd.DataFrame(index=corr_coeff.columns,columns=corr_coeff.columns)
            for col in corr_coeff.columns:
              corr_distance[col]=corr_coeff[col].apply(lambda x: 1-x)
            corr_distance_matrix=corr_distance.to_numpy()
            #print("shape of ditance corr matrix", corr_distance_matrix.shape)
            
            if name=="DTW":
              dist_mtr=cdist_dtw(data_list)
             #heatmap(dist_mtr)
              #print("shape of DTW distance matrix", len(dist_mtr), len(dist_mtr[0]))
            # Initialize: all data points belong to one cluster (label = 0)
              if weights is not None:
                          # Apply weights to the distance matrix
                dist_mtr *= weights  # Assuming weights is a 2D array of the same shape as dist_mtr
            elif name=="CORR":
              dist_mtr=corr_distance_matrix
            
            else:
              dist_mtr = euclidean_distances(data_list)
            clusters = [data_list]  # Start with one cluster
            labels = np.zeros(len(data_list), dtype=int)  # All data points start with label 0
        
            # Store initial index and label together using zip
            index_label_zip = list(zip(original_indices, labels))
            all_medoids_iterations=[]
    
    
    
            medoid_tracker = defaultdict(int)
            
            #for k in tqdm(range(1, num_clusters)):
            progress_bar=tqdm()
            k=0
            count=0
            while count< num_clusters+2:
                k+=1
                max_sum_pairwise_distances = []
                for cluster_idx, cluster in enumerate(clusters):
                    cluster_indices = [i for i, label in enumerate(labels) if label == cluster_idx]
                    #choose the distance matrix of each cluster and calculate the pairwise distances
                    cluster_dist_mtr = dist_mtr[np.ix_(cluster_indices, cluster_indices)]
                    sum_pairwise_distances = np.sum(cluster_dist_mtr)
                    max_sum_pairwise_distances.append(sum_pairwise_distances)
                #choosing the cluster with maximum intra cluster variance as the one to be split
                cluster_to_split = np.argmax(max_sum_pairwise_distances)
            
                # Get the data points in the cluster to split
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_to_split]
                non_cluster_indices = [i for i, label in enumerate(labels) if label != cluster_to_split]
            
                cluster_dist_mtr = dist_mtr[np.ix_(cluster_indices, cluster_indices)]
            
                # Calculate new medoids
                first_medoid = np.argmin(np.sum(cluster_dist_mtr, axis=1))
                second_medoid = np.argmax(cluster_dist_mtr[first_medoid])
    
                #
                first_medoid = np.random.randint(len(cluster_indices)//2, len(cluster_indices))
                second_medoid = np.random.randint(0, len(cluster_indices)//2)
            
                kmedoid_instance = kmedoids(cluster_dist_mtr, [first_medoid, second_medoid], data_type='distance_matrix')
                kmedoid_instance.process()
            
                # Get new labels for the two new clusters
                new_clusters = kmedoid_instance.get_clusters()
                # get medoid indices for the new clusters
                new_medoids = kmedoid_instance.get_medoids()
            
                print('new medoids', new_medoids)
            
                # Update labels for the current clusters and the new clusters
                labels = [lbl + 1 if lbl > cluster_to_split else lbl for lbl in labels]
                for new_cluster_id, new_cluster in enumerate(new_clusters):
                    for index, val in enumerate(new_cluster):
                        labels[cluster_indices[val]] = cluster_to_split + new_cluster_id
            
                # Zip the original indices with updated labels
                index_label_zip = list(zip(original_indices, labels))
            
                # Get the original medoid indices
                original_medoids = [cluster_indices[medoid] for medoid in new_medoids]
            
                # Ensure uniqueness of new medoids before adding them
                original_medoids = list(set(original_medoids))
            
                
                
                # updating the dictionary that has all the medoid indices saved 
                if not transpose:
                    add=self.tracksensors(labels,cluster_to_split,medoid_tracker) 
                    count+=add
                    
                    if cluster_to_split+1 in medoid_tracker:
                        medoid_tracker[max(medoid_tracker.keys())+1]= medoid_tracker[cluster_to_split+1]
                        
                        medoid_tracker[cluster_to_split]=original_medoids[0]
                        medoid_tracker[cluster_to_split+1]=original_medoids[1] if original_medoids[1] is not None else np.choice(original_indices)
                    else:
                        medoid_tracker[cluster_to_split]=original_medoids[0]
                        medoid_tracker[cluster_to_split+1]=original_medoids[1] if original_medoids[1] is not None else np.choice(original_indices)
                    all_medoids_indices.extend(original_medoids)
                # Prevent duplicates in all_medoids_indices
                    if 'all_medoids_indices' not in locals():
                        all_medoids_indices = []  # Initialize if it does not exist
                    all_medoids_indices = list(set(all_medoids_indices + original_medoids))  # Ensure uniqueness
                    
                    # Debugging information
                    print("cluster to split", cluster_to_split)
                    print("original indices:", original_medoids)
                    print("Total medoids:", len(all_medoids_indices))
                    print("Unique medoids:", len(set(all_medoids_indices)))
                    print("Medoid tracker values:", medoid_tracker)
                    print("Unique values in medoid tracker:", len(set(medoid_tracker.values())))
            
            
            
                # Rebuild clusters based on newly added labels
                clusters = [np.array([data_list[i] for i in np.where(labels == l)[0]]) for l in np.unique(labels)]
            
                # Aggregate the data points into a dictionary by label
                diction = {label: [] for label in np.unique(labels)}
                for orig_idx, label in index_label_zip:
                    diction[label].append(orig_idx)
            
                #calculate clustering metrics based on the distance matrix type
                if len(np.unique(labels)) > 1:
                      if name=="DTW" or name== 'CORR':
                        silhouette_avg = silhouette_score(dist_mtr, labels, metric='precomputed')
                        db_score_dtw = davies_bouldin_score_dtw(dist_mtr,labels)
                        ch_score_dtw=calinski_harabasz_score_dtw(dist_mtr, labels)
                        cop_dtw=compute_cop_index(dist_mtr, labels)
                        modified_davies=compute_modified_davies_bouldin(dist_mtr, labels, epsilon=1e-10)
                      else:
                        dist_mtr = euclidean_distances(data_list)
                        silhouette_avg = silhouette_score(data_list, labels)
                        db_score_dtw = davies_bouldin_score_dtw(dist_mtr,labels)
                        ch_score_dtw=calinski_harabasz_score_dtw(dist_mtr, labels)
                        cop_dtw=compute_cop_index(dist_mtr, labels)
                        modified_davies=compute_modified_davies_bouldin(dist_mtr, labels, epsilon=1e-10)
                    
                    
                      
                else:
                    silhouette_avg = -1  # Invalid score for 1 cluster
                    davies_bouldin_avg = +1  # Invalid score for 1 cluster
                    calinski_harabasz_avg = -1  # Invalid score for 1 cluster
                    ch_score_dtw=-np.inf
                    db_score_dtw=np.inf
                    calinski_harabasz_avg = -np.inf
                    cop_dtw=100
                    modified_davies=100
                #saving all clustering evaluation metrics in an array
                silhouette_scores.append((k,silhouette_avg))
                davies_bouldin_scores.append((k,db_score_dtw))
                db_dtw_scores.append((k,db_score_dtw))
                calinski_harabasz_scores.append((k, ch_score_dtw))
                ch_dtw_scores.append((k,ch_score_dtw))
                labelss.append(labels)
                cop_dtws.append((k,cop_dtw))
                modified_daviess.append((k,modified_davies))
                if transpose:
                    count+=1
                #count=count+self.tracksensors(labels,cluster_to_split, medoid_tracker) if not transpose else count+1
                progress_bar.update()
                
            
            progress_bar.close()
            
            return labelss,all_medoids_indices, silhouette_scores,db_dtw_scores,ch_dtw_scores,davies_bouldin_scores,calinski_harabasz_scores,cop_dtws, modified_daviess,medoid_tracker
    
    
    
    
    
    
        # Run Clustering from 1 to 30 clusters
        num_clusters = max_clusters
        
        if 'dateTime' in data.columns:
            data=data.reset_index()
            data.drop(columns=['dateTime'],inplace=True)
        elif 'datetime' in data.columns:
            data=data.reset_index()
            data.drop(columns=['datetime'],inplace=True)
        
        data=data.apply(pd.to_numeric, errors='coerce')
        data=data.to_numpy()
        
        
        # running the clustering algorithm 
        
        labels, medoids,silhouette_scores,db_dtw_scores,ch_dtw_scores,davies_scores, calinski_scores,cop_scores,mdavies_scores,mtracker = customClustering(data, num_clusters,name,transpose, weights)
        # getting the best number of clusters based on each evaluation index/metric
        silhouette_index = np.argmax([y for x,y in silhouette_scores])
        davies_bouldin_scores_index = np.argmin([y for x,y in davies_scores])
        calinski_harabasz_scores_index = np.argmax([y for x,y in calinski_scores ])
        cop_scores_index = np.argmin([y for x,y in cop_scores])
        modified_davies_scores_index = np.argmin([y for x,y in mdavies_scores ])
        db_dtw_scores_index=np.argmin([y for x,y in db_dtw_scores ])
        ch_dtw_scores_index=np.argmax([y for x,y in ch_dtw_scores ])
       
        #printing the evaluation index values
        
        print(f"Optimal number of clusters based on Silhouette Score: {silhouette_scores[silhouette_index][0]} value: {silhouette_scores[silhouette_index][1] }")
        print(f"Optimal number of clusters based on davies with DTW: {db_dtw_scores[db_dtw_scores_index][0]} value: {db_dtw_scores[db_dtw_scores_index][1] }")
        print(f"Optimal number of clusters based on Calinski harabasz with DTW: {ch_dtw_scores[ch_dtw_scores_index][0]} value: {ch_dtw_scores[ch_dtw_scores_index][1] }")
        print(f"Optimal number of clusters based on Davies Bouldin: {davies_scores[davies_bouldin_scores_index][0]} value: {davies_scores[davies_bouldin_scores_index][1] }")
    
        print(f"Optimal number of clusters based on Calinski harabasz: {calinski_scores[calinski_harabasz_scores_index][0]} value: {calinski_scores[calinski_harabasz_scores_index][1] }")
    
        print(f"Optimal number of clusters based on modified davies index: {mdavies_scores[modified_davies_scores_index][0]} value: {mdavies_scores[modified_davies_scores_index][1] }")
    
        print(f"Optimal number of clusters based on cop index: {cop_scores[cop_scores_index][0]} value: {cop_scores[cop_scores_index][1] }")
        f_scores,names=[mdavies_scores,cop_scores],["Modified Davies-Bouldin Index","COP Index"]
        plotscores(f_scores,names)
        #you could plot other metrics of your choice too, but these two metrics were the most suitable
        '''plotscores(silhouette_scores,"silhouette")
        plotscores(db_dtw_scores,"davies bouldin with dtw ")
        plotscores(ch_dtw_scores,"calinski with dtw ")
        plotscores(davies_scores,"davies bouldin")
        
        plotscores(calinski_scores,"calinski harabasz")
        plotscores(cop_scores,"cop index")
        plotscores(mdavies_scores,"modified davies bouldin")'''
        return labels, medoids,silhouette_scores,db_dtw_scores,ch_dtw_scores,davies_scores,calinski_scores, cop_scores, mdavies_scores,mtracker
    
 
