import shap
import numpy as np
import pandas as pd
import torch
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import seaborn as sns
from sklearn import tree
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence
from datetime import datetime
from collections import defaultdict
def remove_outliers(dt_label):

    labelsss=dt_label
    value_counts=(pd.Series(dt_label)).value_counts()
    
    nonoutliers=[]
    
    for value, count in value_counts.items():
      if count>=1:
          nonoutliers.append(value)
    
    res=[max(nonoutliers)+1  if lbl not in nonoutliers else lbl for lbl in labelsss] 
    return res
def showDT(tree_model,bridges,lbl):
    single_tree = tree_model
    print("len of labels",len(lbl))
    feature_names = bridges.columns
    dot_data = tree.export_graphviz(single_tree, out_file=None,
                                    feature_names=feature_names,
                                    class_names=[str(i) for i in range(len(np.unique(lbl)))],
                                    filled=True, rounded=True,
                                    special_characters=True,
                                    impurity=False,
                                    proportion=False)
    dot_data = re.sub(r'samples = \d+', '', dot_data)
    dot_data = re.sub(r'value = \[.*?\]', '', dot_data)
    graph = graphviz.Source(dot_data)
    val=np.random.randint(1,1000,(1))
    current_time=datetime.now().strftime("%Y%m%d_%H%M%S")
    
    graph.render(r"C:\Users\fsgg7\Downloads\gage_height_data\dtgraph\decision_tree_cleaned_"+f"{current_time}", format='png', cleanup=True)
    #graph.view()

def treeClassifier(gage_features,bridges,labels):

   
    result_dict=defaultdict(list)
    gage_features_n=gage_features.copy()
    bridges_n=bridges.copy()
           
    #print("len of labels", len(labels))
    max_acc=0
    importance_arrays=[[0]*len(gage_features.columns) for i in range(len(labels)) ]
    #print("lengths", len(importance_arrays),len(importance_arrays[0]), len(gage_features.columns), len(labels))
    for i in range(len(labels)):
        label = np.array(labels[i])
        #label = np.reshape(label, (label.shape[0],))
        
        X = np.array(gage_features)
        y = label
        print("shape of labels", y.shape)
        print("shape of features", X.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=192)

        # Decision Tree Model
        #tree_model = DecisionTreeClassifier(random_state=192)
        rf = RandomForestClassifier(n_estimators=100, random_state=192)
        rf.fit(X_train, y_train)

        y_pred_tree = rf.predict(X_test)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)
        

        # SHAP Analysis
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        # Feature Names
        feature_names = gage_features.columns.tolist()
        print("feature names", feature_names)

       
        # SHAP Summary Plot
        '''fig,ax= plt.subplots(figsize=(12, 20))
        #axes=axes.flatten()

        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        #
        if ax.get_legend():
            ax.get_legend().remove()
        handles, labl=ax.get_legend_handles_labels()
        fig.legend(handles, labl, bbox_to_anchor=(0.98, 0.09), loc='lower right', fontsize='small', ncol=5)
        plt.show()
        #fig.legend(bbox_to_anchor=(1,0),fontsize='small', loc='lower right',ncol=5)
        #plt.show()'''
        print("length of shap values", len(shap_values), len(shap_values[0]), len(shap_values[0][0]))
        # Feature Importance Plot
        importancess=rf.feature_importances_
        indexed_importances2 = list(enumerate(importancess))
        
        # Step 2: Sort the indexed importance values by the importance (second element in the tuple)
        sorted_importances2 = sorted(indexed_importances2, key=lambda x: x[1], reverse=True)
        
        # Get the column names corresponding to the top sorted indices
        columns_sorted_by_importance2 = [(feature_names[idx], imp) for idx, imp in sorted_importances2]
        
        # Print the top 15 features with their importance
        for ij in range(len(columns_sorted_by_importance2)):
            print(f"{i+1}. Feature {columns_sorted_by_importance2[ij][0]} ({columns_sorted_by_importance2[ij][1]:.4f})")
        
        importance_arrays.append(importancess)
        '''plt.figure(figsize=(10,6))
        plt.title("Sorted Feature Importances")
        
        # Extract feature names and importances
        features = [x[0] for x in columns_sorted_by_importance2][:15]
        importances = [x[1] for x in columns_sorted_by_importance2][:15]
        
        # Create the bar plot
        plt.bar(range(len(features)), importances, color='b', align="center")
        
        # Set feature names as x-ticks
        plt.xticks(range(len(features)), features, rotation=45, ha="right")
        plt.xlim([-1, len(features)])  # Adjust x-limits to fit all features
        
        plt.tight_layout()
        plt.show()'''
    print("passed this tree successfully")
    print( "len", len(importance_arrays), len(importance_arrays[0]))
    a=np.array(importance_arrays)
    print(type(importance_arrays))
    importance_arrays=sum(a)/len(a)
    return importance_arrays
         



