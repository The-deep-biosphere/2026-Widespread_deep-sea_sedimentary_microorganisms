#%%
# Source: https://hands-on.cloud/implementation-of-support-vector-machine-svm-using-python/
# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import os

# sklearn imports
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

# area- and core-stratified splitting, nested feature selection, validation tuning
from core_split import AREA_CORES, area_core_split, downsample_core, select_features, fit_tuned_tree

# =============================================
# ---------- EXPORT in Rules Format -----------
# =============================================

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

def DT_Features(rules):

    rules = str(rules)
    # create list of OTUs in rules
    splitted = rules.split(')')
    features = []

    # get all mentions of OTUs
    for i in range(0,len(splitted)-1):
        x = splitted[i]
        x1 = x.split('(')[1]
        x2 = x1.split(' ')[0]
        
        if 'OTU_' in x2:
            features.append(x2)

    # count frequency of OTUs in list
    frequency = {}
    for word in features:

        count = frequency.get(word,0)
        frequency[word] = count + 1

    # open file for writing, "w" is writing
    w = []

    # loop over dictionary keys and values
    for key, val in frequency.items():
        # write every key and value to file
        w.append([key, val])

    w = pd.DataFrame(w)
    w.columns = ['OTU','value']
    # sort on descending order
    # sort = pd.read_csv(name, names=['OTU','value'])
    sort = w.sort_values(by='value', ascending=False)

    return sort

def zonation (values, threshold):
    # ============
    # Give labels
    # ============
    Zonation = [2]*len(values)

    for i,value in enumerate(values):

        if np.isnan(value):
            Zonation[i] = float('nan')
        elif value > threshold:
            Zonation[i] = 1
        else:
            Zonation[i] = 0
    
    return Zonation


variables = ['O2conc','NOxconc','Mnconc','NH4conc']
thresholds = [0.0,0.003,0.005,0.010,0.015]
thresholds = np.array(thresholds)
accuracy = [[float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
            [float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
            [float('nan'),float('nan'),float('nan'),float('nan'),float('nan')],
            [float('nan'),float('nan'),float('nan'),float('nan'),float('nan')]]

accuracy_AMOR_pres = np.array(accuracy)
accuracy_AMOR_abs = np.array(accuracy)
accuracy_WNAG_pres = np.array(accuracy)
accuracy_WNAG_abs = np.array(accuracy)
accuracy_MAR_pres = np.array(accuracy)
accuracy_MAR_abs = np.array(accuracy)
accuracy_NWPO_pres = np.array(accuracy)
accuracy_NWPO_abs = np.array(accuracy)
accuracy_SCS_pres = np.array(accuracy)
accuracy_SCS_abs = np.array(accuracy)
accuracy_SPO_pres = np.array(accuracy)
accuracy_SPO_abs = np.array(accuracy)

number_AMOR_pres = np.array(accuracy)
number_AMOR_abs = np.array(accuracy)
number_WNAG_pres = np.array(accuracy)
number_WNAG_abs = np.array(accuracy)
number_MAR_pres = np.array(accuracy)
number_MAR_abs = np.array(accuracy)
number_NWPO_pres = np.array(accuracy)
number_NWPO_abs = np.array(accuracy)
number_SCS_pres = np.array(accuracy)
number_SCS_abs = np.array(accuracy)
number_SPO_pres = np.array(accuracy)
number_SPO_abs = np.array(accuracy)

# chemical columns used for the significant-OTU selection (as in 4.5)Significant_genes.py)
var_columns = ['O2conc','NOxconc','Mnconc','NH4conc']
iterations = 1000

# per area: (accuracy absent, accuracy present, number absent, number present)
results_arrays = {'AMOR': (accuracy_AMOR_abs, accuracy_AMOR_pres, number_AMOR_abs, number_AMOR_pres),
                  'WNAG': (accuracy_WNAG_abs, accuracy_WNAG_pres, number_WNAG_abs, number_WNAG_pres),
                  'MAR':  (accuracy_MAR_abs,  accuracy_MAR_pres,  number_MAR_abs,  number_MAR_pres),
                  'NWPO': (accuracy_NWPO_abs, accuracy_NWPO_pres, number_NWPO_abs, number_NWPO_pres),
                  'SCS':  (accuracy_SCS_abs,  accuracy_SCS_pres,  number_SCS_abs,  number_SCS_pres),
                  'SPO':  (accuracy_SPO_abs,  accuracy_SPO_pres,  number_SPO_abs,  number_SPO_pres)}

def nanmean_or_nan(values):
    values = np.array(values, dtype=float)
    return float('nan') if np.all(np.isnan(values)) else np.nanmean(values)

for i, var in enumerate(variables):

    for j, thres in enumerate(thresholds):
         
        print("========================== \n")       
        print(f"Variable is: {var} \n")
        print(f"Threshold is: {thres} \n")
        print("========================== \n")    
        # ===================
        # Data selection
        # ===================
        data = pd.read_csv(os.getcwd()+"/../1)Preprocessing/4)Significant_gene/Data_prep_ML_FAMILY_datacomparison.csv",
                delimiter=",", dtype=str, index_col=None)
        data.columns = data.columns.str.replace(data.columns[0], 'Core')
        # data['Core'] = data['Core'].astype(float).astype(int)
        # data.reset_index(drop=True, inplace=True)
        data = data.astype(float)
        data['Core'] = data['Core'].astype(int)
        # all samples (every chemical) of the cores of interest, used for the feature selection
        data_fs = data[~data['Core'].isin([18, 19, 21, 22])]
        data = data.dropna(subset = [var], how ='any')
        data.reset_index(drop=True, inplace=True) 
        data['zonation'] = zonation(data[var].values,thres)
        data['zonation']


        # ==================================================
        # ---------- REMOVE sample in TRANSITION -----------
        # ==================================================
        predictor_loc = np.where(data.columns == 'zonation')[0][0]

        # remove cores that are not of interest (GS19GC25, GS21GC09, GS20GC20, GS20GC21)
        data = data[data['Core'] != 18]
        data = data[data['Core'] != 19]
        data = data[data['Core'] != 22]
        data = data[data['Core'] != 21]

        data.reset_index(drop=True, inplace=True)

        # ================================
        # Assigning X and Y variables 
        # ================================
        otu_columns = [c for c in data.columns if c.startswith('OTU_')]
        X_all = data[otu_columns]
        y = np.array(data.iloc[:, predictor_loc])
        core = np.array(data['Core'])

        # accuracy and number of samples per area over the iterations
        results = {area: {'abs': [], 'pres': [], 'n_abs': [], 'n_pres': []} for area in results_arrays}

        for it in range(iterations):

            # =================================
            # ---------- SPLIT DATA -----------
            # =================================
            seed = 650 + it               # new, reproducible split in every iteration
            # train / validation / test split (0.7/0.15/0.15 of the cores),
            # stratified on area and then on core: a core is in only one set
            train_idx, val_idx, test_idx = area_core_split(core, seed)

            # core GS19GC10 (core == 17) represents the dataset the most (207 samples);
            # remove 1/4 of its samples from the training set to reduce its weight
            train_idx = downsample_core(train_idx, core, seed)

            # select relevant features according to q-value, using the training cores only
            feature_selection = select_features(data_fs, np.unique(core[train_idx]), var_columns,
                                                otu_columns, seed, clr=True)
            X = X_all[feature_selection].values.astype(float)

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test, core_test = X[test_idx], y[test_idx], core[test_idx]

            # ===========================================
            # ---------- TRAIN AND TEST MODEL -----------
            # ===========================================
            # max_depth is chosen on the validation cores (ccp_alpha = 0 as before)
            dtree_sklearn = fit_tuned_tree(DecisionTreeClassifier(criterion='gini', ccp_alpha=0, random_state=seed),
                                           X_train, y_train, X_val, y_val)

            # ------------------------------------
            # Accuracy per area on the test cores
            # ------------------------------------
            for area in results:
                sel = np.isin(core_test, AREA_CORES[area])
                y_area = y_test[sel]
                y_area_pred = dtree_sklearn.predict(X_test[sel]) if sel.any() else np.array([])
                A = y_area == 0
                P = y_area == 1
                results[area]['abs'].append(accuracy_score(y_area[A], y_area_pred[A]) if A.any() else float('nan'))
                results[area]['pres'].append(accuracy_score(y_area[P], y_area_pred[P]) if P.any() else float('nan'))
                results[area]['n_abs'].append(A.sum() if A.any() else float('nan'))
                results[area]['n_pres'].append(P.sum() if P.any() else float('nan'))

        # mean over the iterations
        for area, (acc_abs, acc_pres, n_abs, n_pres) in results_arrays.items():
            acc_abs[i,j] = nanmean_or_nan(results[area]['abs'])
            acc_pres[i,j] = nanmean_or_nan(results[area]['pres'])
            n_abs[i,j] = nanmean_or_nan(results[area]['n_abs'])
            n_pres[i,j] = nanmean_or_nan(results[area]['n_pres'])
        
# fig.delaxes(ax[2][1])
# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(2,3,figsize=(20, 10))

locations = ['AMOR','WNAG','MAR','NWPO',"SCS","SPO"]
col = ['orange','red','darkgreen','purple']
variables = [r'$O_2$',r'$NO_x$',r'$Mn$',r'$NH_4$']
r = 0
c = 0

accuracy_abs = [accuracy_AMOR_abs,accuracy_WNAG_abs,accuracy_MAR_abs,accuracy_NWPO_abs,accuracy_SCS_abs,accuracy_SPO_abs]
accuracy_pres = [accuracy_AMOR_pres,accuracy_WNAG_pres,accuracy_MAR_pres,accuracy_NWPO_pres,accuracy_SCS_pres,accuracy_SPO_pres]
number_abs = [number_AMOR_abs,number_WNAG_abs,number_MAR_abs,number_NWPO_abs,number_SCS_abs,number_SPO_abs]
number_pres = [number_AMOR_pres,number_WNAG_pres,number_MAR_pres,number_NWPO_pres,number_SCS_pres,number_SPO_pres]

for l, loc in enumerate(locations):
    ax[r,c].grid(axis='both')
    for v, var in enumerate(variables):
        ax[r,c].plot(thresholds*1000, accuracy_pres[l][v,:], color = col[v],linestyle='-')
        ax[r,c].scatter(thresholds*1000, accuracy_pres[l][v,:], color = col[v])
        ax[r,c].plot(thresholds*1000, accuracy_abs[l][v,:], color = col[v], linestyle='--')
        ax[r,c].scatter(thresholds*1000, accuracy_abs[l][v,:], color = col[v],  label = var)

    ax[r,c].set_title(loc, fontweight='bold')
    ax[r,c].set_ylim(-0.05,1.05)
    ax[r,c].set_xlabel('Threshold (μM)')
    ax[r,c].set_ylabel('Accuracy (%)')
    
    if c < 2:
        c += 1
    else:
        r += 1
        c = 0

 
handles, labels = ax[0,0].get_legend_handles_labels()

linePres = Line2D([0], [0], label='Present', color='darkgrey', linestyle = '-')
lineAbs = Line2D([0], [0], label='Absent', color='darkgrey', linestyle = '--')

handles.extend([linePres,lineAbs])

fig.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.245, wspace=0.2)
plt.savefig(os.getcwd()+'/Sensitivity_analysis_accuracy.png')

fig.show()

fig, ax = plt.subplots(2,3,figsize=(20, 10))

locations = ['AMOR','WNAG','MAR','NWPO',"SCS","SPO"]
col = ['orange','red','darkgreen','purple']
variables = [r'$O_2$',r'$NO_x$',r'$Mn$',r'$NH_4$']
r = 0
c = 0

for l, loc in enumerate(locations):
    ax[r,c].grid(axis='both')
    for v, var in enumerate(variables):
        ax[r,c].plot(thresholds*1000, number_pres[l][v,:], color = col[v],linestyle='-')
        ax[r,c].scatter(thresholds*1000, number_pres[l][v,:], color = col[v])
        ax[r,c].plot(thresholds*1000, number_abs[l][v,:], color = col[v], linestyle='--')
        ax[r,c].scatter(thresholds*1000, number_abs[l][v,:], color = col[v],  label = var)

    ax[r,c].set_title(loc, fontweight='bold')
    ax[r,c].set_xlabel('Threshold (μM)')
    ax[r,c].set_ylabel('Number of samples')
    
    if c < 2:
        c += 1
    else:
        r += 1
        c = 0

handles, labels = ax[0,0].get_legend_handles_labels()

linePres = Line2D([0], [0], label='Present', color='darkgrey', linestyle = '-')
lineAbs = Line2D([0], [0], label='Absent', color='darkgrey', linestyle = '--')

handles.extend([linePres,lineAbs])

fig.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.245, wspace=0.2)

plt.savefig(os.getcwd()+'/Sensitivity_analysis_samples.png')
fig.show()
# %%
