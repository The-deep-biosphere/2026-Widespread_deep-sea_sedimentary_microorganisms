#%%
# =======================================================
# Comparison classification and regression decision tree 
# Outputs accuracy table for both methods with presence
# and absence of chemicals.
# =======================================================

# Source: https://hands-on.cloud/implementation-of-support-vector-machine-svm-using-python/
# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import composition_stats as cs
import os

# sklearn imports 
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

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

    w = pd.DataFrame(w, columns=['OTU','value'])   # also valid when no OTU is found
    # sort on descending order
    # sort = pd.read_csv(name, names=['OTU','value'])
    sort = w.sort_values(by='value', ascending=False)

    return sort

def DT_features_threshold(rules, threshold):
    
    rules = str(rules)
    # create list of OTUs in rules
    splitted = rules.split('|')

    # analyze which rules are based on rate > 0
    splitted2low = []
    splitted2high = []
    line = []
    rates = []
    for i in range(0,len(splitted)):
        value = splitted[i].split('response:')
        if len(value) == 2:
            rate = float(value[1].strip())
            rates.append(rate)

        else:
            rates.append(0)

    for i in range(0,len(splitted)):

        if rates[i] > threshold:
            splitted2high.append(splitted[i])

        if rates[i] <= threshold:
            splitted2low.append(splitted[i])

    splitted3high = ''.join(splitted2high)
    splitted3low = ''.join(splitted2low)
    splitted4high = splitted3high.split(')')
    splitted4low = splitted3low.split(')')
    features_high = []
    features_low = []

    # get all mentions of OTUs higher threshold
    for i in range(0,len(splitted4high)):
        x = splitted4high[i]

        if len(x.split('(')) > 1:
            x1 = x.split('(')[1]
            x2 = x1.split(' ')[0]
        
        if 'OTU_' in x2:
            features_high.append(x2)

    # get all mentions of OTUs lower threshold
    for i in range(0,len(splitted4low)):
        x = splitted4low[i]

        if len(x.split('(')) > 1:
            x1 = x.split('(')[1]
            x2 = x1.split(' ')[0]
        
        if 'OTU_' in x2:
            features_low.append(x2)

    # count frequency of OTUs in list
    frequency = {}
    for word in features_high:

        count = frequency.get(word,0)
        frequency[word] = count + 1

    # open file for writing, "w" is writing
    w = []

    # loop over dictionary keys and values
    for key, val in frequency.items():
        # write every key and value to file
        w.append([key, val])

    w = pd.DataFrame(w, columns=['OTU','value'])   # also valid when no OTU is found
    # sort on descending order
    # sort = pd.read_csv(name, names=['OTU','value'])
    sort_high = w.sort_values(by='value', ascending=False)

    # count frequency of OTUs in list
    frequency = {}
    for word in features_low:

        count = frequency.get(word,0)
        frequency[word] = count + 1

    # open file for writing, "w" is writing
    w = []

    # loop over dictionary keys and values
    for key, val in frequency.items():
        # write every key and value to file
        w.append([key, val])

    w = pd.DataFrame(w, columns=['OTU','value'])   # also valid when no OTU is found
    # sort on descending order
    # sort = pd.read_csv(name, names=['OTU','value'])
    sort_low = w.sort_values(by='value', ascending=False)

    return sort_high, sort_low

def Data_Extraction (Cores_selection, All_cores, X, y):
        
        data = np.where(np.isin(All_cores, Cores_selection))[0]

        X_selection = X[data,:]
        y_selection = y[data]

        return X_selection,y_selection

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
        elif value <= threshold:
            Zonation[i] = 0
    
    return Zonation


def AP_division(y, y_pred, threshold, perc_absent, number_AMOR_absent, perc_present, number_AMOR_present):

    real = np.array(zonation(y,threshold))
    pred = np.array(zonation(y_pred,threshold))

    loc_high = np.where(real == 1)
    loc_low = np.where(real == 0)

    # append the result of this iteration to the lists of earlier iterations
    number_present = np.sum(real == 1)
    perc_present.append(np.sum(pred[loc_high] == 1)/number_present*100 if number_present > 0 else float('nan'))
    number_AMOR_present.append(number_present)
    number_absent = np.sum(real == 0)
    perc_absent.append(np.sum(pred[loc_low] == 0)/number_absent*100 if number_absent > 0 else float('nan'))
    number_AMOR_absent.append(number_absent)

    return perc_absent, number_AMOR_absent, perc_present, number_AMOR_present, loc_low, loc_high

def AP_renaming(X,y,name):
    Y_pred = dtree_sklearn.predict(X)
    
    P = np.where(y == 1)
    A = np.where(y == 0)
    
    y = np.array(y,dtype='object')

    if len(P[0]) > 0:
        y[P] = np.array(len(y[P])*[f'{name}_P'])
    if len(A[0]) > 0:
        y[A] = np.array(len(y[A])*[f'{name}_A'])

    P = np.where(Y_pred == 1)
    A = np.where(Y_pred == 0)
    
    Y_pred = np.array(Y_pred,dtype='object')

    if len(P[0]) > 0:
        Y_pred[P] = np.array(len(Y_pred[P])*[f'{name}_P'])
    if len(A[0]) > 0:
        Y_pred[A] = np.array(len(Y_pred[A])*[f'{name}_A'])

    return y, Y_pred

#%%
rule_features = []
rule_features_high = []
rule_features_low = []
variables = ['oxygen','nitrate','ammonium','manganese']

dic_name = {'oxygen':'O2conc', 'nitrate':'NO3conc','ammonium':'NH4conc','manganese':'Mnconc'}
# chemical columns used for the significant-OTU selection (as in 4.5)Significant_genes.py)
var_columns = ['O2conc','NO3conc','Mnconc','NH4conc']

zero_values = True
fact = 10
labelsize_ticks = 18
markersize = 18
fs = 15
iter = 1000 # number of training iterations

col = []
y_acc = []
x_acc = []

plt.tick_params(axis='both',which = 'major',labelsize=30)
plt.tick_params(axis='both',which = 'minor',labelsize=30)

# var = variables[0]
for var in variables:
    # ===================
    # Data selection
    # ===================
    data = pd.read_csv(os.getcwd()+"/../1)Preprocessing/4)Significant_gene/Data_prep_ML_FAMILY_datacomparison_clr.csv",
                delimiter=",", dtype=str, index_col=None)
    
    data.columns = data.columns.str.replace(data.columns[0], 'Core')
    data['Core'] = data['Core'].astype(float).astype(int)

    otu_columns = [c for c in data.columns if c.startswith('OTU_')]
    # all samples (every chemical) of the cores of interest, used for the feature selection
    data_fs = data[~data['Core'].isin([18, 19, 21, 22])].copy()
    data_fs[var_columns + otu_columns] = data_fs[var_columns + otu_columns].astype(float)

    if zero_values == False:
        zeros = data[dic_name[var]].values.astype(float) != 0
        data = data.iloc[zeros,:]
    data = data.dropna(subset = [dic_name[var]], how ='any') 

    # remove cores that are not of interest (GS19GC25, GS21GC09, GS20GC20, GS20GC21)
    data = data[data['Core'] != 18]
    data = data[data['Core'] != 19]
    data = data[data['Core'] != 22]
    data = data[data['Core'] != 21]

    # reset the index numbering
    data.reset_index(drop=True, inplace=True)

    # Change to CLR
    y_tot = data[dic_name[var]].values.astype(float)*1000
    y_tot_clr = cs.clr(np.array(y_tot+0.00001))
    data['y'] = (y_tot_clr*fact).astype(int)
    loc_y = data.columns.get_loc('y')

    # Get CLR threshold
    if var == 'nitrate':
        threshold = 5
        val = (data[dic_name[var]].values.astype(float)*1000).astype(int) == threshold
        threshold = data.iloc[val,loc_y].values[0]
    if var == 'oxygen':
        threshold = 5
        val = (data[dic_name[var]].values.astype(float)*1000).astype(int) == threshold
        threshold = data.iloc[val,loc_y].values[0]
    if var == 'manganese':
        threshold = 0
        val = (data[dic_name[var]].values.astype(float)*1000).astype(int) == threshold
        threshold = data.iloc[val,loc_y].values[0]
    if var == 'ammonium':
        threshold = 5
        val = (data[dic_name[var]].values.astype(float)*1000).astype(int) == threshold
        threshold = data.iloc[val,loc_y].values[0]


    # ==================================================
    # ---------- REMOVE sample in TRANSITION -----------
    # ==================================================
    # predictor_loc = np.where(data.columns == 'zonation')[0][0]

    # ===================================
    # Assigning X and Y variables ARCTIC
    # ===================================
    # the significant OTUs are selected inside every iteration from the
    # training cores only (see SPLIT DATA below)
    X_all = data[otu_columns]
    y = data.iloc[:,loc_y].values
    core = np.array(data['Core'])

    # ===================================
    # Resample making equal groups
    # ===================================
    colour_code = np.array(data.Core)
    colour_label = {0:'AMOR',1:"Pacific",2:'Atlantic',3:'North Pond',4:'SCS',5:'SPO'}

    for i,c in enumerate(data.Core):
        # AMOR
        if c <= 25:
            colour_code[i] = 0
        # NWPO
        if c in np.arange(26,40):
            colour_code[i] = 1
        # WNAG
        if c in np.arange(40,43):
            # store atlantic in another variable (40)
            colour_code[i] = 2
        # MAR
        if c in np.arange(43,45):
            # store north pond in another variable (41 and 42)
            colour_code[i] = 3
        # SCS
        if c in np.arange(45,55):
            # store south china sea in another variable (45 up to 55)
            colour_code[i] = 4
        # SPO
        if c in np.arange(55,68):
            # store south pacific ocean in another variable (56 up to 68)
            colour_code[i] = 5

    iterations = iter
    # =====================================
    # ------------ REGRESSION -------------
    # =====================================
    
    # accuracy values over X iterations
    perc_AMOR_absent = []
    perc_AMOR_present = []
    perc_MAR_absent = [] 
    perc_MAR_present = [] 
    perc_NWPO_absent = []
    perc_NWPO_present = []
    perc_WNAG_absent = []
    perc_WNAG_present = []
    perc_SCS_absent = []
    perc_SCS_present = []
    perc_SPO_present = []
    perc_SPO_absent = []

    # amount of samples used for accuracy calculation
    number_AMOR_absent = []
    number_AMOR_present = []
    number_MAR_absent = [] 
    number_MAR_present = [] 
    number_NWPO_absent = []
    number_NWPO_present = []
    number_WNAG_absent = []
    number_WNAG_present = []
    number_SCS_absent = []
    number_SCS_present = []
    number_SPO_absent = []
    number_SPO_present = []

    splits = []
    for it in range(iterations):

        # =================================
        # ---------- SPLIT DATA -----------
        # =================================

        seed = 651 + it               # new, reproducible split in every iteration
        # train / validation / test split (0.7/0.15/0.15 of the cores),
        # stratified on area and then on core: a core is in only one set
        train_idx, val_idx, test_idx = area_core_split(core, seed)

        # core GS19GC10 (core == 17) represents the dataset the most (207 samples);
        # remove 1/4 of its samples from the training set to reduce its weight
        train_idx = downsample_core(train_idx, core, seed)

        # load significant features tested using Storey's q-value, computed on the training cores only
        feature_selection = select_features(data_fs, np.unique(core[train_idx]), var_columns,
                                            otu_columns, seed, clr=False)
        X = X_all[feature_selection].values.astype(float)
        # the classification below re-uses the same splits and features
        splits.append((train_idx, val_idx, test_idx, feature_selection))

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test, core_test = X[test_idx], y[test_idx], core[test_idx]
        
        # Classifier presence/absence based on threshold
        y_zonation_test = np.array(zonation(y_test,threshold))
        y_zonation_train = np.array(zonation(y_train,threshold))


        # ===========================================
        # ---------- TRAIN AND TEST MODEL -----------
        # ===========================================

        # ------------------------------------
        # Test choosen model on test dataset
        # ------------------------------------

        # max_depth is chosen on the validation cores (ccp_alpha = 0 as before)
        dtree_sklearn = fit_tuned_tree(DecisionTreeRegressor(ccp_alpha=0, random_state=seed),
                                       X_train, y_train, X_val, y_val)


        print("============ Train data ============")
        y_pred_train = dtree_sklearn.predict(X_train)

        MSE = mean_squared_error(y_train,y_pred_train)
        MSE1 = mean_absolute_error(y_train,y_pred_train)

        print("y - y.pred: "+str(math.sqrt(MSE)))
        print("MSE: "+str((MSE)))
        print("mean absolute error: "+str((MSE1)))



        print("============ All together data ============")
        y_pred_test = dtree_sklearn.predict(X_test)

        MSE = mean_squared_error(y_test,y_pred_test)
        MSE1 = mean_absolute_error(y_test,y_pred_test)

        print("y - y.pred: "+str(math.sqrt(MSE)))
        print("MSE: "+str((MSE)))
        print("mean absolute error: "+str((MSE1)))


        # ================================
        # ---- Extract core locations ----
        # ================================

        y_test = np.array(y_test)

        # --------------
        # -- PACIFIC ---
        # --------------

        # store NWPO in another variable
        # and remove 26 - 39 from test Data
        NWPO = AREA_CORES['NWPO']
        X_NWPO, y_NWPO = Data_Extraction(NWPO, core_test, X_test, y_test)

        # -- ATLANTIC --
        # --------------

        # store WNAG in another variable (40)
        WNAG_data = AREA_CORES['WNAG']
        X_WNAG, y_WNAG = Data_Extraction(WNAG_data, core_test, X_test, y_test)

        # --------------
        # - NORTH POND -
        # --------------

        # store north pond in another variable (43 and 44)
        MAR_data = AREA_CORES['MAR']
        X_MAR, y_MAR = Data_Extraction(MAR_data, core_test, X_test, y_test)

        # --------------
        # ---- SCS -----
        # --------------

        # store south china sea in another variable (45 up to 54)
        SCS = AREA_CORES['SCS']
        X_SCS, y_SCS = Data_Extraction(SCS, core_test, X_test, y_test)

        # --------------
        # ---- SPO -----
        # --------------

        # store South Pacific Ocean in another variable (55 up to 67)
        SPO = AREA_CORES['SPO'] # cores >= 63 are Atacama
        X_SPO, y_SPO = Data_Extraction(SPO, core_test, X_test, y_test)

        # --------------
        # ---- AMOR ----
        # --------------

        AMOR = AREA_CORES['AMOR']
        X_AMOR, y_AMOR = Data_Extraction(AMOR, core_test, X_test, y_test)

    
        if var == 'oxygen' or var == 'nitrate':
            # ==============================
            # ----- Predict North Pond -----
            # ==============================
            print("============ North Pond ============")
            y_pred_MAR = dtree_sklearn.predict(X_MAR)

            MSE = mean_squared_error(y_MAR,y_pred_MAR)
            MSE1 = mean_absolute_error(y_MAR,y_pred_MAR)

            print("y - y.pred: "+str(math.sqrt(MSE)))
            print("MSE: "+str((MSE)))
            print("mean absolute error: "+str((MSE1)))

            perc_MAR_absent, number_MAR_absent, perc_MAR_present, number_MAR_present, loc_low_MAR, loc_high_MAR = AP_division(y_MAR, y_pred_MAR, threshold, perc_MAR_absent, number_MAR_absent, perc_MAR_present, number_MAR_present)

        if  len(y_NWPO) > 0:
            # ============================
            # ----- Predict Pacific -----
            # ============================
        
            print("============ PACIFIC ============")
            y_pred_NWPO = dtree_sklearn.predict(X_NWPO)

            MSE = mean_squared_error(y_NWPO,y_pred_NWPO)
            MSE1 = mean_absolute_error(y_NWPO,y_pred_NWPO)

            print("y - y.pred: "+str(math.sqrt(MSE)))
            print("MSE: "+str((MSE)))
            print("mean absolute error: "+str((MSE1)))

            perc_NWPO_absent, number_NWPO_absent, perc_NWPO_present, number_NWPO_present, loc_low_NWPO, loc_high_NWPO = AP_division(y_NWPO, y_pred_NWPO, threshold, perc_NWPO_absent, number_NWPO_absent, perc_NWPO_present, number_NWPO_present)
        else: 
            y_pred_NWPO = float('nan')
            perc_NWPO_absent, number_NWPO_absent, perc_NWPO_present, number_NWPO_present, loc_low_NWPO, loc_high_NWPO = float('nan'), 0, float('nan'), 0, float('nan'), float('nan')

        if var == 'manganese' or var == 'nitrate' or var == 'ammonium' or var == 'oxygen':

            # ============================
            # ------- Predict AMOR -------
            # ============================
            print("============ AMOR ============")
            y_pred_AMOR = dtree_sklearn.predict(X_AMOR)


            MSE = mean_squared_error(y_AMOR,y_pred_AMOR)
            MSE1 = mean_absolute_error(y_AMOR,y_pred_AMOR)

            print("y - y.pred: "+str(math.sqrt(MSE)))
            print("MSE: "+str((MSE)))
            print("mean absolute error: "+str((MSE1)))

            perc_AMOR_absent, number_AMOR_absent, perc_AMOR_present, number_AMOR_present, loc_low_AMOR, loc_high_AMOR = AP_division(y_AMOR, y_pred_AMOR, threshold, perc_AMOR_absent, number_AMOR_absent, perc_AMOR_present, number_AMOR_present)


            # ============================
            # ----- Predict SCS -----
            # ============================

            print("============ SCS ============")
            
            if zero_values or var != 'oxygen':
                y_pred_SCS = dtree_sklearn.predict(X_SCS)

                MSE = mean_squared_error(y_SCS,y_pred_SCS)
                MSE1 = mean_absolute_error(y_SCS,y_pred_SCS)

                print("y - y.pred: "+str(math.sqrt(MSE)))
                print("MSE: "+str((MSE)))
                print("mean absolute error: "+str((MSE1)))

                perc_SCS_absent, number_SCS_absent, perc_SCS_present, number_SCS_present, loc_low_SCS, loc_high_SCS = AP_division(y_SCS, y_pred_SCS, threshold, perc_SCS_absent, number_SCS_absent, perc_SCS_present, number_SCS_present)

            else: 
                y_pred_SCS = float('nan')
                perc_SCS_absent, number_SCS_absent, perc_SCS_present, number_SCS_present, loc_low_SCS, loc_high_SCS = float('nan'), 0, float('nan'), 0, float('nan'), float('nan')

        # =======================
        # ----- Predict SPO -----
        # =======================
        print("============ SPO ============")
        
        if var == 'nitrate' or var == 'ammonium' or var == 'oxygen' or var == 'manganese':
            if len(y_SPO) > 0:
                y_pred_SPO = dtree_sklearn.predict(X_SPO)

                MSE = mean_squared_error(y_SPO,y_pred_SPO)
                MSE1 = mean_absolute_error(y_SPO,y_pred_SPO)

                print("y - y.pred: "+str(math.sqrt(MSE)))
                print("MSE: "+str((MSE)))
                print("mean absolute error: "+str((MSE1)))

                perc_SPO_absent, number_SPO_absent, perc_SPO_present, number_SPO_present, loc_low_SPO, loc_high_SPO = AP_division(y_SPO, y_pred_SPO, threshold, perc_SPO_absent, number_SPO_absent, perc_SPO_present, number_SPO_present)

            else: 
                y_pred_SPO = float('nan')
                perc_SPO_absent, number_SPO_absent, perc_SPO_present, number_SPO_present, loc_low_SPO, loc_high_SPO = float('nan'), 0, float('nan'), 0, float('nan'), float('nan')


        # ============================
        # ----- Predict Atlantic -----
        # ============================

        if var == 'oxygen':
            print("============ Atlantic ============")
            y_pred_WNAG = dtree_sklearn.predict(X_WNAG)

            MSE = mean_squared_error(y_WNAG,y_pred_WNAG)
            MSE1 = mean_absolute_error(y_WNAG,y_pred_WNAG)

            print("y - y.pred: "+str(math.sqrt(MSE)))
            print("MSE: "+str((MSE)))
            print("mean absolute error: "+str((MSE1)))

            perc_WNAG_absent, number_WNAG_absent, perc_WNAG_present, number_WNAG_present, loc_low_WNAG, loc_high_WNAG = AP_division(y_WNAG, y_pred_WNAG, threshold, perc_WNAG_absent, number_WNAG_absent, perc_WNAG_present, number_WNAG_present)


    # =====================================
    # ----- SAVING RESULTS REGRESSION -----
    # =====================================
    # The percentages are generated for the number of iterations i of training
    # Now we want to plot the results

    # mean accuracy percentage
    perc_AMOR_absent = np.round(np.nanmean(perc_AMOR_absent),1)
    perc_AMOR_present = np.round(np.nanmean(perc_AMOR_present),1)
    perc_WNAG_absent = np.round(np.nanmean(perc_WNAG_absent),1)
    perc_WNAG_present = np.round(np.nanmean(perc_WNAG_present),1)
    perc_MAR_absent = np.round(np.nanmean(perc_MAR_absent),1)
    perc_MAR_present = np.round(np.nanmean(perc_MAR_present),1)
    perc_NWPO_absent = np.round(np.nanmean(perc_NWPO_absent),1)
    perc_NWPO_present = np.round(np.nanmean(perc_NWPO_present),1)
    perc_SCS_present = np.round(np.nanmean(perc_SCS_present),1)
    perc_SCS_absent = np.round(np.nanmean(perc_SCS_absent),1)
    perc_SPO_present = np.round(np.nanmean(perc_SPO_present),1)
    perc_SPO_absent = np.round(np.nanmean(perc_SPO_absent),1)

    # mean number of samples used 
    number_AMOR_absent = np.round(np.nanmean(number_AMOR_absent))
    number_AMOR_present = np.round(np.nanmean(number_AMOR_present))

    number_WNAG_absent = np.round(np.nanmean(number_WNAG_absent))
    number_WNAG_present = np.round(np.nanmean(number_WNAG_present))

    number_MAR_absent = np.round(np.nanmean(number_MAR_absent))
    number_MAR_present = np.round(np.nanmean(number_MAR_present))

    number_NWPO_absent = np.round(np.nanmean(number_NWPO_absent))
    number_NWPO_present = np.round(np.nanmean(number_NWPO_present))

    number_SCS_absent = np.round(np.nanmean(number_SCS_absent))
    number_SCS_present = np.round(np.nanmean(number_SCS_present))

    number_SPO_absent = np.round(np.nanmean(number_SPO_absent))
    number_SPO_present = np.round(np.nanmean(number_SPO_present))

    if var == 'oxygen':
        RG_absence_O2 = np.array([perc_AMOR_absent,perc_WNAG_absent,perc_MAR_absent,perc_NWPO_absent,perc_SPO_absent,perc_SCS_absent])
        RG_presence_O2 = np.array([perc_AMOR_present,perc_WNAG_present,perc_MAR_present,perc_NWPO_present,perc_SPO_present,perc_SCS_present])
        RG_N_ab_O2 = np.array([number_AMOR_absent, number_WNAG_absent, number_MAR_absent, number_NWPO_absent, number_SPO_absent, number_SCS_absent])
        RG_N_pre_O2 = np.array([number_AMOR_present, number_WNAG_present, number_MAR_present, number_NWPO_present, number_SPO_present, number_SCS_present])
        RG_N_pre_O2[np.isnan(RG_N_pre_O2)] = 0
        RG_N_ab_O2[np.isnan(RG_N_ab_O2)] = 0
        RG_N_pre_O2 = RG_N_pre_O2.astype(int)
        RG_N_ab_O2 = RG_N_ab_O2.astype(int)

    if var == 'nitrate':
        RG_absence_NOx = np.array([perc_AMOR_absent,perc_WNAG_absent,perc_MAR_absent,perc_NWPO_absent,perc_SPO_absent,perc_SCS_absent])
        RG_presence_NOx = np.array([perc_AMOR_present,perc_WNAG_present,perc_MAR_present,perc_NWPO_present,perc_SPO_present,perc_SCS_present])
        RG_N_ab_NOx = np.array([number_AMOR_absent, number_WNAG_absent, number_MAR_absent, number_NWPO_absent, number_SPO_absent, number_SCS_absent])
        RG_N_pre_NOx = np.array([number_AMOR_present, number_WNAG_present, number_MAR_present, number_NWPO_present, number_SPO_present, number_SCS_present])
        RG_N_pre_NOx[np.isnan(RG_N_pre_NOx)] = 0
        RG_N_ab_NOx[np.isnan(RG_N_ab_NOx)] = 0
        RG_N_pre_NOx = RG_N_pre_NOx.astype(int)
        RG_N_ab_NOx = RG_N_ab_NOx.astype(int)

    if var == 'ammonium':
        RG_absence_NH4 = np.array([perc_AMOR_absent,perc_WNAG_absent,perc_MAR_absent,perc_NWPO_absent,perc_SPO_absent,perc_SCS_absent])
        RG_presence_NH4 = np.array([perc_AMOR_present,perc_WNAG_present,perc_MAR_present,perc_NWPO_present,perc_SPO_present,perc_SCS_present])
        RG_N_ab_NH4 = np.array([number_AMOR_absent, number_WNAG_absent, number_MAR_absent, number_NWPO_absent, number_SPO_absent, number_SCS_absent])
        RG_N_pre_NH4 = np.array([number_AMOR_present, number_WNAG_present, number_MAR_present, number_NWPO_present, number_SPO_present, number_SCS_present])
        RG_N_pre_NH4[np.isnan(RG_N_pre_NH4)] = 0
        RG_N_ab_NH4[np.isnan(RG_N_ab_NH4)] = 0
        RG_N_pre_NH4 = RG_N_pre_NH4.astype(int)
        RG_N_ab_NH4 = RG_N_ab_NH4.astype(int)

    if var == 'manganese':
        RG_absence_Mn = np.array([perc_AMOR_absent,perc_WNAG_absent,perc_MAR_absent,perc_NWPO_absent,perc_SPO_absent,perc_SCS_absent])
        RG_presence_Mn = np.array([perc_AMOR_present,perc_WNAG_present,perc_MAR_present,perc_NWPO_present,perc_SPO_present,perc_SCS_present])
        RG_N_ab_Mn = np.array([number_AMOR_absent, number_WNAG_absent, number_MAR_absent, number_NWPO_absent, number_SPO_absent, number_SCS_absent])
        RG_N_pre_Mn = np.array([number_AMOR_present, number_WNAG_present, number_MAR_present, number_NWPO_present, number_SPO_present, number_SCS_present])
        RG_N_pre_Mn[np.isnan(RG_N_pre_Mn)] = 0
        RG_N_ab_Mn[np.isnan(RG_N_ab_Mn)] = 0
        RG_N_pre_Mn = RG_N_pre_Mn.astype(int)
        RG_N_ab_Mn = RG_N_ab_Mn.astype(int)

        RG_presence = np.array([RG_presence_O2,RG_presence_NOx,RG_presence_NH4,RG_presence_Mn,[f'{RG_N_pre_O2[n]}/{RG_N_pre_NOx[n]}/{RG_N_pre_NH4[n]}/{RG_N_pre_Mn[n]}' for n in range(len(RG_N_pre_O2))]])
        RG_absence = np.array([RG_absence_O2,RG_absence_NOx,RG_absence_NH4,RG_absence_Mn,[f'{RG_N_ab_O2[n]}/{RG_N_ab_NOx[n]}/{RG_N_ab_NH4[n]}/{RG_N_ab_Mn[n]}' for n in range(len(RG_N_ab_O2))]])


    # =====================================
    # ---------- CLASSIFICATION -----------
    # =====================================


    cm_iterations = []   # confusion-matrix based accuracy of every iteration
    # rule frequencies of this chemical, counted over all iterations of this run
    rule_features, rule_features_high, rule_features_low = [], [], []

    for it in range(iterations):
        # same train / validation / test cores and features as the regression
        train_idx, val_idx, test_idx, feature_selection = splits[it]
        seed = 651 + it
        X = X_all[feature_selection].values.astype(float)
        X_train, y_zonation_train = X[train_idx], np.array(zonation(y[train_idx],threshold))
        X_val, y_zonation_val = X[val_idx], np.array(zonation(y[val_idx],threshold))
        X_test, y_zonation_test, core_test = X[test_idx], np.array(zonation(y[test_idx],threshold)), core[test_idx]

        # store every area of the test cores in another variable
        NWPO_data = np.where(np.isin(core_test, AREA_CORES['NWPO']))[0]
        X_NWPO = X_test[NWPO_data,:]
        y_NWPO = y_zonation_test[NWPO_data]

        WNAG_data = np.where(np.isin(core_test, AREA_CORES['WNAG']))[0]
        X_WNAG = X_test[WNAG_data,:]
        y_WNAG = y_zonation_test[WNAG_data]

        # north pond (MAR)
        NP_data = np.where(np.isin(core_test, AREA_CORES['MAR']))[0]
        X_NP = X_test[NP_data,:]
        y_NP = y_zonation_test[NP_data]

        SCS_data = np.where(np.isin(core_test, AREA_CORES['SCS']))[0]
        X_SCS = X_test[SCS_data,:]
        y_SCS = y_zonation_test[SCS_data]

        SPO_data = np.where(np.isin(core_test, AREA_CORES['SPO']))[0]
        X_SPO = X_test[SPO_data,:]
        y_SPO = y_zonation_test[SPO_data]

        AMOR_data = np.where(np.isin(core_test, AREA_CORES['AMOR']))[0]
        X_AMOR = X_test[AMOR_data,:]
        y_AMOR = y_zonation_test[AMOR_data]

        # ===========================================
        # ---------- TRAIN AND TEST MODEL -----------
        # ===========================================

        # ------------------------------------
        # Test choosen model on test dataset
        # ------------------------------------
        # max_depth is chosen on the validation cores (ccp_alpha = 0 as before)
        dtree_sklearn = fit_tuned_tree(DecisionTreeClassifier(criterion='gini', ccp_alpha=0, random_state=seed),
                                       X_train, y_zonation_train, X_val, y_zonation_val)

        print("============ Train data ============")
        y_pred_train = dtree_sklearn.predict(X_train)

        print("============ All together data ============")
        y_pred_test = dtree_sklearn.predict(X_test)


        # Save important features
        rules = get_rules(dtree_sklearn, feature_selection, None) 
        
        if len(rule_features) == 0:
            rule_features = DT_Features(rules)
            rule_features_high, rule_features_low = DT_features_threshold(rules, 0)

        else:
            rule_features = pd.concat([rule_features,DT_Features(rules)])
            h, l = DT_features_threshold(rules, 0)
            rule_features_high, rule_features_low = pd.concat([rule_features_high,h]), pd.concat([rule_features_low,l])
    

        # ===================
        # RENAMING OUTPUT
        # ===================
        # -------------
        # AMOR
        # -------------
        print("AMOR")

        y_AMOR, y_AMOR_pred = AP_renaming(X_AMOR,y_AMOR,'AMOR')
    
        # -------------
        # Atlantic (WNAG)
        # -------------
        print("WNAG")
        if var == 'oxygen':
            y_WNAG, y_WNAG_pred = AP_renaming(X_WNAG,y_WNAG,'WNAG')
        else:
            y_WNAG_pred = []
            y_WNAG = []

        # ----------------
        # North Pond (MAR)
        # ----------------
        print("MAR")
        if (var == 'oxygen') | (var == 'nitrate'):
            y_NP, y_NP_pred = AP_renaming(X_NP,y_NP,'MAR')
        else:
            y_NP_pred = []
            y_NP = []

        # -------------
        # Pacific (NWPO)
        # -------------
        print("NWPO")
        if (var == 'oxygen') | (var == 'nitrate')| (var == 'ammonium'):
            y_NWPO, y_NWPO_pred = AP_renaming(X_NWPO,y_NWPO,'NWPO')
        else:
            y_NWPO, y_NWPO_pred = [], []
        
        # -------------
        # SCS
        # -------------
        print("SCS")
        y_SCS, y_SCS_pred = AP_renaming(X_SCS,y_SCS,'SCS')
        
        # -------------
        # Pacific (SPO)
        # -------------
        print("SPO")
        if (var == 'oxygen') | (var == 'nitrate') | (var == 'ammonium') | (var == 'manganese') :
            y_SPO, y_SPO_pred = AP_renaming(X_SPO,y_SPO,'SPO')
        else:
            y_SPO, y_SPO_pred = [], []

        y_pred_all = np.concatenate([y_AMOR_pred,y_WNAG_pred,y_NP_pred,y_NWPO_pred,y_SCS_pred,y_SPO_pred])
        y_all = np.concatenate([y_AMOR,y_WNAG,y_NP,y_NWPO,y_SCS,y_SPO])

        labels =  ['AMOR_P','AMOR_A','WNAG_P','WNAG_A','MAR_P','MAR_A','NWPO_P','NWPO_A','SPO_P','SPO_A','SCS_P','SCS_A']#list(np.unique(y_pred_all))
        cm = confusion_matrix(y_all, y_pred_all, labels = labels)
        acc = cm.diagonal()/cm.sum(axis=1)#accuracy_score(y_all, y_pred_all)
        # rows: True labels, columns: Predicted labels
        cm = pd.DataFrame({'Accuracy':acc,'N_sampels':cm.sum(axis=1)})
        cm.index = labels

        # DTC_class_O2
        # df_concat =pd.concat((DTC_class_O2,DTC_class_O2))
        # df_concat.mean()
        # df_concat.groupby(['A', 'B']).mean()
        # df_concat = df_concat.groupby(df_concat.index)

        cm_iterations.append(cm)

    # save the OTU frequencies in the rules of all classification trees of this run
    # (own file names: the regression trees of continuous_prediction_errordistr.py
    # write feature_frequency_rules_{var}_*.csv)
    name = ['general','high','low']
    for n,i in enumerate([rule_features,rule_features_high,rule_features_low]):
        rule_features2 = i.groupby(["OTU"]).sum()
        rule_features2 = rule_features2.sort_values(by='value', ascending=False)
        rule_features2 = rule_features2.reset_index()
        np.savetxt(os.getcwd()+f'/rules_extract/feature_frequency_rules_classifier_{var}_{name[n]}.csv', rule_features2, delimiter=",", fmt="%s",header = 'OTU,value')

    # mean over the iterations (areas/classes without test samples are ignored)
    DTC_class = pd.concat(cm_iterations).groupby(level=0, sort=False).mean()

    if var == 'oxygen':
        DTC_class_O2 = DTC_class

    if var == 'nitrate':
        DTC_class_NO3 = DTC_class

    if var == 'ammonium':
        DTC_class_NH4 = DTC_class

    if var == 'manganese':
        DTC_class_Mn = DTC_class


DTC_presence_O2,DTC_presence_NOx,DTC_presence_NH4,DTC_presence_Mn = [], [], [], []
DTC_absence_O2,DTC_absence_NOx,DTC_absence_NH4,DTC_absence_Mn = [], [], [], []
DTC_N_pre_O2, DTC_N_pre_NOx, DTC_N_pre_NH4, DTC_N_pre_Mn = [], [], [], []
DTC_N_ab_O2, DTC_N_ab_NOx, DTC_N_ab_NH4, DTC_N_ab_Mn = [], [], [], []

DTC_presence = [DTC_presence_O2,DTC_presence_NOx,DTC_presence_NH4,DTC_presence_Mn]
DTC_absence = [DTC_absence_O2,DTC_absence_NOx,DTC_absence_NH4,DTC_absence_Mn]
DTC_N_pre = [DTC_N_pre_O2, DTC_N_pre_NOx, DTC_N_pre_NH4, DTC_N_pre_Mn]
DTC_N_ab = [DTC_N_ab_O2, DTC_N_ab_NOx, DTC_N_ab_NH4, DTC_N_ab_Mn]

for r, result in enumerate([DTC_class_O2, DTC_class_NO3, DTC_class_NH4, DTC_class_Mn]):
    for n, ind in enumerate(result.index):
        if ind.endswith('_P'):
            # if result.index[n][:-2] == 'AMOR':
            DTC_presence[r].append(np.round(result.iloc[n,0]*100,1))
            DTC_N_pre[r].append(result.iloc[n,1].astype(int))

        if ind.endswith('_A'):
            # if result.index[n][:-2] == 'AMOR':
            DTC_absence[r].append(np.round(result.iloc[n,0]*100,1))
            DTC_N_ab[r].append(result.iloc[n,1].astype(int))

DTC_presence = np.array([DTC_presence_O2,DTC_presence_NOx,DTC_presence_NH4,DTC_presence_Mn,[f'{DTC_N_pre_O2[n]}/{DTC_N_pre_NOx[n]}/{DTC_N_pre_NH4[n]}/{DTC_N_pre_Mn[n]}' for n in range(len(DTC_N_pre_O2))]])
DTC_absence = np.array([DTC_absence_O2,DTC_absence_NOx,DTC_absence_NH4,DTC_absence_Mn,[f'{DTC_N_ab_O2[n]}/{DTC_N_ab_NOx[n]}/{DTC_N_ab_NH4[n]}/{DTC_N_ab_Mn[n]}' for n in range(len(DTC_N_ab_O2))]])


# columns = ['Absent (%)', 'Present (%)', '# samples A/P'] # , # samples
# rows = ['AMOR', 'WNAG', 'MAR', 'NWPO', 'SCS', 'SPO']

fig, ax = plt.subplots(2,2,figsize=(20, 20))

rows = ["AMOR", "WNAG", "MAR", "NWPO", "SPO", "SCS"]
columns = [f'$O_2$', f'$NO_3$', f'$NH_4$', f'$Mn$', f'Number of Samples']
height_columns = .4
# --------------------
# Presence Regression
# --------------------
df = pd.DataFrame(np.transpose(RG_presence), columns=columns)

ax[0,0].set_frame_on(False) # turn off frame for the table subplot
ax[0,0].set_xticks([]) # turn off x ticks for the table subplot
ax[0,0].set_yticks([]) # turn off y ticks for the table subplot
# tabcolours = [[1,0,0,.5],[1,128/255,0,.5],[0,128/255,0,.5],[0,0,1,.5],[139/255,34/255,82/255,.5],[9/255,255/255,255/255,.5]]
tabcolours = [[1,1,1],[211/255, 211/255, 211/255],[1,1,1],[211/255, 211/255, 211/255],[1,1,1],[211/255, 211/255, 211/255]]
the_table = ax[0,0].table(cellText=df.values,colLabels=df.columns, 
                                rowColours=tabcolours, 
                            #  colWidths=[2 for x in df.columns],
                                colColours=[[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255]],
                                rowLabels=rows,bbox = [.3, 0.2, 1.5, height_columns])
the_table.auto_set_font_size(False)
the_table.set_fontsize(18)
the_table.auto_set_column_width(col=list([4])) # Provide integer list of columns to adjust
ax[0,0].set_title('Presence')


# --------------------
# Absence Regression
# --------------------
df = pd.DataFrame(np.transpose(RG_absence), columns=columns)

ax[0,1].set_frame_on(False) # turn off frame for the table subplot
ax[0,1].set_xticks([]) # turn off x ticks for the table subplot
ax[0,1].set_yticks([]) # turn off y ticks for the table subplot
the_table = ax[0,1].table(cellText=df.values,colLabels=df.columns, 
                                rowColours=tabcolours, 
                                colColours=[[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255]],
                                rowLabels=rows, bbox = [.3, 0.2, 1.5, height_columns])
the_table.auto_set_font_size(False)
the_table.set_fontsize(18)
the_table.auto_set_column_width(col=list([4])) # Provide integer list of columns to adjust
ax[0,1].set_title('Absence')


# --------------------
# Presence Classifier
# --------------------
df = pd.DataFrame(np.transpose(DTC_presence), columns=columns)

ax[1,0].set_frame_on(False) # turn off frame for the table subplot
ax[1,0].set_xticks([]) # turn off x ticks for the table subplot
ax[1,0].set_yticks([]) # turn off y ticks for the table subplot
# tabcolours = [[1,0,0,.5],[1,128/255,0,.5],[0,128/255,0,.5],[0,0,1,.5],[139/255,34/255,82/255,.5],[9/255,255/255,255/255,.5]]
tabcolours = [[1,1,1],[211/255, 211/255, 211/255],[1,1,1],[211/255, 211/255, 211/255],[1,1,1],[211/255, 211/255, 211/255]]
the_table = ax[1,0].table(cellText=df.values,colLabels=df.columns, 
                                rowColours=tabcolours, 
                            #  colWidths=[2 for x in df.columns],
                                colColours=[[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255]],
                                rowLabels=rows,bbox = [.3, 0.2, 1.5, height_columns])
the_table.auto_set_font_size(False)
the_table.set_fontsize(18)
the_table.auto_set_column_width(col=list([4])) # Provide integer list of columns to adjust
ax[1,0].set_title('Presence')


# --------------------
# Absence Classifier
# --------------------
df = pd.DataFrame(np.transpose(DTC_absence), columns=columns)

ax[1,1].set_frame_on(False) # turn off frame for the table subplot
ax[1,1].set_xticks([]) # turn off x ticks for the table subplot
ax[1,1].set_yticks([]) # turn off y ticks for the table subplot
the_table = ax[1,1].table(cellText=df.values,colLabels=df.columns, 
                                rowColours=tabcolours, 
                                colColours=[[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255],[128/255,128/255,128/255]],
                                rowLabels=rows, bbox = [.3, 0.2, 1.5, height_columns])
the_table.auto_set_font_size(False)
the_table.set_fontsize(18)
the_table.auto_set_column_width(col=list([4])) # Provide integer list of columns to adjust
ax[1,1].set_title('Absence')

# ax[0,ax_n].text(-0.15, 1.05, string.ascii_uppercase[3*ax_n], transform=ax[0,ax_n].transAxes, 
#         size=30, weight='bold')
# ax[1,ax_n].text(-0.15, 1.05, string.ascii_uppercase[3*ax_n+1], transform=ax[1,ax_n].transAxes, 
#         size=30, weight='bold')
# # ax[4,ax_n].text(-0.15, 1.05, string.ascii_uppercase[4*ax_n+2], transform=ax[4,ax_n].transAxes, 
# #         size=30, weight='bold')
# ax[0,ax_n].text(-0.15, 1.05, string.ascii_uppercase[3*ax_n+2], transform=ax[0,ax_n].transAxes, 
#         size=30, weight='bold')

plt.subplots_adjust(top=0.99,
bottom=0.095,
left=0.01,
right=0.7,
hspace=0.02,
wspace=1.0)
# # plt.savefig('DT_regressor_results2.png')
# plt.tight_layout()

plt.savefig(os.getcwd()+f'/DT_comparison_{iterations}it.png')

plt.show()

# %%
