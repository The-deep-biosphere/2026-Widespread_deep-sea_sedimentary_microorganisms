#%%
# ==================================================================================
# Regression decision tree 
# Accuracy of regression decision tree plot predicted against actual concentration
# with plots of error distribution underneath
# (First results figure) 
# ===================================================================================

# Source: https://hands-on.cloud/implementation-of-support-vector-machine-svm-using-python/
# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
import math
import string
import composition_stats as cs
from scipy.stats import pearsonr

# sklearn imports
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree
from sklearn.metrics import r2_score
import os

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
    splitted = rules.split('>')
    features = []

    # get all mentions of OTUs
    for i in range(0,len(splitted)-1):
        x = splitted[i]

        if x[len(x)-1] == ' ':
            x1 = x.split('(')
            x1 = x1[len(x1)-1]
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
    splitted4high = splitted3high.split('>')
    splitted4low = splitted3low.split('>')
    features_high = []
    features_low = []

    # get all mentions of OTUs higher threshold
    for i in range(0,len(splitted4high)):
        x = splitted4high[i]

        if x[len(x)-1] == ' ':
            x1 = x.split('(')
            x1 = x1[len(x1)-1]
            x2 = x1.split(' ')[0]
        
        if 'OTU_' in x2:
            features_high.append(x2)

    # get all mentions of OTUs lower threshold
    # take only the OTUs that are present indicating low threshold
    for i in range(0,len(splitted4low)):
        x = splitted4low[i]

        if x[len(x)-1] == ' ':
            x1 = x.split('(')
            x1 = x1[len(x1)-1]
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

def AP_division(y, y_pred, threshold, perc_absent, number_AMOR_absent, perc_present, number_AMOR_present):
    loc_low = np.where(y <= threshold)[0]
    
    if len(loc_low) == 0:
        perc_absent.append(float('nan'))
        number_AMOR_absent.append(len(loc_low))
    else:
        perc_absent.append(round(sum(y_pred[loc_low] <= threshold)/len(y[loc_low])*100,1))
        number_AMOR_absent.append(len(loc_low))

    loc_high = np.where(y > threshold)[0]
    if len(loc_high) == 0:
        perc_present.append(float('nan'))
        number_AMOR_present.append(len(loc_high))
    else:
        perc_present.append(sum(y_pred[loc_high] > threshold)/len(y[loc_high])*100)
        number_AMOR_present.append(len(loc_high))
    
    return perc_absent, number_AMOR_absent, perc_present, number_AMOR_present, loc_low, loc_high

rule_features = []
rule_features_high = []
rule_features_low = []
variables = ['oxygen','nitrate','ammonium','manganese']

# target columns are selected by name (Data_prep_ML_FAMILY_datacomparison.csv)
dic_name = {'oxygen':'O2conc', 'nitrate':'NOxconc','ammonium':'NH4conc','manganese':'Mnconc'}
# chemical columns used for the significant-OTU selection (as in 4.5)Significant_genes.py)
var_columns = ['O2conc','NOxconc','Mnconc','NH4conc']

zero_values = True
fact = 1000
fig, ax = plt.subplots(2,4,figsize=(15, 8))
labelsize_ticks = 18
markersize = 30
fs = 15

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
    data = pd.read_csv(os.getcwd()+"/../1)Preprocessing/4)Significant_gene/Data_prep_ML_FAMILY_datacomparison.csv",
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

    y_tot = data[dic_name[var]].values.astype(float)*1000
    y_tot_clr = cs.clr(np.array(y_tot+0.001))
    data['y'] = y_tot_clr
    loc_y = data.columns.get_loc('y')

    # ===================================
    # Assigning X and Y variables 
    # ===================================
    # the significant OTUs are selected inside every iteration from the
    # training cores only (see SPLIT DATA below)
    X_all = data[otu_columns]
    y = (data.iloc[:,loc_y]*fact).astype(int).values
    core = data['Core'].values


    iterations = 1000
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

    # overall
    perc_test_absent, number_test_absent, perc_test_present, number_test_present = [], [], [], []


    # ------------------------------------------
    # Set threshold absence presence chemical
    # ------------------------------------------
    threshold = 5 # uM, for every chemical

    # the CLR value (times fact) that corresponds to the threshold concentration,
    # using the same transformation as y: clr(x) = log(x + 0.001) - mean(log(y_tot + 0.001))
    threshold = (np.log(threshold + 0.001) - np.mean(np.log(y_tot + 0.001)))*fact

    # rule frequencies of this chemical, counted over all iterations of this run
    rule_features, rule_features_high, rule_features_low = [], [], []
    path = os.getcwd()+"/rules_extract/"

    # for the plots: one out-of-sample prediction per sample, taken from the first
    # iteration in which its core was a test core
    first_test_pred = np.full(len(y), np.nan)

    for i in range(iterations):

        # =================================
        # ---------- SPLIT DATA -----------
        # =================================

        seed = 651 + i                # new, reproducible split in every iteration
        # train / validation / test split (0.7/0.15/0.15 of the cores),
        # stratified on area and then on core: a core is in only one set
        train_idx, val_idx, test_idx = area_core_split(core, seed)

        # core GS19GC10 (core == 17) represents the dataset the most (207 samples);
        # remove 1/4 of its samples from the training set to reduce its weight
        train_idx = downsample_core(train_idx, core, seed)

        # load significant features tested using Storey's q-value, computed on the training cores only
        feature_selection = select_features(data_fs, np.unique(core[train_idx]), var_columns,
                                            otu_columns, seed, clr=True)
        X = X_all[feature_selection].values.astype(float)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test, core_test = X[test_idx], y[test_idx], core[test_idx]


        # ===========================================
        # ---------- TRAIN AND TEST MODEL -----------
        # ===========================================

        # ------------------------------------
        # Test choosen model on test dataset
        # ------------------------------------

        # max_depth is chosen on the validation cores (ccp_alpha = 0 as before)
        dtree_sklearn = fit_tuned_tree(DecisionTreeRegressor(criterion='absolute_error', ccp_alpha=0, random_state=seed),
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
        new = np.isnan(first_test_pred[test_idx])
        first_test_pred[test_idx[new]] = y_pred_test[new]

        MSE = mean_squared_error(y_test,y_pred_test)
        MSE1 = mean_absolute_error(y_test,y_pred_test)

        print("y - y.pred: "+str(math.sqrt(MSE)))
        print("MSE: "+str((MSE)))
        print("mean absolute error: "+str((MSE1)))



        # Save important features
        rules = get_rules(dtree_sklearn, feature_selection, None) 

        if len(rule_features) == 0:
            rule_features = DT_Features(rules)
            rule_features_high, rule_features_low = DT_features_threshold(rules, threshold)
        else:
            rule_features = pd.concat([rule_features,DT_Features(rules)])
            h, l = DT_features_threshold(rules, threshold)
            rule_features_high, rule_features_low = pd.concat([rule_features_high,h]), pd.concat([rule_features_low,l])
    

        # ================================
        # ---- Extract core locations ----
        # ================================

        y_test = np.array(y_test)

        # --------------
        # ---- AMOR ----
        # --------------

        AMOR = AREA_CORES['AMOR']
        X_AMOR, y_AMOR = Data_Extraction(AMOR, core_test, X_test, y_test)

        # --------------
        # -- PACIFIC ---
        # --------------

        # store NWPO in another variable
        # and remove 26 - 39 from test Data
        NWPO = AREA_CORES['NWPO']
        X_NWPO, y_NWPO = Data_Extraction(NWPO, core_test, X_test, y_test)

        # --------------
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


    # =============================================
    # ----- SAVE RULES (once per chemical) -----
    # =============================================
    # OTU frequencies in the rules of all trees of this run
    name = ['general','high','low']
    for n,i in enumerate([rule_features,rule_features_high,rule_features_low]):
        rule_features2 = i.groupby(["OTU"]).sum()
        rule_features2 = rule_features2.sort_values(by='value', ascending=False)
        rule_features2 = rule_features2.reset_index()
        np.savetxt(path+f'feature_frequency_rules_{var}_{name[n]}.csv', rule_features2, delimiter=",", fmt="%s",header = 'OTU,value')

    # =====================================================
    # ----- Out-of-sample predictions for the plots -----
    # =====================================================
    # every sample once, predicted by a tree that did not see its core
    tested = ~np.isnan(first_test_pred)

    def area_predictions(area):
        sel = np.where(tested & np.isin(core, AREA_CORES[area]))[0]
        y_area, y_area_pred = y[sel], first_test_pred[sel]
        return y_area, y_area_pred, np.where(y_area <= threshold)[0], np.where(y_area > threshold)[0]

    y_AMOR, y_pred_AMOR, loc_low_AMOR, loc_high_AMOR = area_predictions('AMOR')
    y_NWPO, y_pred_NWPO, loc_low_NWPO, loc_high_NWPO = area_predictions('NWPO')
    y_WNAG, y_pred_WNAG, loc_low_WNAG, loc_high_WNAG = area_predictions('WNAG')
    y_MAR, y_pred_MAR, loc_low_MAR, loc_high_MAR = area_predictions('MAR')
    y_SCS, y_pred_SCS, loc_low_SCS, loc_high_SCS = area_predictions('SCS')
    y_SPO, y_pred_SPO, loc_low_SPO, loc_high_SPO = area_predictions('SPO')

    # ============================
    # ----- PLOTTING RESULTS -----
    # ============================
    
    if var == 'oxygen':
        ax_n = 0

    if var == 'nitrate':
        ax_n = 1

    if var == 'ammonium':
        ax_n = 2

    if var == 'manganese':
        ax_n = 3

    nbin = 30

    if var == 'oxygen':
        bins = np.linspace(min(np.concatenate([y_pred_AMOR-y_AMOR,y_pred_NWPO-y_NWPO,y_pred_MAR-y_MAR,y_pred_WNAG-y_WNAG])), max(np.concatenate([y_pred_AMOR-y_AMOR,y_pred_NWPO-y_NWPO,y_pred_MAR-y_MAR,y_pred_WNAG-y_WNAG])), nbin)/fact

    if var == 'nitrate':
        bins = np.linspace(min(np.concatenate([y_pred_AMOR-y_AMOR,y_pred_NWPO-y_NWPO,y_pred_MAR-y_MAR,y_pred_SCS-y_SCS,y_pred_SPO-y_SPO])), max(np.concatenate([y_pred_AMOR-y_AMOR,y_pred_NWPO-y_NWPO,y_pred_MAR-y_MAR,y_pred_SCS-y_SCS,y_pred_SPO-y_SPO])), nbin)/fact

    if var == 'ammonium':
        bins = np.linspace(min(np.concatenate([y_pred_AMOR-y_AMOR,y_pred_NWPO-y_NWPO,y_pred_SCS-y_SCS,y_pred_SPO-y_SPO])), max(np.concatenate([y_pred_AMOR-y_AMOR,y_pred_NWPO-y_NWPO,y_pred_SCS-y_SCS,y_pred_SPO-y_SPO])), nbin)/fact

    if var == 'manganese':
        bins = np.linspace(min(np.concatenate([y_pred_AMOR-y_AMOR,y_pred_SCS-y_SCS])), max(np.concatenate([y_pred_AMOR-y_AMOR])), nbin)/fact



    print("============ AMOR ============")
    loc_low = np.where(y_pred_AMOR <= threshold)[0]
    loc_low = np.unique(np.append([loc_low],[loc_low_AMOR]))
    ax[0,ax_n].scatter(np.delete(y_AMOR,loc_low)/fact,np.delete(y_pred_AMOR,loc_low)/fact, c =  'red', label='AMOR', s=markersize, alpha = 0.5,marker='+')
    loc_high = np.where(y > threshold)[0]
    ax[0,ax_n].plot(y[loc_high]/fact,y[loc_high]/fact, c = 'grey', linestyle='solid', alpha = 0.5)
    ax[0,ax_n].set_xlabel(f'Observed', fontweight = 'bold', fontsize = fs)
    if ax_n == 0:
        ax[0,ax_n].set_ylabel(f'Predicted', fontweight = 'bold', fontsize = fs)
    ax[0,ax_n].tick_params(labelsize=labelsize_ticks)
    ax[0,ax_n].axis('scaled')
    # ax[1,ax_n].hist(y_pred_AMOR[loc_low_AMOR]/fact-y_AMOR[loc_low_AMOR]/fact, alpha=0.5, color='red', bins=bins, stacked=True, histtype='bar')
    ax[1,ax_n].set_xlabel(f'Error', fontweight='bold', fontsize = fs)
    if ax_n == 0:
        ax[1,ax_n].set_ylabel('Frequency', fontweight='bold', fontsize = fs)
    ax[1,ax_n].tick_params(labelsize=labelsize_ticks)

    hist1 = [y_pred_AMOR[loc_low_AMOR]/fact-y_AMOR[loc_low_AMOR]/fact]
    hist2 = [y_pred_AMOR[loc_high_AMOR]/fact-y_AMOR[loc_high_AMOR]/fact]
    hist3 = [y_pred_AMOR/fact-y_AMOR/fact]
    label1 = ["AMOR"]
    label2 = ["AMOR"]
    label3 = ["AMOR"]
    col1 = ['red']
    col2 = ['red']
    col3 = ['red']

    if var == 'oxygen' or var == 'nitrate':
        print("============ North Pond ============")
        loc_low = np.where(y_pred_MAR <= threshold)[0]
        loc_low = np.unique(np.append([loc_low],[loc_low_MAR]))
        ax[0,ax_n].scatter(np.delete(y_MAR,loc_low)/fact,np.delete(y_pred_MAR,loc_low)/fact, c = 'green', label='MAR', s=markersize, alpha = 0.5, marker='*')

        hist1.append(y_pred_MAR[loc_low_MAR]/fact-y_MAR[loc_low_MAR]/fact)
        hist2.append(y_pred_MAR[loc_high_MAR]/fact-y_MAR[loc_high_MAR]/fact)
        hist3.append(y_pred_MAR/fact-y_MAR/fact)
        label1.append("MAR")
        label2.append("MAR")
        label3.append("MAR")
        col1.append('green')
        col2.append('green')
        col3.append('green')

        if  len(y_NWPO) > 0:
            print("============ PACIFIC ============")
            loc_low = np.where(y_pred_NWPO <= threshold)[0]
            loc_low = np.unique(np.append([loc_low],[loc_low_NWPO]))
            ax[0,ax_n].scatter(np.delete(y_NWPO,loc_low)/fact,np.delete(y_pred_NWPO,loc_low)/fact, c = 'blue', label='NWPO', s=markersize, alpha = 0.5, marker='d')
    
            hist3.append(y_pred_NWPO/fact-y_NWPO/fact)
            label3.append("NWPO")
            col3.append('blue')

        if zero_values or var == 'nitrate':
            print("============ SCS ============")
            loc_low = np.where(y_pred_SCS <= threshold)[0]
            loc_low = np.unique(np.append([loc_low],[loc_low_SCS]))
            ax[0,ax_n].scatter(np.delete(y_SCS,loc_low)/fact,np.delete(y_pred_SCS,loc_low)/fact, c = '#8B2252', label='SCS', s=markersize, alpha = 0.5, marker='s')

            hist3.append(y_pred_SCS/fact-y_SCS/fact)
            label3.append("SCS")
            col3.append('#8B2252')
            
        
        if zero_values or var != 'oxygen':
            print("============ SPO ============")
            loc_low = np.where(y_pred_SPO <= threshold)[0]
            loc_low = np.unique(np.append([loc_low],[loc_low_SPO]))
            ax[0,ax_n].scatter(np.delete(y_SPO,loc_low)/fact,np.delete(y_pred_SPO,loc_low)/fact, c = 'aqua', label='SPO', s=markersize, alpha = 0.5)
            hist3.append(y_pred_SPO/fact-y_SPO/fact)
            label3.append("SPO")
            col3.append('aqua')


        # mean accuracy percentage
        perc_AMOR_absent = np.nanmean(perc_AMOR_absent)
        perc_AMOR_present = np.nanmean(perc_AMOR_present)
        perc_MAR_absent = np.nanmean(perc_MAR_absent)
        perc_MAR_present = np.nanmean(perc_MAR_present)
        perc_NWPO_absent = np.nanmean(perc_NWPO_absent)
        perc_NWPO_present = np.nanmean(perc_NWPO_present)
        perc_SCS_present = np.nanmean(perc_SCS_present)
        perc_SCS_absent = np.nanmean(perc_SCS_absent)
        perc_SPO_present = np.nanmean(perc_SPO_present)
        perc_SPO_absent = np.nanmean(perc_SPO_absent)


        # mean number of samples used 
        number_AMOR_absent = np.round(np.nanmean(number_AMOR_absent))
        number_AMOR_present = np.round(np.nanmean(number_AMOR_present))

        number_MAR_absent = np.round(np.nanmean(number_MAR_absent))
        number_MAR_present = np.round(np.nanmean(number_MAR_present))

        number_NWPO_absent = np.round(np.nanmean(number_NWPO_absent))
        number_NWPO_present = np.round(np.nanmean(number_NWPO_present))

        number_SCS_absent = np.round(np.nanmean(number_SCS_absent))
        number_SCS_present = np.round(np.nanmean(number_SCS_present))

        number_SPO_absent = np.round(np.nanmean(number_SPO_absent))
        number_SPO_present = np.round(np.nanmean(number_SPO_present))
        
        accuracy = [[np.round(perc_AMOR_absent,1), np.round(perc_AMOR_present,1), f'{number_AMOR_absent}/{number_AMOR_present}'],
                    [float('nan'),float('nan'),float('nan')],
                    [np.round(perc_MAR_absent,1),np.round(perc_MAR_present,1), f'{number_MAR_absent}/{number_MAR_present}'],
                    [np.round(perc_NWPO_absent,1),np.round(perc_NWPO_present,1), f'{number_NWPO_absent}/{number_NWPO_present}'],
                    [np.round(perc_SCS_absent,1),np.round(perc_SCS_present,1), f'{number_SCS_absent}/{number_SCS_present}'],
                    [np.round(perc_SPO_absent,1),np.round(perc_SPO_present,1), f'{number_SPO_absent}/{number_SPO_present}']]

        labels =  ['AMOR_P','AMOR_A','MAR_P','MAR_A','WNAG_P','WNAG_A','NWPO_P','NWPO_A','SPO_P','SPO_A','SCS_P','SCS_A']#list(np.unique(y_pred_all))
        acc = [perc_AMOR_present,perc_AMOR_absent,perc_MAR_present,perc_MAR_absent,float('nan'),float('nan'),perc_NWPO_present,perc_NWPO_absent,perc_SPO_present,perc_SPO_absent,perc_SCS_present,perc_SCS_absent]
        
    if var == 'oxygen':

        print("============ Atlantic ============")
        loc_low = np.where(y_pred_WNAG <= threshold)[0]
        loc_low = np.unique(np.append([loc_low],[loc_low_WNAG]))
        ax[0,ax_n].scatter(np.delete(y_WNAG,loc_low)/fact,np.delete(y_pred_WNAG,loc_low)/fact, c = 'orange', label='WNAG', s=markersize, alpha = 0.5, marker='h')
    
        hist1.append(y_pred_WNAG[loc_low_WNAG]/fact-y_WNAG[loc_low_WNAG]/fact)
        label1.append("WNAG")
        col1.append('orange')
        hist2.append(y_pred_WNAG[loc_high_WNAG]/fact-y_WNAG[loc_high_WNAG]/fact)
        label2.append("WNAG")
        col2.append('orange')
        hist3.append(y_pred_WNAG/fact-y_WNAG/fact)
        label3.append("WNAG")
        col3.append('orange')

       # mean accuracy percentage
        perc_WNAG_absent = np.nanmean(perc_WNAG_absent)
        perc_WNAG_present = np.nanmean(perc_WNAG_present)

        # mean number of samples used 
        if len(number_WNAG_absent) > 0:
            number_WNAG_absent = np.round(np.nanmean(number_WNAG_absent))
        else:
            number_WNAG_absent = float('nan')
        number_WNAG_present = np.round(np.nanmean(number_WNAG_present))

        accuracy = [[np.round(perc_AMOR_absent,1), np.round(perc_AMOR_present,1), f'{number_AMOR_absent}/{number_AMOR_present}'],
                    [np.round(perc_WNAG_absent,1),np.round(perc_WNAG_present,1), f'{number_WNAG_absent}/{number_WNAG_present}'],
                    [np.round(perc_MAR_absent,1),np.round(perc_MAR_present,1), f'{number_MAR_absent}/{number_MAR_present}'],
                    [np.round(perc_NWPO_absent,1),np.round(perc_NWPO_present,1), f'{number_NWPO_absent}/{number_NWPO_present}'],
                    [np.round(perc_SCS_absent,1),float('nan'), f'{number_SCS_absent}/{0}'],
                    [np.round(perc_SPO_absent,1),np.round(perc_SPO_present,1), f'{number_SPO_absent}/{number_SPO_present}']]

        labels =  ['AMOR_P','AMOR_A','MAR_P','MAR_A','WNAG_P','WNAG_A','NWPO_P','NWPO_A','SPO_P','SPO_A','SCS_P','SCS_A']#list(np.unique(y_pred_all))
        acc = [perc_AMOR_present,perc_AMOR_absent,perc_MAR_present,perc_MAR_absent,perc_WNAG_present,perc_WNAG_absent,perc_NWPO_present,perc_NWPO_absent,perc_SPO_present,perc_SPO_absent,perc_SCS_present,perc_SCS_absent]
        

    if var == 'manganese' or var == 'ammonium':

        print("============ SCS ============")
        loc_low = np.where(y_pred_SCS <= threshold)[0]
        loc_low = np.unique(np.append([loc_low],[loc_low_SCS]))
        ax[0,ax_n].scatter(np.delete(y_SCS,loc_low)/fact,np.delete(y_pred_SCS,loc_low)/fact, c = '#8B2252', label='SCS', s=markersize, alpha = 0.5, marker='s')

        hist3.append(y_pred_SCS/fact-y_SCS/fact)
        label3.append("SCS")
        col3.append('#8B2252')
        
        # mean accuracy percentage
        perc_SCS_absent = np.nanmean(perc_SCS_absent)
        perc_SCS_present = np.nanmean(perc_SCS_present)

        # mean number of samples used 
        if len(number_SCS_absent) > 0:
            number_SCS_absent = np.round(np.nanmean(number_SCS_absent))
        else:
            number_SCS_absent = float('nan')
        number_SCS_present = np.round(np.nanmean(number_SCS_present))

        print("============ SPO ============")
        loc_low = np.where(y_pred_SPO <= threshold)[0]
        loc_low = np.unique(np.append([loc_low],[loc_low_SPO]))
        ax[0,ax_n].scatter(np.delete(y_SPO,loc_low)/fact,np.delete(y_pred_SPO,loc_low)/fact, c = 'aqua', label='SPO', s=markersize, alpha = 0.5)
        
        hist3.append(y_pred_SPO/fact-y_SPO/fact)
        label3.append("SPO")
        col3.append('aqua')
 
        perc_SPO_present = np.nanmean(perc_SPO_present)
        perc_SPO_absent = np.nanmean(perc_SPO_absent)

        number_SPO_absent = np.round(np.nanmean(number_SPO_absent))
        number_SPO_present = np.round(np.nanmean(number_SPO_present))
        
        
        print("============ AMOR ============")
        # mean accuracy percentage
        perc_AMOR_absent = np.nanmean(perc_AMOR_absent)
        perc_AMOR_present = np.nanmean(perc_AMOR_present)

        # mean number of samples used 
        number_AMOR_absent = np.round(np.nanmean(number_AMOR_absent))
        number_AMOR_present = np.round(np.nanmean(number_AMOR_present))

        accuracy = [[np.round(perc_AMOR_absent,1), np.round(perc_AMOR_present,1), f'{number_AMOR_absent}/{number_AMOR_present}'],
            [float('nan'),float('nan'),float('nan')],
            [float('nan'),float('nan'),float('nan')],
            [float('nan'),float('nan'),float('nan')],
            [np.round(perc_SCS_absent,1),np.round(perc_SCS_present,1), f'{number_SCS_absent}/{number_SCS_present}'],
            [np.round(perc_SPO_absent,1),np.round(perc_SPO_present,1), f'{number_SPO_absent}/{number_SPO_present}']]


        labels =  ['AMOR_P','AMOR_A','MAR_P','MAR_A','WNAG_P','WNAG_A','NWPO_P','NWPO_A','SPO_P','SPO_A','SCS_P','SCS_A']#list(np.unique(y_pred_all))
        acc = [perc_AMOR_present,perc_AMOR_absent,float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),perc_SPO_present,number_SPO_absent,perc_SCS_present,perc_SCS_absent]
        

    if var == 'ammonium':
        
        print("============ PACIFIC ============")
        loc_low = np.where(y_pred_NWPO <= threshold)[0]
        loc_low = np.unique(np.append([loc_low],[loc_low_NWPO]))
        ax[0,ax_n].scatter(np.delete(y_NWPO,loc_low)/fact,np.delete(y_pred_NWPO,loc_low)/fact, c = 'blue', label='NWPO', s=markersize, alpha = 0.5, marker='d')

        hist3.append(y_pred_NWPO/fact-y_NWPO/fact)
        label3.append("NWPO")
        col3.append('blue')


        print("============ SPO ============")
        loc_low = np.where(y_pred_SPO <= threshold)[0]
        loc_low = np.unique(np.append([loc_low],[loc_low_SPO]))
        ax[0,ax_n].scatter(np.delete(y_SPO,loc_low)/fact,np.delete(y_pred_SPO,loc_low)/fact, c = 'aqua', label='SPO', s=markersize, alpha = 0.5)
        
        hist3.append(y_pred_SPO/fact-y_SPO/fact)
        label3.append("SPO")
        col3.append('aqua')

        perc_NWPO_absent = np.nanmean(perc_NWPO_absent)
        perc_NWPO_present = np.nanmean(perc_NWPO_present)
        perc_SPO_present = np.nanmean(perc_SPO_present)
        perc_SPO_absent = np.nanmean(perc_SPO_absent)
        
        if len(number_NWPO_absent) >0:
            number_NWPO_absent = np.round(np.nanmean(number_NWPO_absent))
        else:
            number_NWPO_absent = float('nan')
        number_NWPO_present = np.round(np.nanmean(number_NWPO_present))

        
        number_SPO_absent = np.round(np.nanmean(number_SPO_absent))
        number_SPO_present = np.round(np.nanmean(number_SPO_present))
        

        accuracy = [[np.round(perc_AMOR_absent,1), np.round(perc_AMOR_present,1), f'{number_AMOR_absent}/{number_AMOR_present}'],
                    [float('nan'),float('nan'),float('nan')],
                    [float('nan'),float('nan'),float('nan')],
                    [np.round(perc_NWPO_absent,1),np.round(perc_NWPO_present,1), f'{number_NWPO_absent}/{number_NWPO_present}'],
                    [np.round(perc_SCS_absent,1),np.round(perc_SCS_present,1), f'{number_SCS_absent}/{number_SCS_present}'],
                    [np.round(perc_SPO_absent,1),np.round(perc_SPO_present,1), f'{number_SPO_absent}/{number_SPO_present}']]

        labels =  ['AMOR_P','AMOR_A','MAR_P','MAR_A','WNAG_P','WNAG_A','NWPO_P','NWPO_A','SPO_P','SPO_A','SCS_P','SCS_A']#list(np.unique(y_pred_all))
        acc = [perc_AMOR_present,perc_AMOR_absent,float('nan'),float('nan'),float('nan'),float('nan'),perc_NWPO_present,perc_NWPO_absent,perc_SPO_present,perc_SPO_absent,perc_SCS_present,perc_SCS_absent]
        

    ax[1,ax_n].hist(hist3, alpha=0.5, color=col3, bins = bins, stacked=True, histtype='bar', log=True)

    # Define titles per column   
    ax[0,0].set_title(r'$O_2$', fontweight='bold', fontsize = 20)
    ax[0,1].set_title(r'$NO_3$', fontweight='bold', fontsize = 20)
    ax[0,2].set_title(r'$NH_4$', fontweight='bold', fontsize = 20)
    ax[0,3].set_title(r'$Mn$', fontweight='bold', fontsize = 20)

    # R2 score
    # -----------
    y_pred_test = dtree_sklearn.predict(X_test)
    loc_high = np.where(y_test > threshold)[0]
                    
    r2 = r2_score(y_pred_test[loc_high],y_test[loc_high])
    p_value_test = pearsonr(y_pred_test[loc_high],y_test[loc_high])[1]

    text_box = AnchoredText(f'p = {np.round(p_value_test,3)}', frameon=False, loc=4, pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    ax[0,ax_n].add_artist(text_box)

    ax[0,0].text(-0.15, 1.05, string.ascii_uppercase[0], transform=ax[0,0].transAxes, 
            size=30, weight='bold')
    ax[1,0].text(-0.15, 1.05, string.ascii_uppercase[1], transform=ax[1,0].transAxes, 
            size=30, weight='bold')

    if var == 'oxygen':
        col = [labels]
        y_acc = [np.array(acc)]
        x_acc = [var]
    else:
        col.append(labels)
        y_acc.append(np.array(acc))
        x_acc.append(var)


ax[0,0].legend(loc='upper center', bbox_to_anchor=(2.5, 1.5),
          ncol=6, fancybox=True, shadow=False,fontsize="18")


plt.subplots_adjust(left=0.06, right=0.872, top=0.85, bottom=0.095, wspace=0.3, hspace=0.55)


plt.savefig(os.getcwd()+f'/DT_regressor_results_{iterations}it.png')

plt.show()


