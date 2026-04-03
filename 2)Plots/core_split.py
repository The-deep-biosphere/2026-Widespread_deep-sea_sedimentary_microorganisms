"""
Leakage-free data splitting, feature selection and model selection shared by
the decision-tree scripts in this folder.

Samples from one sediment core are neighbouring depths of the same profile and
therefore strongly correlated. A random sample-level split puts samples of the
same core in both train and test, which inflates the test scores. Everything
here works on whole cores:

* area_core_split : train / validation / test split, stratified on the study
                    area first and on cores second. A core is in exactly one
                    of the three sets.
* downsample_core : reduce the weight of the dominant core GS19GC10 inside the
                    training set only.
* select_features : the significant-OTU selection of
                    1)Preprocessing/4)Significant_gene/4.5)Significant_genes.py
                    (Welch t-statistic, permutation p-values, Storey q-values)
                    computed on the training cores only.
* fit_tuned_tree  : choose max_depth on the validation cores and return the
                    fitted tree. The test cores are never used for any choice.
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, BSpline

# ------------------------------------------------------------------
# Core -> study area mapping (authoritative table in this folder)
# ------------------------------------------------------------------
_COORDS = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Core_Coordinates.csv'))
AREA_OF_CORE = dict(zip(_COORDS['ID'].astype(int), _COORDS['Location']))
AREAS = ['AMOR', 'MAR', 'NWPO', 'SCS', 'SPO', 'WNAG']
AREA_CORES = {a: np.array(sorted(c for c, l in AREA_OF_CORE.items() if l == a)) for a in AREAS}

GS19GC10 = 17   # core that dominates the data set (207 samples)


def area_core_split(core, seed, val_frac=0.15, test_frac=0.15):
    """Split sample indices into train / validation / test by whole cores,
    stratified by study area.

    Within every area the cores are shuffled and allocated separately, so each
    area contributes cores to every set it has enough cores for:
      * >= 3 cores : >= 1 test core, >= 1 validation core, the rest train
      * 2 cores    : 1 train core, 1 test core (no validation core possible)
      * 1 core     : train only (the area cannot be tested without leakage)

    Returns three sorted integer index arrays into `core`.
    """
    core = np.asarray(core).astype(int)
    unknown = set(np.unique(core)) - set(AREA_OF_CORE)
    if unknown:
        raise ValueError(f'Cores without area in Core_Coordinates.csv: {sorted(unknown)}')

    rng = np.random.RandomState(seed)
    train_cores, val_cores, test_cores = [], [], []
    for area in AREAS:
        cores = np.intersect1d(np.unique(core), AREA_CORES[area])
        n = len(cores)
        if n == 0:
            continue
        cores = rng.permutation(cores)
        if n == 1:
            print(f'WARNING: {area} has a single core; it is only used for training')
            n_test, n_val = 0, 0
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test = max(1, int(round(test_frac * n)))
            n_val = max(1, int(round(val_frac * n)))
        test_cores += list(cores[:n_test])
        val_cores += list(cores[n_test:n_test + n_val])
        train_cores += list(cores[n_test + n_val:])

    train_idx = np.where(np.isin(core, train_cores))[0]
    val_idx = np.where(np.isin(core, val_cores))[0]
    test_idx = np.where(np.isin(core, test_cores))[0]

    # no core may appear in more than one set
    assert not (set(core[train_idx]) & set(core[val_idx]))
    assert not (set(core[train_idx]) & set(core[test_idx]))
    assert not (set(core[val_idx]) & set(core[test_idx]))
    return train_idx, val_idx, test_idx


def downsample_core(idx, core, seed, core_id=GS19GC10, drop_frac=0.25):
    """Remove `drop_frac` of the samples of `core_id` from the index set `idx`
    (sampling without replacement). Apply to the training indices only, so the
    dominant core GS19GC10 weighs less during training. Samples are discarded,
    never moved to another set."""
    core = np.asarray(core).astype(int)
    idx = np.asarray(idx)
    in_core = idx[core[idx] == core_id]
    n_drop = int(len(in_core) * drop_frac)
    if n_drop == 0:
        return idx
    drop = np.random.RandomState(seed).choice(in_core, n_drop, replace=False)
    return np.setdiff1d(idx, drop)


# ------------------------------------------------------------------
# Feature selection (4.5)Significant_genes.py, on training cores only)
# ------------------------------------------------------------------
def clr_rows(X):
    """Per-sample centred log-ratio of relative abundances. A pseudocount of
    half the smallest non-zero abundance replaces zeros. Each row is transformed
    on its own, so no information passes between samples."""
    X = np.asarray(X, dtype=float)
    pc = X[X > 0].min() / 2
    L = np.log(X + pc)
    return L - L.mean(axis=1, keepdims=True)


def _welch_t(X, neg):
    """Welch t-statistics (mean_neg - mean_pos) for every column of X.
    `neg` is a boolean vector (1-D) or a (B, n) boolean matrix of permutations."""
    neg = np.atleast_2d(neg).astype(float)
    pos = 1.0 - neg
    n_neg = neg.sum(axis=1, keepdims=True)
    n_pos = pos.sum(axis=1, keepdims=True)
    m_neg = neg @ X / n_neg
    m_pos = pos @ X / n_pos
    v_neg = np.clip(neg @ (X ** 2) / n_neg - m_neg ** 2, 0, None) / n_neg
    v_pos = np.clip(pos @ (X ** 2) / n_pos - m_pos ** 2, 0, None) / n_pos
    with np.errstate(divide='ignore', invalid='ignore'):
        return (m_neg - m_pos) / np.sqrt(v_neg + v_pos)


def _storey_qvalues(p, lam0):
    """Storey & Tibshirani (2003) q-values. pi0 is the cubic smoothing spline of
    pi0(lambda) evaluated at lam0, as in 4.5)Significant_genes.py."""
    m = len(p)
    lamb = np.arange(0., .96, 0.01)
    pi = np.array([(p > l).sum() / (m * (1 - l)) for l in lamb])
    pi0 = float(BSpline(*splrep(lamb, pi, s=3))(lam0))
    pi0 = min(1.0, max(pi0, 1.0 / m))
    order = np.argsort(p)
    q_sorted = pi0 * m * p[order] / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]     # q_(i) = min_{j>=i}
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def significant_otus(X, conc, otu_names, seed, n_permutations=100, lam0=0.8):
    """OTUs that differ between samples where the chemical is absent
    (conc <= 0) and present (conc > 0), following 4.5)Significant_genes.py:
    Welch t-statistic, empirical p-values from a pooled permutation null and
    Storey q-values. OTUs are kept when their expected number of false
    positives is zero (q-value == 0), the criterion used in 4.5."""
    X = np.asarray(X, dtype=float)
    otu_names = np.asarray(otu_names)
    neg = np.asarray(conc) * 1000 <= 0
    if neg.sum() < 2 or (~neg).sum() < 2:
        return np.array([], dtype=otu_names.dtype)

    # remove OTUs that are constant in these samples (t-statistic undefined)
    keep = X.std(axis=0) > 0
    X, otu_names = X[:, keep], otu_names[keep]
    m = X.shape[1]

    t = _welch_t(X, neg)[0]
    # permutations as in 4.5: shuffle the rows and call the first n_neg "absent"
    rng = np.random.RandomState(seed)
    n, n_neg = len(neg), int(neg.sum())
    perms = np.zeros((n_permutations, n), dtype=bool)
    for b in range(n_permutations):
        perms[b, rng.permutation(n)[:n_neg]] = True
    null = np.sort(np.abs(_welch_t(X, perms)).ravel())
    null = null[np.isfinite(null)]
    # fraction of all (OTU, permutation) null statistics at least as extreme
    p = (len(null) - np.searchsorted(null, np.abs(t), side='left')) / len(null)
    p[~np.isfinite(t)] = 1.0

    q = _storey_qvalues(p, lam0)
    return otu_names[q * m == 0]


_FS_CACHE = {}


def select_features(data, train_cores, var_columns, otu_columns, seed, clr=False, n_permutations=100):
    """Union of the significant OTUs for all chemical variables in
    `var_columns`, computed only on samples from `train_cores`.

    `data` is the full data frame (all variables, not filtered on one chemical),
    so the significance test for each chemical uses every training-core sample
    where it was measured. Set `clr=True` when the OTU columns hold relative
    abundances, so the test runs on CLR values as in 4.5. Results are cached per
    set of training cores."""
    key = (tuple(sorted(int(c) for c in train_cores)), tuple(var_columns), clr)
    if key in _FS_CACHE:
        return _FS_CACHE[key]

    sub = data[np.isin(data['Core'].astype(int), list(train_cores))]
    X_all = sub[otu_columns].values.astype(float)
    if clr:
        X_all = clr_rows(X_all)
    selected = []
    for v in var_columns:
        has = sub[v].notna().values
        conc = sub[v].values.astype(float)[has]
        lam0 = 0.7 if v.startswith('Mn') else 0.8
        selected.append(significant_otus(X_all[has], conc, otu_columns, seed, n_permutations, lam0))
    result = np.unique(np.concatenate(selected))
    _FS_CACHE[key] = result
    return result


# ------------------------------------------------------------------
# Model selection on the validation cores
# ------------------------------------------------------------------
def _predict_at_depth(tree, X, depth):
    """Predictions of `tree` truncated at `depth` (the node reached at that
    depth is treated as a leaf)."""
    path = tree.decision_path(X).tocsr()
    path.sort_indices()                      # node ids increase with depth
    lengths = np.diff(path.indptr)
    node = path.indices[path.indptr[:-1] + np.minimum(depth, lengths - 1)]
    value = tree.tree_.value[node, 0, :]
    if hasattr(tree, 'classes_'):
        return tree.classes_[np.argmax(value, axis=1)]
    return value[:, 0]


def _balanced_accuracy(y, y_pred):
    y, y_pred = np.asarray(y), np.asarray(y_pred)
    return np.mean([np.mean(y_pred[y == c] == c) for c in np.unique(y)])


def fit_tuned_tree(estimator, X_train, y_train, X_val, y_val):
    """Fit `estimator` (a DecisionTreeClassifier/Regressor) with the max_depth
    that scores best on the validation set: balanced accuracy for classifiers,
    mean absolute error for regressors. The candidate depths are 1 up to the
    depth of the fully grown tree, scored by truncating one fully grown tree;
    the final tree is refitted with the chosen depth. Ties go to the shallower
    tree. Returns the fitted estimator."""
    full = estimator.set_params(max_depth=None).fit(X_train, y_train)
    if len(y_val) == 0:
        print('WARNING: empty validation set, using fully grown tree')
        return full
    y_val = np.asarray(y_val)
    classifier = hasattr(full, 'classes_')
    best_depth, best_score = None, -np.inf
    for d in range(1, full.get_depth() + 1):
        pred = _predict_at_depth(full, X_val, d)
        score = _balanced_accuracy(y_val, pred) if classifier else -np.mean(np.abs(y_val - pred))
        if score > best_score + 1e-12:
            best_depth, best_score = d, score
    if best_depth is None or best_depth == full.get_depth():
        return full
    return estimator.set_params(max_depth=best_depth).fit(X_train, y_train)
