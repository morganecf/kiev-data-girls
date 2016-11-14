import numpy as np
import scipy as sp
import pandas as pd
import random
import math

def deciles(x, n=10):
    ''' Compute decile (breakpoints) of a variable '''
    if x.min() == x.max():
        return [x.min()]
    qseq = [i / n for i in range(n + 1)]
    dec = [x.quantile(q) for q in qseq]
    seen = set()
    seen_add = seen.add
    out = [i for i in dec if i not in seen and not seen_add(i)]
    if len(out) > 2:
        return out
    else:
        return deciles(x, n=2*n)

def cut(x, decs):
    '''
    Compute cuts of a numeric variable and returns a series (to use in groupby)
    '''
    xmin = x.min()
    x = x.fillna(xmin - 1)
    sep = sorted(decs)[:-1]
    cuts = np.digitize(x, sep)
    return pd.Series(data=cuts)

def lift(act, pred, n=10):
    '''
    Calculate the values to plot a lift chart using n buckets.
    Parameters
    ----------
    pred: 1D row or column array
        The predicted values
    act: 1D row array
        The actual target values
    n: integer
        The number of bins to use in the lift calculation

    Returns
    -------
    lift_summary: dict
        {pred: array, act: array, rows: array}
        each array has n elements
        pred = sum of predicted values in bin
        act = sum of actual values in bin
        rows = row count per bin
    '''
    random.seed(1234)

    # Prediction and actual
    pred1 = pd.Series(data=np.squeeze(pred), dtype=np.float64)
    act = pd.Series(data=act)
    pred = pred1.apply(lambda x: float(x) + max(abs(float(x) * 1e-4), 1e-6) * random.uniform(-1, 1))

    # Compute prediction deciles
    # Points in pred data that make equal-size cuts
    dec = deciles(pred, n)

    # Prediction --> bin mapping
    pred_bins = cut(pred, dec)

    pred_for_group = pred1
    act_for_group = act

    # Bin counts -- should be all about equal
    counts = pred_bins.value_counts().to_dict()

    # Group prediction values by bin
    grouped_pred = pred_for_group.groupby(pred_bins)

    # Actual --> bin mapping
    act_bins = pd.DataFrame({'act': act_for_group.values, 'bins': pred_bins})

    # Group actual values by prediction bin
    grouped_act = act_bins.groupby('bins')

    # Get mean, median, quantiles
    out = pd.DataFrame()

    out['act_mean'] = list(grouped_act.mean()['act'])
    out['pred_mean'] = list(grouped_pred.mean())

    # out['act_median'] = list(grouped_act.median()['act'])
    # out['act_top10'] = list(grouped_act.quantile(0.9)['act'])
    # out['act_top25'] = list(grouped_act.quantile(0.75)['act'])
    # out['act_bottom10'] = list(grouped_act.quantile(0.1)['act'])
    # out['act_bottom25'] = list(grouped_act.quantile(0.25)['act'])
    # out['pred_median'] = list(grouped_pred.median())

    return out

