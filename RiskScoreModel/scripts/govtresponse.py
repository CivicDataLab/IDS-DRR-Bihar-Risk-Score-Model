import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
path = os.getcwd() #+ r"/HP/flood-data-ecosystem-Himachal-Pradesh"
master_variables = pd.read_csv(path+'/RiskScoreModel/data/MASTER_VARIABLES.csv')

def get_financial_year(timeperiod):
    if int(timeperiod.split('_')[1]) >= 4:
        return str(int(timeperiod.split('_')[0]))+'-'+str(int(timeperiod.split('_')[0])+1)
    else:
        return str(int(timeperiod.split('_')[0]) - 1)+'-'+str(int(timeperiod.split('_')[0]))
    
# Apply the function to create the 'FinancialYear' column
master_variables['FinancialYear'] = master_variables['timeperiod'].apply(lambda x: get_financial_year(x))

#INPUT VARS
government_response_vars = ["total_tender_awarded_value",
                       # 'Repair and Restoration_tenders_awarded_value',
                       # 'LWSS_tenders_awarded_value', 'NDRF_tenders_awarded_value', 
                       # 'SDMF_tenders_awarded_value', 'WSS_tenders_awarded_value', 
                       # 'Preparedness Measures_tenders_awarded_value', 
                       # 'Immediate Measures_tenders_awarded_value', 
                       # "Others_tenders_awarded_value",
                       #'relief_and_mitigation_sanction_value'
                      ]

# Find cumsum in each FY of the government response vars
# (on a copy - master_variables must keep its raw monthly values. This file is
# saved below as factor_scores_l1_government-response.csv and is picked up by
# topis_riskscore_district.py as the FIRST of the factor files, which makes it
# the base of the merge. Writing the cumsum back over the raw column therefore
# published the running FY total as the monthly 'total-tender-awarded-value':
# 1,097,157,680,324 instead of the true 190,064,496,085, and
# 'total-tender-awarded-value-fy-cumsum' then took the cumsum of a cumsum,
# reaching 5,162,028,128,948. Both were in the plugin's bihar_data.csv.
govtresponse_df = master_variables[government_response_vars + ['timeperiod', 'object_id']].copy()
for var in government_response_vars:
    govtresponse_df[var] = master_variables.groupby(['object_id','FinancialYear'])[var].cumsum()


# How the cumulative spend becomes the 1-5 class. 5 is the WORST response and
# therefore the highest contribution to risk; 1 is the best.
#
#   'zband' - symmetric z-score bands on log spend, pooled over all months
#   'sigma' - the original one-sided ladder (kept so the old numbers can be
#             reproduced), which does not work on this data:
#
# The old rule put every block at or below the monthly MEAN into class 5, and
# reserved classes 2 and 1 for blocks more than 2 and 3 standard deviations
# ABOVE it. Bihar's tender spend is extremely right-skewed - the per-month skew
# has a median of 15.08 and runs as high as 22.69 - so "at or below the mean"
# is almost everybody: a median of 95.1% of blocks per month, and 100% in the
# worst month. The result was 90.5% of all block-months in class 5, with class
# 1 at 1.0% and class 2 at 0.5%, and 282 of 533 blocks sitting at class 5 in
# every one of the 66 months. The whole range from "received nothing at all" to
# "received just under the average" collapsed into one class, which is where
# the discrimination went.
#
# Taking log1p first makes the spread roughly symmetric, so the band edges land
# inside the data instead of out in the tail, and centring the bands on the mean
# instead of starting at it uses both halves of the distribution. Pooling the
# fit over all months (rather than refitting per month) keeps the class absolute:
# a month in which no block was funded shows up as every block scoring 5,
# instead of being rescaled into a spread that implies some blocks did well.
#
# This mirrors the same fix in the Himachal Pradesh model, which feeds two
# variables here; Bihar feeds only total_tender_awarded_value, so the pooled
# z-score below is of that one variable. Adding
# relief_and_mitigation_sanction_value (available in MASTER_VARIABLES) would
# change what the factor measures and is a separate decision.
RESPONSE_SCALE = 'zband'   # 'zband' | 'sigma'

# Band edges in standard deviations from the mean, worst-to-best. ZBAND_EDGES
# is used when spend is spread across the units; ZBAND_EDGES_FUNDED is used for
# the funded units alone when there is a lump of unfunded ones (see below).
ZBAND_EDGES = (-1.5, -0.5, 0.5, 1.5)
ZBAND_EDGES_FUNDED = (-1.0, 0.0, 1.0)

# Bihar has far more unfunded units than Himachal Pradesh: 77.8% of block-months
# have no flood tender spend at all in the financial year to date, against 10.6%
# of HP tehsil-months. That lump cannot be banded with the rest, because it
# dominates the pooled mean and standard deviation: a zero row z-scores to
# -0.53, which lands it in class 4 rather than the worst class, and leaves
# classes 3 and 5 empty. (With relief_and_mitigation_sanction_value added the
# overlap is near-total - the all-zero share stays 77.8% - so a second variable
# does not help; it just moves the lump to class 3.)
#
# So the unfunded units are classed 5 directly - "received nothing this
# financial year to date" - and the funded ones are banded among themselves
# into 1-4. This is what the HP numbers already mean: there, class 5 came out
# as exactly and only the zero-spend rows. Stating it explicitly makes the two
# states agree by construction rather than by coincidence.
#
# Note the 77.8% is a source-coverage question, not a scoring one. Bihar's
# sdrf_tenders_awarded_value and sdrf_sanctions_awarded_value are zero for
# every row in the data, so SDRF spend is not reaching this model at all.


def _sigma_ladder(score):
    """Original one-sided ladder: <= mean is 5, then +1 sd steps down to 1."""
    mean = score.mean()
    std = score.std()
    conditions = [
        (score <= mean),
        (score > mean) & (score <= mean + std),
        (score > mean + std) & (score <= mean + 2 * std),
        (score > mean + 2 * std) & (score <= mean + 3 * std),
        (score > mean + 3 * std)
    ]
    return np.select(conditions, [5, 4, 3, 2, 1])


def _z_bands(score, unfunded=None):
    """Score the units 1-5 from their z-scored log spend, worst (5) to best (1).

    unfunded, when given, flags units that received nothing. Those are classed
    5 outright and the rest are banded among themselves into 1-4, so the lump
    at zero cannot drag the band edges off the funded distribution.
    """
    if unfunded is None or not bool(unfunded.any()):
        mean = score.mean()
        std = score.std()
        if not np.isfinite(std) or std == 0:
            return np.full(len(score), 3)
        lo, mlo, mhi, hi = [mean + e * std for e in ZBAND_EDGES]
        conditions = [score <= lo, score <= mlo, score <= mhi, score <= hi, score > hi]
        return np.select(conditions, [5, 4, 3, 2, 1])

    out = np.full(len(score), 5)
    funded = ~np.asarray(unfunded)
    if not funded.any():
        return out
    s = np.asarray(score)[funded]
    mean = s.mean()
    std = s.std(ddof=1) if funded.sum() > 1 else 0.0
    if not np.isfinite(std) or std == 0:
        out[funded] = 3
        return out
    lo, mid, hi = [mean + e * std for e in ZBAND_EDGES_FUNDED]
    out[funded] = np.select([s <= lo, s <= mid, s <= hi, s > hi], [4, 3, 2, 1])
    return out


if RESPONSE_SCALE == 'zband':
    # log1p pulls in the right tail; zeros stay pinned at the bottom, so class 5
    # ends up meaning "this block received nothing this financial year to date".
    logged = np.log1p(govtresponse_df[government_response_vars])
    # z-score each variable over the pooled set so they are commensurate, then
    # average rather than sum - a sum would let the wider-spread variable
    # dominate purely because of its spread.
    z = (logged - logged.mean()) / logged.std()
    govtresponse_df['sum'] = z.mean(axis=1)
    unfunded = (govtresponse_df[government_response_vars].fillna(0) == 0).all(axis=1)
    govtresponse_df['government-response'] = _z_bands(govtresponse_df['sum'], unfunded)
    govtresponse_df_months = [govtresponse_df]
else:
    govtresponse_df_months = []
    for month in tqdm(govtresponse_df.timeperiod.unique()):
        govtresponse_df_month = govtresponse_df[govtresponse_df.timeperiod == month]
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()
        # Fit scaler to the data and transform it
        govtresponse_df_month[government_response_vars] = scaler.fit_transform(govtresponse_df_month[government_response_vars])

        # Sum scaled exposure vars
        govtresponse_df_month['sum'] = govtresponse_df_month[government_response_vars].sum(axis=1)

        govtresponse_df_month['government-response'] = _sigma_ladder(govtresponse_df_month['sum'])

        govtresponse_df_months.append(govtresponse_df_month)

govtresponse = pd.concat(govtresponse_df_months)
master_variables = master_variables.merge(govtresponse[['timeperiod', 'object_id', 'government-response']],
                       on = ['timeperiod', 'object_id'])

master_variables.to_csv(path+'/RiskScoreModel/data/factor_scores_l1_government-response.csv', index=False)
