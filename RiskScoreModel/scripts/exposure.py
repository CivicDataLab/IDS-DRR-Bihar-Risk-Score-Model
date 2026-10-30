import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm 
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

master_variables = pd.read_csv(os.getcwd()+r'/RiskScoreModel/data/MASTER_VARIABLES.csv')



exposure_vars = ['total_hhd','sum_population',"sum_aged_population","schools_count","rail_length", #"net_sown_area_in_hac",
                      "road_length", "health_centres_count"
                     ]

# What the 1-5 exposure class is computed from.
#   'varsum'         - the min-max scaled sum of exposure_vars above (default)
#   'sum_population' - population alone, as the Himachal Pradesh model does
EXPOSURE_BASIS = 'varsum'   # 'varsum' | 'sum_population'

# How that magnitude becomes the class.
#   'qcut'  - quantile bins within each month (default)
#   'sigma' - the original one-sided ladder, kept to reproduce the old numbers
#
# The old rule put every block at or below the monthly MEAN into class 1 and
# reserved class 5 for blocks more than 3 standard deviations ABOVE it. Exposure
# is driven by population and household counts, which are heavily right-skewed
# across Bihar's blocks, so "at or below the mean" was the bulk of the state:
# 55.0% of block-months landed in class 1 and only 0.6% ever reached class 5.
# Worse, because the census inputs barely move between months, the class was
# effectively frozen - 529 of 533 blocks never changed exposure class in any of
# the 66 months, so the factor carried almost no information.
#
# Quantile bins within the month give a usable spread and keep exposure
# comparable across months, which is what the TOPSIS step needs from it.
EXPOSURE_SCALE = 'qcut'   # 'qcut' | 'sigma'


def _sigma_ladder(score):
    """Original one-sided ladder: <= mean is 1, then +1 sd steps up to 5."""
    mean = score.mean()
    std = score.std()
    conditions = [
        (score <= mean),
        (score > mean) & (score <= mean + std),
        (score > mean + std) & (score <= mean + 2 * std),
        (score > mean + 2 * std) & (score <= mean + 3 * std),
        (score > mean + 3 * std)
    ]
    return np.select(conditions, [1, 2, 3, 4, 5])


exposure_df = master_variables[exposure_vars + ['timeperiod', 'object_id']]


exposure_df_months = []
for month in tqdm(exposure_df.timeperiod.unique()):
    exposure_df_month = exposure_df[exposure_df.timeperiod == month].copy()
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    # Fit scaler to the data and transform it
    exposure_df_month[exposure_vars] = scaler.fit_transform(exposure_df_month[exposure_vars])
    
    # Sum scaled exposure vars
    
    exposure_df_month['sum'] = exposure_df_month[exposure_vars].sum(axis=1)

    if EXPOSURE_BASIS == 'sum_population':
        exposure_raw = master_variables.loc[exposure_df_month.index, 'sum_population']
    else:
        exposure_raw = exposure_df_month['sum']

    if EXPOSURE_SCALE == 'qcut':
        # duplicates='drop' because many blocks tie; when bins collapse, qcut
        # needs the label count to match the bins that survive.
        binned = pd.qcut(exposure_raw, 5, labels=False, duplicates='drop')
        exposure_df_month['exposure'] = (binned + 1).astype(int)
    else:
        exposure_df_month['exposure'] = _sigma_ladder(exposure_raw)

    exposure_df_months.append(exposure_df_month)

exposure = pd.concat(exposure_df_months)
exposure = exposure.drop_duplicates(subset=['timeperiod','object_id'])


master_variables = master_variables.merge(
    exposure[['timeperiod','object_id','exposure']].drop_duplicates(),
    on=['timeperiod','object_id'],
    how='left'
)

dups = (
    master_variables
    .groupby(['timeperiod','object_id'])
    .size()
    .reset_index(name='count')
)
print(dups['count'].value_counts())

#master_variables = master_variables.merge(exposure[['timeperiod', 'object_id', 'exposure']],
#                       on = ['timeperiod', 'object_id'])
print(master_variables.shape)
master_variables.to_csv(os.getcwd()+r'/RiskScoreModel/data/factor_scores_l1_exposure.csv', index=False)
