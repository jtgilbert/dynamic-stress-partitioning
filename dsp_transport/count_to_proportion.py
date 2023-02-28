import argparse
from typing import Dict
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import fmin

def med_err(loc, a, scale, med_obs):
    dist = stats.skewnorm(a, loc, scale).rvs(10000)
    med_est = np.percentile(dist, 50)
    err = (np.log2(med_obs) - med_est) ** 2

    return err


def std_err(scale, a, loc, d16, d84):
    dist = stats.skewnorm(a, loc, scale).rvs(10000)
    err = (np.percentile(dist, 16) - np.log2(d16)) ** 2 + (np.percentile(dist, 84) - np.log2(d84)) ** 2

    return err


def count_to_prop(count_in: str, grain_size_col: str, min_vals: Dict[int, float] = None):

    df = pd.read_csv(count_in)

    phi_vals = []
    for i in df.index:
        if df.loc[i, grain_size_col] < 0.5:
            phi_vals.append(-2)
        elif 0.5 <= df.loc[i, grain_size_col] < 1:
            phi_vals.append(-1)
        elif 1 <= df.loc[i, grain_size_col] < 2:
            phi_vals.append(0)
        elif 2 <= df.loc[i, grain_size_col] < 4:
            phi_vals.append(1)
        elif 4 <= df.loc[i, grain_size_col] < 8:
            phi_vals.append(2)
        elif 8 <= df.loc[i, grain_size_col] < 11.3:
            phi_vals.append(3)
        elif 11.3 <= df.loc[i, grain_size_col] < 16:
            phi_vals.append(3.5)
        elif 16 <= df.loc[i, grain_size_col] < 22.6:
            phi_vals.append(4)
        elif 22.6 <= df.loc[i, grain_size_col] < 32:
            phi_vals.append(4.5)
        elif 32 <= df.loc[i, grain_size_col] < 45:
            phi_vals.append(5)
        elif 45 <= df.loc[i, grain_size_col] < 64:
            phi_vals.append(5.5)
        elif 64 <= df.loc[i, grain_size_col] < 90:
            phi_vals.append(6)
        elif 90 <= df.loc[i, grain_size_col] < 128:
            phi_vals.append(6.5)
        elif 128 <= df.loc[i, grain_size_col] < 180:
            phi_vals.append(7)
        elif 180 <= df.loc[i, grain_size_col] < 256:
            phi_vals.append(7.5)
        elif 256 <= df.loc[i, grain_size_col] < 360:
            phi_vals.append(8)
        elif 360 <= df.loc[i, grain_size_col] < 512:
            phi_vals.append(8.5)
        elif 512 <= df.loc[i, grain_size_col] < 725:
            phi_vals.append(9)
        elif 725 <= df.loc[i, grain_size_col] < 1024:
            phi_vals.append(9.5)
        elif df.loc[i, grain_size_col] >= 1024:
            phi_vals.append(10)

    params = stats.skewnorm.fit(phi_vals)
    a, loc, scale = params[0], params[1], params[2]

    d50 = np.percentile(df[grain_size_col], 50)
    d16 = np.percentile(df[grain_size_col], 16)
    d84 = np.percentile(df[grain_size_col], 84)

    res = fmin(med_err, loc, args=(a, scale, d50))
    loc_opt = res[0]

    res2 = fmin(std_err, scale, args=(a, loc_opt, d16, d84))
    scale_opt = res2[0]

    new_data = stats.skewnorm(a, loc_opt, scale_opt).rvs(500)

    d50_out = 2**np.percentile(new_data, 50)

    tmp_counts = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 3.5: 0, 4: 0, 4.5: 0, 5: 0, 5.5: 0, 6: 0, 6.5: 0, 7: 0,
                 7.5: 0, 8: 0, 8.5: 0, 9: 0, 9.5: 0, 10: 0}
    for i in new_data:
        if -2 <= i < -1:
            tmp_counts[-2] += 1
        elif -1 <= i < 0:
            tmp_counts[-1] += 1
        elif 0 <= i < 1:
            tmp_counts[0] += 1
        elif 1 <= i < 2:
            tmp_counts[1] += 1
        elif 2 <= i < 3:
            tmp_counts[2] += 1
        elif 3 <= i < 3.5:
            tmp_counts[3] += 1
        elif 3.5 <= i < 4:
            tmp_counts[3.5] += 1
        elif 4 <= i < 4.5:
            tmp_counts[4] += 1
        elif 4.5 <= i < 5:
            tmp_counts[4.5] += 1
        elif 5 <= i < 5.5:
            tmp_counts[5] += 1
        elif 5.5 <= i < 6:
            tmp_counts[5.5] += 1
        elif 6 <= i < 6.5:
            tmp_counts[6] += 1
        elif 6.5 <= i < 7:
            tmp_counts[6.5] += 1
        elif 7 <= i < 7.5:
            tmp_counts[7] += 1
        elif 7.5 <= i < 8:
            tmp_counts[7.5] += 1
        elif 8 <= i < 8.5:
            tmp_counts[8] += 1
        elif 8.5 <= i < 9:
            tmp_counts[8.5] += 1
        elif 9 <= i < 9.5:
            tmp_counts[9] += 1
        elif 9.5 <= i < 10:
            tmp_counts[9.5] += 1
        elif i >= 10:
            tmp_counts[10] += 1

    tmp_props = {}
    for key, val in tmp_counts.items():
        tmp_props[key] = (val / np.sum(list(tmp_counts.values()))) * (2**key / d50_out)**2

    out_props = {}
    for key, val in tmp_props.items():
        out_props[key] = val / np.sum(list(tmp_props.values()))

    if min_vals:
        d_vals = []
        for bin, prop in min_vals.items():
            count = 0
            for size, proportion in out_props.items():
                if size > np.max(list(min_vals.keys())) and proportion > 0.1:
                    count += 1
                    d_vals.append(size)
            subtract = prop / count
            for s, p in out_props.items():
                if s > np.max(list(min_vals.keys())) and s in d_vals:
                    out_props[s] = out_props[s] - subtract
            out_props[bin] = prop

    return out_props


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('count_in', help='Input csv file containing pebble count data', type=str)
    parser.add_argument('grain_size_col', help='The name of the column containing pebble counts in the input csv',
                        type=str)
    parser.add_argument('--min_vals', help='A dictionary of the form {-phi size: proportion} to add to grain size'
                                           'fractions. This functionality is to add specified proportions of fine'
                                           'fractions that would be missed in pebble counts', type=Dict[int, float])

    args = parser.parse_args()

    count_to_prop(args.count_in, args.grain_size_col, args.min_vals)


#if __name__ == '__main__':
#    main()

count_to_prop('/media/jordan/Elements/Geoscience/Bitterroot/transport_datasets/fourthofjuly_d.csv', 'D',
              {-1: 0.01, 0: 0.01, 1: 0.01, 2: 0.01})
