import argparse
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
    #err = (np.percentile(dist, 16) - np.log2(d16)) ** 2 + (np.percentile(dist, 84) - np.log2(d84)) ** 2
    err = (np.percentile(dist, 84) - np.log2(d84)) ** 2

    return err


def count_to_prop(count_in: str, grain_size_col: str, fine_frac: float = None):

    df = pd.read_csv(count_in)

    phi_vals = [np.log2(i) for i in list(df[grain_size_col])]

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

    tmp_counts = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 3.5: 0, 4: 0, 4.5: 0, 5: 0, 5.5: 0, 6: 0, 6.5: 0, 7: 0,
                 7.5: 0, 8: 0, 8.5: 0, 9: 0}
    for i in new_data:
        if -1 <= i < 0:
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
        elif i >= 9:
            tmp_counts[9] += 1

    if fine_frac:
        min_ct = int(fine_frac * len(new_data))
        added = 0
        for key, val in tmp_counts.items():
            if val < min_ct and key in [-1, 0, 1, 2]:
                tmp_counts[key] = min_ct
                added += min_ct
            if val < int(min_ct/2) and key in [3, 3.5]:
                tmp_counts[key] = int(min_ct/2)
                added += int(min_ct/2)

        keys_with_vals = [key for key, val in tmp_counts.items() if val > (added/3) and key not in [-1,0,1,2,3,3.5]]
        if tmp_counts[8.5] != 0 and 8.5 not in keys_with_vals:
            keys_with_vals.append(8.5)
        if tmp_counts[9] != 0 and 9 not in keys_with_vals:
            keys_with_vals.append(9)

        #subtr_value = int(added / len(keys_with_vals))
        while added > 0:
            for k, v in tmp_counts.items():
                if k in keys_with_vals and v != 0:
                    if k == 9 and v >= 4:
                        tmp_counts[k] = v - 4
                        added -= 4
                    elif k == 8.5 and v >= 3:
                        tmp_counts[k] = v - 3
                        added -= 3
                    elif k == 8 and v >= 2:
                        tmp_counts[k] = v - 2
                        added -= 2
                    else:
                        tmp_counts[k] = v - 1
                        added -= 1

    tmp_props = {}
    for key, val in tmp_counts.items():
        tmp_props[key] = (val / np.sum(list(tmp_counts.values()))) * (2**key / d50_out)**2

    out_props = {}
    for key, val in tmp_props.items():
        out_props[key] = val / np.sum(list(tmp_props.values()))

    return out_props


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('count_in', help='Input csv file containing pebble count data', type=str)
    parser.add_argument('grain_size_col', help='The name of the column containing pebble counts in the input csv',
                        type=str)
    parser.add_argument('--fine_frac', help='A minimum fraction of fines (<= 4mm) to include in the count',
                        type=float)

    args = parser.parse_args()

    count_to_prop(args.count_in, args.grain_size_col, args.fine_frac)


if __name__ == '__main__':
   main()

# count_to_prop('../Input_data/woods_generated2.csv', 'D')
