from scipy import stats
from scipy.optimize import fmin
import numpy as np
import pandas as pd
import argparse


def med_err(loc, scale, med_obs):
    dist = stats.norm(loc, scale).rvs(10000)
    dist = dist**2
    dist = dist[dist > 0.0005]  # gets rid of sub-zero values (makes it sand and greater)
    med_est = np.percentile(dist, 50)
    err = (med_obs - med_est) ** 2

    return err


def std_err(scale, loc, d16, d84):
    dist = stats.norm(loc, scale).rvs(10000)
    dist = dist**2
    dist = dist[dist > 0.0005]
    err = (np.percentile(dist, 16) - d16) ** 2 + (np.percentile(dist, 84) - d84) ** 2

    return err


def generate_distribution(csv_out, csv_in=None, d50=None, d16=None, d84=None):

    if csv_in is None and d50 is None:
        raise Exception('If no grain count data is provided (csv_in) you must provide D50, D16, and D84 values')
    if csv_in is None and d16 is None:
        raise Exception('If no grain count data is provided (csv_in) you must provide D50, D16, and D84 values')
    if csv_in is None and d84 is None:
        raise Exception('If no grain count data is provided (csv_in) you must provide D50, D16, and D84 values')

    if csv_in is None:
        loc, scale = 0.2, 0.05
        d50_p, d16_p, d84_p = d50 / 1000, d16 / 1000, d84 / 1000

    else:
        indata = pd.read_csv(csv_in)
        indata_trans = np.sqrt(indata['D'] / 1000)
        k2, pval = stats.normaltest(indata_trans)
        if pval < 1e-3:
            print('p-value: ', pval, ' indicates that the transformed grain size distribution may not be normal')

        params = stats.norm.fit(indata_trans)
        loc, scale = params[0], params[1]

        if d50 is not None:
            d50_p = d50 / 1000
        else:
            d50_p = np.percentile(indata, 50) / 1000
        if d16 is not None:
            d16_p = d16 / 1000
        else:
            d16_p = np.percentile(indata, 16) / 1000
        if d84 is not None:
            d84_p = d84 / 1000
        else:
            d84_p = np.percentile(indata, 84) / 1000

    print(f'Input D50: {d50_p * 1000}')
    print(f'Input D16: {d16_p * 1000}')
    print(f'Input D84: {d84_p * 1000}')

    res = fmin(med_err, loc, args=(scale, d50_p))
    loc_opt = res[0]

    res2 = fmin(std_err, scale, args=(loc_opt, d16_p, d84_p))
    scale_opt = res2[0]

    new_data = stats.norm(loc_opt, scale_opt).rvs(1000)
    new_data = ((new_data**2)*1000)
    new_data = new_data[new_data >= 0.5]

    print(f'generated {len(new_data)} grain size measurements')
    print(f'new grain size distribution D50: {np.percentile(new_data, 50)}')
    print(f'new grain size distribution D16: {np.percentile(new_data, 16)}')
    print(f'new grain size distribution D84: {np.percentile(new_data, 84)}')

    df = pd.DataFrame({'D': new_data})
    df.to_csv(csv_out, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_out', help='Path to save the output csv of generated grain size data', type=str)
    parser.add_argument('--csv_in', help='Path to a csv file containing grain size data; used to estimate '
                                         'distribution parameters and find D50, D16, and D84 if not provided.'
                                         'If no input csv is entered, values MUST be provided for D50, D16, and D84'
                                         'The csv should have a column with header "D" containing grain size counts'
                                         'in millimeters',
                        type=str)
    parser.add_argument('--d50', help='A value for D50 (mm). If no input csv is entered, a value for D50 must be '
                                      'provided. If an input csv IS entered, the calculated D50 can be overridden '
                                      'by providing a value here', type=int)
    parser.add_argument('--d16', help='A value for D16 (mm). If no input csv is entered, a value for D16 must be '
                                      'provided. If an input csv IS entered, the calculated D16 can be overridden '
                                      'by providing a value here', type=int)
    parser.add_argument('--d84', help='A value for D84 (mm). If no input csv is entered, a value for D84 must be '
                                      'provided. If an input csv IS entered, the calculated D84 can be overridden '
                                      'by providing a value here', type=int)

    args = parser.parse_args()

    generate_distribution(args.csv_out, args.csv_in, args.d50, args.d16, args.d84)


if __name__ == '__main__':
    main()
