from scipy import stats
from scipy.optimize import fmin
import numpy as np
import pandas as pd
import argparse


def med_err(loc, a, scale, med_obs, p_sand=None):
    dist = stats.skewnorm(a, loc, scale).rvs(10000)
    #dist = dist**2
    #dist = dist[dist > 0.0005]  # gets rid of sub-zero values (makes it sand and greater)
    if p_sand:
        num_sand = p_sand * len(dist)
        finesand = np.empty(int(num_sand/2))
        finesand.fill(-1)
        coarsesand = np.empty(int(num_sand/2))
        coarsesand.fill(0)
        dist = np.concatenate([dist, finesand, coarsesand])
    med_est = np.percentile(dist, 50)
    err = (np.log2(med_obs) - med_est) ** 2

    return err


def std_err(scale, a, loc, d16, d84, p_sand=None):
    dist = stats.skewnorm(a, loc, scale).rvs(10000)
    #dist = dist**2
    #dist = dist[dist > 0.0005]
    if p_sand:
        num_sand = p_sand * len(dist)
        finesand = np.empty(int(num_sand/2))
        finesand.fill(-1)
        coarsesand = np.empty(int(num_sand/2))
        coarsesand.fill(0)
        dist = np.concatenate([dist, finesand, coarsesand])
    err = (np.percentile(dist, 16) - np.log2(d16)) ** 2 + (np.percentile(dist, 84) - np.log2(d84)) ** 2

    return err

def skew_err(a, loc, scale, dmax, dmin, p_sand):
    dist = stats.skewnorm(a, loc, scale).rvs(10000)
    if p_sand:
        num_sand = p_sand * len(dist)
        finesand = np.empty(int(num_sand/2))
        finesand.fill(-1)
        coarsesand = np.empty(int(num_sand/2))
        coarsesand.fill(0)
        dist = np.concatenate([dist, finesand, coarsesand])
    err = (np.max(dist) - np.log2(dmax)) ** 2 + (np.min(dist) - np.log2(dmin)) ** 2

    return err

def generate_distribution(csv_out: str, num_obs: int, csv_in: str = None, d50: int = None, d16: int = None,
                          d84: int = None, p_sand: float = None, dmax: int = None):
    """

    :param csv_out: path to store a csv file with generated grain sizes
    :param num_obs: the number of observations to include in the output csv
    :param csv_in: path to a csv containing grain size measurements (optional)
    :param d50: a D50 value for fitting the output grain size distribution (optional if csv_in provided)
    :param d16: a D16 value for fitting the output grain size distribution (optional if csv_in provided)
    :param d84: a D84 value for fitting the output grain size distribution (optional if csv_in provided)
    :param p_sand: a percentage of sand to include in the output
    :param dmax: a maximum grain size value (optional)
    :return: saves a csv file with simulated grain size measurements
    """

    if csv_in is None and d50 is None:
        raise Exception('If no grain count data is provided (csv_in) you must provide D50, D16, and D84 values')
    if csv_in is None and d16 is None:
        raise Exception('If no grain count data is provided (csv_in) you must provide D50, D16, and D84 values')
    if csv_in is None and d84 is None:
        raise Exception('If no grain count data is provided (csv_in) you must provide D50, D16, and D84 values')

    if csv_in is None:
        a, loc, scale = -10, 7.5, 2.5
        d50_p, d16_p, d84_p = d50, d16, d84
        print(f'Input D50: {d50}')
        print(f'Input D16: {d16}')
        print(f'Input D84: {d84}')

    else:
        indata = pd.read_csv(csv_in)
        # indata_trans = np.sqrt(indata['D'] / 1000)
        # k2, pval = stats.normaltest(indata_trans)
        # if pval < 1e-3:
        #     print('p-value: ', pval, ' indicates that the transformed grain size distribution may not be normal')
        in_data_trans = []
        for i in list(indata['D']):
            if i < 2:
                continue
            elif 2 <= i < 4:
                in_data_trans.append(1)
            elif 4 <= i < 8:
                in_data_trans.append(2)
            elif 8 <= i <= 11.3:
                in_data_trans.append(3)
            elif 11.3 <= i < 16:
                in_data_trans.append(3.5)
            elif 16 <= i < 22.6:
                in_data_trans.append(4)
            elif 22.6 <= i < 32:
                in_data_trans.append(4.5)
            elif 32 <= i < 45:
                in_data_trans.append(5)
            elif 45 <= i < 64:
                in_data_trans.append(5.5)
            elif 64 <= i < 90:
                in_data_trans.append(6)
            elif 90 <= i < 128:
                in_data_trans.append(6.5)
            elif 128 <= i < 180:
                in_data_trans.append(7)
            elif 180 <= i < 256:
                in_data_trans.append(7.5)
            elif 256 <= i < 360:
                in_data_trans.append(8)
            elif 360 <= i < 512:
                in_data_trans.append(8.5)
            elif i >= 512:
                in_data_trans.append(9)

        params = stats.skewnorm.fit(in_data_trans)
        a, loc, scale = params[0], params[1], params[2]

        if d50 is not None:
            d50_p = d50
        else:
            d50_p = 2**np.percentile(in_data_trans, 50)
        if d16 is not None:
            d16_p = d16
        else:
            d16_p = 2**np.percentile(in_data_trans, 16)
        if d84 is not None:
            d84_p = d84
        else:
            d84_p = 2**np.percentile(in_data_trans, 84)

        print(f'Data D50: {d50_p}')
        print(f'Data D16: {d16_p}')
        print(f'Data D84: {d84_p}')



    res = fmin(med_err, loc, args=(a, scale, d50_p, p_sand))
    loc_opt = res[0]

    res2 = fmin(std_err, scale, args=(a, loc_opt, d16_p, d84_p, p_sand))
    scale_opt = res2[0]

    new_data = stats.skewnorm(a, loc_opt, scale_opt).rvs(num_obs)

    #new_data = ((new_data**2)*1000)
    #new_data = new_data[new_data >= 0.5]
    if p_sand is not None:
        num_sand = p_sand * len(new_data)
        finesand = np.empty(int(num_sand/2))
        finesand.fill(-0.5)
        coarsesand = np.empty(int(num_sand/2))
        coarsesand.fill(0.5)
        rem_data = new_data[int(num_sand):]
        new_data = np.concatenate([rem_data, finesand, coarsesand])
        # need to remove elements that sand replaces....

    if dmax:
        new_data = new_data[new_data < dmax]

    new_data = 2**new_data

    print(f'generated {len(new_data)} grain size measurements')
    print(f'new grain size distribution D50: {np.percentile(new_data, 50)}')
    print(f'new grain size distribution D16: {np.percentile(new_data, 16)}')
    print(f'new grain size distribution D84: {np.percentile(new_data, 84)}')

    df = pd.DataFrame({'D': new_data})
    df.to_csv(csv_out, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_out', help='Path to save the output csv of generated grain size data', type=str)
    parser.add_argument('num_obs', help='Number of output data points to include', type=int)
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
    parser.add_argument('--p_sand', help='A percentage of sand to include in the output csv', type=float)
    parser.add_argument('--dmax', help='A maximum grain size (mm) for the output distribution. Generated values above'
                                       'this threshold will be discarded', type=int)

    args = parser.parse_args()

    generate_distribution(args.csv_out, args.num_obs, args.csv_in, args.d50, args.d16, args.d84, args.p_sand, args.dmax)


if __name__ == '__main__':
    main()
