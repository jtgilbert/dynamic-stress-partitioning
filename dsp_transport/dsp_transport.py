# imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
from tqdm import tqdm
import argparse
import logging


class FractionalTransport:
    """
    Create an instance of fractional transport calculations for a given stream reach. This class is designed to be
    used specifically with the tables in the Input_data folder of this repository. The standalone function
    'calculate_transport' can be used to apply this transport function to other applications.
    """

    def __init__(self, stream_id: str, reach_slope: float, discharge_interval: int, minimum_fraction: float = 0.005,
                 lwd_factor: int = None):
        """
        Using data from input tables, calculate fractional bedload transport rates for a stream reaching using the
        Dynamic Stress Partitioning method.
        :param stream_id: the name of the stream from the input tables
        :param reach_slope: the reach averaged slope (m/m)
        :param discharge_interval: the time (s) between each discharge value in the input table
        :param minimum_fraction: a minimum fraction of the bed to assign to each half-phi grain size interval (optional)
        :param lwd_factor: a factor (1-3) to account for density of lwd and its affect on incipient sediment motion
        (optional)
        """

        geom_df = pd.read_csv('../Input_data/hydraulic_geometry.csv')
        d_df = pd.read_csv('../Input_data/grain_size.csv')

        # pull the selected stream id from the hydraulic geometry csv
        if stream_id not in geom_df.stream_id.unique():
            raise Exception(f'stream_id: {stream_id} does not exist in hydraulic_geometry.csv')
        else:
            self.geom_df = geom_df[geom_df['stream_id'] == stream_id]
            if len(self.geom_df) < 3:
                raise Exception('stream must contain 3 or more hydraulic geometry measurements')

        # pull the selected stream id from the grain size distribution csv
        if stream_id not in d_df.stream_id.unique():
            raise Exception(f'stream id: {stream_id} does not exist in grain_size.csv')
        else:
            self.d_df = d_df[d_df['stream_id'] == stream_id]

        self.q_df = pd.read_csv('../Input_data/discharge.csv')

        #
        self.s = reach_slope
        self.minimum_fraction = minimum_fraction
        self.discharge_interval = discharge_interval
        self.lwd_factor = lwd_factor

        # Calculate grain size characteristics
        self.d50 = np.percentile(self.d_df.D, 50)/1000
        self.d84 = np.percentile(self.d_df.D, 84)/1000

        # set up a log file
        # path = os.getcwd()
        if not os.path.isdir('../Outputs/'):
            os.mkdir('../Outputs/')
        if not os.path.isdir(f'../Outputs/{stream_id}'):
            os.mkdir(f'../Outputs/{stream_id}')
        if os.path.isfile(f'../Outputs/{stream_id}/{stream_id}.log'):
            os.remove(f'../Outputs/{stream_id}/{stream_id}.log')

        filename = f'../Outputs/{stream_id}/{stream_id}.log'
        logging.basicConfig(filename=filename, format='%(levelname)s:%(message)s', level=logging.DEBUG)
        logging.info('Fractional transport rates calculated for stream: %s using Gilbert dynamic'
                     ' shear stress partitioning method', stream_id)
        logging.info('Reach-average slope: %s', str(reach_slope))
        logging.info('Median grain size (D50): %s', str(self.d50))
        logging.info(f'D84: %s', str(self.d84))

        # get proportion for each grain size fraction
        self.d_fractions = self.grain_sizes()

        # To be able to calculate depth for each discharge (used to get V0)
        self.h_coef, self.h_exponent = self.h_from_q_params()
        logging.info('depth equation: h = %s * Q ^ %s', str(self.h_coef), str(self.h_exponent))

        # to find width for any depth
        self.w_coef, self.w_exponent = self.width_from_depth_params()
        logging.info('width equation: w = %s * h ^ %s', str(self.w_coef), str(self.w_exponent))

        # set up output table - columns:
        q = self.q_df['Q']
        d = self.d_fractions.keys()
        iterables = [q, d]
        index = pd.MultiIndex.from_product(iterables, names=['Q', 'D'])
        zeros = np.zeros((len(q)*len(d), 3))
        self.out_df = pd.DataFrame(zeros, index=index, columns=['qb (kg/m/s)', 'Qb(kg/s)', 'Yield (kg)'])
        # self.out_df = out_df.sort_index()

        # run calculations
        print('Calculating transport')
        self.find_fractional_transport()

        # save output table
        print('Saving output file')
        self.out_df.to_csv(f'../Outputs/{stream_id}/{stream_id}_qb.csv')

    def grain_sizes(self):
        """
        creates a dictionary with each grain size range and the fraction of the bed in each bin
        :return:
        """

        # set up output dictionary which associates a fraction value with each size range
        d_dict = {}

        # set up the size ranges
        intervals = [[0.5,1],[1,2],[2,4],[4,8],[8,11.3],[11.3,16],[16,22.6],[22.6,32],[32,45],[45,64],[64,91],
                     [91,128],[128,181],[181,256],[256,5000]]

        # append values to the dictionary
        for i in intervals:
            count = 0
            for d in self.d_df.index:
                if i[0] <= (self.d_df.loc[d, 'D']) < i[1]:
                    count += 1
            d_dict[i[0]/1000] = count/len(self.d_df)
            if self.minimum_fraction is not None:
                if count/len(self.d_df) < self.minimum_fraction and i[0] <= 4:
                    d_dict[i[0]/1000] = self.minimum_fraction

        # correct for a total > 1 due to enforcing a minimum fraction (SHOULD THIS CHANGE D50??)
        tot = 0
        for i in d_dict:
            tot += d_dict[i]
        if tot > 1:
            for x in d_dict:
                if d_dict[x] == max(d_dict.values()):
                    d_dict[x] = d_dict[x] - (tot - 1)

        logging.info('grain size fractions (size (m): Fraction): %s', str(d_dict))

        tmp_vals = {}
        for key, value in d_dict.items():
            tmp_vals[key] = (value * (key/self.d50)**2)
        denom = np.sum(list(tmp_vals.values()))

        d_dict_adj = {}
        for key, value in d_dict.items():
            d_dict_adj[key] = tmp_vals[key] / denom

        updatekeys = [0.0005, 0.001, 0.002]
        update_val = d_dict_adj[0.0005] + d_dict_adj[0.001] + d_dict_adj[0.002]
        for i in updatekeys:
            d_dict_adj[i] = update_val

        logging.info('grain size fractions adjusted for area: %s', str(d_dict_adj))

        return d_dict_adj

    # use function with hydraulic geometry data to come up with relationship width as function of depth
    # this (as is now) assumes a power function relationship between width and depth
    def width_from_depth_params(self):  # maybe change to do two types of regression compare r2 and use the better
        """
        comes up with the coefficient and intercept for a linear regression of width as a function depth
        (from measurements)
        :return:
        """

        x_data = np.array(np.log(self.geom_df['h'])).reshape(-1, 1)
        y_data = np.array(np.log(self.geom_df['w']))
        lr = LinearRegression()
        lr.fit(x_data, y_data)
        print('depth-width r2: ', str(lr.score(x_data, y_data)))

        logging.info('r-squared for width-depth relationship: %s', str(lr.score(x_data, y_data)))
        return np.exp(lr.intercept_), lr.coef_[0]

    def calc_w_from_h(self, h):
        """
        Calculate channel width as a function of depth based on calculated parameters
        :param h: average flow depth
        :return:
        """

        return self.w_coef*h**self.w_exponent

    def ferguson_vpe(self, h):
        """
        The Ferguson variable power flow resistance function; is used to create an initial flow velocity estimate
        for the optimization that calculates depth (h_i)
        :param h: average flow depth
        :return:
        """

        v_est = ((6.5*(h/self.d84)) / (((h/self.d84)**(5/3)+(6.5/2.5)**2)**0.5))*(9.81*h*self.s)**0.5
        return v_est

    def h_from_q_params(self):
        """
        Finds the linear regression parameters to calculate depth as a function of discharge (from measurements)
        :return:
        """
        q = np.array(np.log(self.geom_df['Q'])).reshape(-1, 1)
        h = np.array(np.log(self.geom_df['h']))
        lr = LinearRegression()
        lr.fit(q, h)
        print('discharge-depth r2: ', str(lr.score(q, h)))

        logging.info('r-squared for discharge-depth relationship: %s', str(lr.score(q, h)))
        return np.exp(lr.intercept_), lr.coef_[0]

    def calc_h_from_q(self, q):
        """
        Calculates average flow depth as a function of discharge using calculated parameters
        :param q: discharge
        :return:
        """

        return self.h_coef*q**self.h_exponent

    def tau_gc_star_i(self, d, lwd_factor=None):
        """
        Calculate the value for critical grain Shields stress for a grain size fraction
        :param d: the grain size
        :param lwd_factor: a large wood factor to account for effects of large wood on hydraulics
        :return:
        """

        roughness = self.d84/self.d50
        if roughness <= 2:
            coef = 0.029
        else:
            coef = 0.043*np.log(roughness)-0.0005

        if lwd_factor is None:
            return coef * (d/self.d50) ** -0.67
        elif lwd_factor == 1:
            return (coef * (d/self.d50) ** -0.67) + 0.01
        elif lwd_factor == 2:
            return (coef * (d / self.d50) ** -0.67) + 0.02
        elif lwd_factor == 3:
            return (coef * (d / self.d50) ** -0.67) + 0.03

    def tau_g_star_i(self, h_i, d):
        """
        Calculate the grain Shields stress for a given grain size
        :param h_i: solved grain depth
        :param d: grain size
        :return:
        """

        return (9810*h_i*self.s)/(1650*9.81*d)

    def err(self, v, d, q_obs):
        h = d ** 0.25 * ((v ** 1.5 / (9.81 * self.s) ** 0.75) / 22.627)
        w = self.calc_w_from_h(h)
        q_pred = v * h * w
        err = (q_obs - q_pred) ** 2

        return err

    def calc_hi(self, d, q, h):
        """
        solve for depth if only mobile fractions were causing roughness
        :param d: grain size
        :param q: discharge
        :param h: depth
        :return:
        """
        v0 = self.ferguson_vpe(h)*2

        res = minimize(self.err, v0, args=(d, q))
        if res.x[0] <= 0.01:
            print('Very low or negative velocity solution (< 0.01 m/s)')
            logging.info(f'Very low or negative velocity solution at flow: {q}, size: {d}')
        #    res.x[0] = 0.01
        h_adj = d ** 0.25 * ((res.x[0] ** 1.5 / (9.81 * self.s) ** 0.75) / 22.627)

        return h_adj

    def find_fractional_transport(self):
        """
        Finds fractional transport rates for each fraction in the bed for each flow in the discharge table
        :return:
        """

        logging.info(f'Calculations started: %s', str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

        for i in tqdm(self.q_df.index):
            q = self.q_df.loc[i, 'Q']
            h = self.calc_h_from_q(q)

            for d in self.d_fractions.keys():
                tau_gc_star_i = self.tau_gc_star_i(d, self.lwd_factor)
                hi = self.calc_hi(d, q, h)
                w = self.calc_w_from_h(h)
                tau_g_star_i = self.tau_g_star_i(hi, d)
                ratio = tau_g_star_i/tau_gc_star_i
                min_ratio = (0.02*(d/self.d50)**-0.67)/tau_gc_star_i
                if ratio < min_ratio:
                    ratio = 0
                if ratio < 2:
                    wi_star = 0.0002 * ratio**13
                else:
                    wi_star = 100 * (1 - (1.348 / ratio**2))**10

                # convert from wi_star to qs
                q_b_vol = (wi_star * self.d_fractions[d] * (9.81 * h * self.s)**(3/2)) / (1.65 * 9.81)
                q_b_mass = q_b_vol * 2650
                Qb = q_b_mass * w
                tot = Qb*self.discharge_interval

                # add result to output table
                self.out_df.loc[q, d] = [q_b_mass, Qb, tot]
                # self.out_df.loc[len(self.out_df)]=[q, d, q_b, Qb]

        logging.info('Calculations ended: %s', str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        logging.info("Total bedload transport: %s kg", str(self.out_df['Yield (kg)'].sum()))

        return


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('stream_id', help='The name of the stream as entered in the Input_data csv files', type=str)
    parser.add_argument('reach_slope', help='A value for the reach averaged slope', type=float)
    parser.add_argument('discharge_interval', help='The time (in seconds) between each discharge measurement in the '
                                                   'discharge csv file', type=int)
    parser.add_argument('--minimum_fraction', help='(optional) A minimum fraction to assign to fine grain size '
                                                   'classes', type=float, default=0.005)
    parser.add_argument('--lwd_factor', help='Alters the critical shields stress to account for the affects of large '
                                             'wood, None is no wood present, 1 is some scattered pieces, 2 is wood '
                                             'throughout the reach and 3 is jams present', type=int, default=None)

    args = parser.parse_args()

    FractionalTransport(args.stream_id, args.reach_slope, args.discharge_interval, args.minimum_fraction,
                        args.lwd_factor)


if __name__ == '__main__':
    main()
