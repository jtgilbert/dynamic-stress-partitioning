# imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os


class FractionalTransport:
    """
    Create an instance of fractional transport calculations for a given stream reach.
    """

    def __init__(self, stream_id, reach_slope, minimum_fraction=0.01):
        """
        Calculates fractional bedload transport rates using the dynamic shear stress partitioning approach of Gilbert
        :param stream_id: The name of the stream in the input data tables (str)
        :param reach_slope: The reach averaged slope - unitless e.g., m/m - (float)
        """

        geom_df = pd.read_csv('Input_data/hydraulic_geometry.csv')
        d_df = pd.read_csv('Input_data/grain_size.csv')

        # pull the selected stream id from the hydraulic geometry csv
        if stream_id not in geom_df.stream_id.unique():
            raise Exception('stream_id: {} does not exist in hydraulic_geometry.csv'.format(stream_id))
        else:
            self.geom_df = geom_df[geom_df['stream_id'] == stream_id]

        # pull the selected stream id from the grain size distribution csv
        if stream_id not in d_df.stream_id.unique():
            raise Exception('stream id: {} does not exist in grain_size.csv'.format(stream_id))
        else:
            self.d_df = d_df[d_df['stream_id'] == stream_id]

        self.q_df = pd.read_csv('Input_data/discharge.csv')

        #
        self.s = reach_slope
        self.minimum_fraction = minimum_fraction

        # Calculate grain size characteristics
        self.d50 = np.percentile(self.d_df.D, 50)/1000
        self.d84 = np.percentile(self.d_df.D, 84)/1000

        # set up a log file
        path = os.getcwd()
        metatxt = path + '/Outputs/{}_log.txt'.format(stream_id)
        self.md = open(metatxt, 'w+')
        init_lines = ['Fractional transport rates calculated for stream: {} using Gilbert dynamic shear stress '
                      'partitioning method \n \n'.format(stream_id),
                      'Reach-average slope: {} \n'.format(str(reach_slope)),
                      'Median grain size (D50): {}m \n'.format(str(self.d50)),
                      'D84: {}m \n \n'.format(str(self.d84))]
        self.md.writelines(init_lines)

        # get proportion for each grain size fraction
        self.d_fractions = self.grain_sizes()

        # To be able to calculate depth for each discharge (used to get V0)
        self.h_coef, self.h_exponent = self.h_from_q_params()

        # to find width for any depth
        self.w_coef, self.w_intercept = self.width_from_depth_params()

        # set up output table - columns:
        q = self.q_df['Q']
        d = self.d_fractions.keys()
        iterables = [q, d]
        index = pd.MultiIndex.from_product(iterables, names=['Q', 'D'])
        zeros = np.zeros((len(q)*len(d), 2))
        self.out_df = pd.DataFrame(zeros, index=index, columns=['qb', 'Qb'])
        # self.out_df = pd.DataFrame(columns=['Q', 'D', 'qb'])

        # run calculations
        self.find_fractional_transport()

        # save output table
        self.out_df.to_csv('Outputs/{}_qb.csv'.format(stream_id))
        self.md.close()

    def grain_sizes(self):
        """
        creates a dictionary with each grain size range and the fraction of the bed in each bin
        :return:
        """

        # set up output dictionary which associates a fraction value with each size range
        d_dict = {}

        # set up the size ranges
        intervals = [[0.5,1],[1,2],[2,4],[4,6],[6,8],[8,12],[12,16],[16,24],[24,32],[32,48],[48,64],[64,96],
                     [96,128],[128,192],[192,256]]

        # append values to the dictionary
        for i in intervals:
            count = 0
            for d in self.d_df.index:
                if i[0] < (self.d_df.loc[d, 'D']) <= i[1]:
                    count += 1
            if count > 0:
                d_dict[i[0]/1000] = count/len(self.d_df)
            else:
                d_dict[i[0]/1000] = self.minimum_fraction

        self.md.writelines('grain size fractions: {} \n \n'.format(str(d_dict)))

        return d_dict

    # use function with hydraulic geometry data to come up with relationship width as function of depth
    # this (as is now) assumes a linear relationship between width and depth (instead of choosing between log and exp)
    def width_from_depth_params(self): # maybe change to do two types of regression compare r2 and use the better
        """
        comes up with the coefficient and intercept for a linear regression of width as a function depth
        (from measurements)
        :return:
        """

        x_data = np.array(self.geom_df['h']).reshape(-1, 1)
        y_data = np.array(self.geom_df['w'])
        lr = LinearRegression()
        lr.fit(x_data, y_data)
        print('depth-width R2: ', str(lr.score(x_data, y_data)))

        self.md.writelines('r-squared for width-depth relationship: {} \n'.format(str(lr.score(x_data, y_data))))
        return lr.coef_[0], lr.intercept_

    def calc_w_from_h(self, h):
        """
        Calculate channel width as a function of depth based on calculated parameters
        :param h: average flow depth
        :return:
        """

        return self.w_coef*h+self.w_intercept

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

        self.md.writelines('r-squared for discharge-depth relationship: {} \n'.format(str(lr.score(q, h))))
        return np.exp(lr.intercept_), lr.coef_[0]

    def calc_h_from_q(self, q):
        """
        Calculates average flow depth as a function of discharge using calculated parameters
        :param q: discharge
        :return:
        """

        return self.h_coef*q**self.h_exponent

    def tau_gc_star_i(self, d):

        return 0.038*(d/self.d50)**-0.65

    def tau_g_star_i(self, h_i, d):

        return (9810*h_i*self.s)/(1.65*9.81*d)

    def err(self, v, d, q_obs):
        h = d ** 0.25 * ((v ** 1.5 / (9.81 * self.s) ** 0.75) / 22.627)
        w = self.calc_w_from_h(h)
        q_pred = v * h * w
        err = (q_obs - q_pred) ** 2

        return err

    def calc_hi(self, d, q, h):
        v0 = self.ferguson_vpe(h)*2

        res = minimize(self.err, v0, args=(d, q))
        h_adj = d ** 0.25 * ((res.x[0] ** 1.5 / (9.81 * self.s) ** 0.75) / 22.627)

        return h_adj

    def find_fractional_transport(self):
        """
        Finds fractional transport rates for each fraction in the bed for each flow in the discharge table
        :return:
        """

        self.md.writelines('\n Calculations started: {} \n'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

        for i in self. q_df.index:
            q = self.q_df.loc[i, 'Q']
            h = self.calc_h_from_q(q)
            for d in self.d_fractions.keys():
                tau_gc_star_i = self.tau_gc_star_i(d)
                hi = self.calc_hi(d, q, h)
                w = self.calc_w_from_h(h)
                tau_g_star_i = self.tau_g_star_i(hi, d)
                ratio = tau_g_star_i/tau_gc_star_i
                if ratio < 0:
                    ratio = 0
                if ratio < 2:
                    wi_star = 0.0008*ratio**7.5
                else:
                    wi_star = 14*(1-(1.11/ratio**0.8))**4.5

                # convert from wi_star to qs
                q_b = (wi_star*self.d_fractions[d]*(9.81*h*self.s)**(3/2)) / (1.65*9.81)
                Qb = q_b * w

                # add result to output table
                self.out_df.loc[q, d] = [q_b, Qb]
                # self.out_df.loc[len(self.out_df)]=[q, d, q_b, Qb]

        self.md.writelines('Calculations ended: {} \n'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

        return



