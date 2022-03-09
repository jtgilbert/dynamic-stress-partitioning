# imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


class FractionalTransport:
    """
    Create an instance of fractional transport calculations for a given stream reach.
    """

    def __init__(self, stream_id):

        geom_df = pd.read_csv('Input_data/hydraulic_geometry.csv')
        d_df = pd.read_csv('Input_data/grain_size.csv')
        q_df = pd.read_csv('Input_data/discharge.csv') # do I make this a table you add to, ie has stream name, or you just fill out every time...?

        # pull the selected stream id from the hydraulic geometry csv
        if stream_id not in geom_df.stream_id.unique():
            raise Exception('stream id: {} does not exist in hydraulic_geometry.csv'.format(stream_id))
        else:
            self.geom_df = geom_df[geom_df['stream_id'] == stream_id]

        # pull the selected stream id from the grain size distribution csv
        if stream_id not in d_df.stream_id.unique():
            raise Exception('stream id: {} does not exist in grain_size.csv'.format(stream_id))
        else:
            self.d_df = d_df[d_df['stream_id'] == stream_id]

        self.d50 = np.percentile(self.d_df, 50)
        self.d84 = np.percentile(self.d_df, 84)
        self.d_fractions = self.grain_sizes()

    def grain_sizes(self):
        # set up output dictionary which associates a fraction value with each size range
        d_dict = {}

        # set up the size ranges
        intervals = [[0,0.5],[0.5,1],[1,2],[2,4],[4,6],[6,8],[8,12],[12,16],[16,24],[24,32],[32,48],[48,64],[64,96],[96,128],
                     [128,192],[192,256]]

        # append values to the dictionary
        for i in intervals:
            count = 0
            for d in self.d_df:
                if i[0] < d <= i[1]:
                    count += 1
            d_dict[str(i)] = count/len(self.d_df)

        return d_dict # dictionary {half phi: fraction}

    # use function with hydraulic geometry data to come up with relationship width as function of depth
    def calc_width_params(self):
        x_data = np.array(self.geom_df['h']).reshape(-1,1)
        y_data = np.array(self.geom_df['w'])
        lr = LinearRegression()
        lr.fit(x_data, y_data)
        print('depth-width R2: ', str(lr.score(x_data, y_data)))

        return lr.coef_, lr.intercept_

    def ferguson_vpe(self, h, s):
        
        v_est = ((6.5*(h/self.d84)) / (((h/self.d84)**(5/3)+(6.5/2.5)**2)**0.5))*(9.81*h*s)**0.5
        return v_est

    # use function to find D50 and Fi for each size class i from grain size data

    # UPDATE THESE FUNCTIONS FOR USE IN THIS CLASS
    def err(v, d, s, chan_type, coef, intercept, q_obs):
        h = d ** 0.25 * ((v ** 1.5 / (9.81 * s) ** 0.75) / 22.627)
        if chan_type == 'confined':
            w = coef * np.log(h) + intercept
        elif chan_type == 'floodplain':
            w = coef * np.exp(intercept * h)
        q_pred = v * h * w
        err = (q_obs - q_pred) ** 2

            return err

    def get_gs_ratio(d, S, chan_type, coef, intercept, Q, D50):
        v0 = 5

        res = minimize(err, v0, args=(d, S, chan_type, coef, intercept, Q))
        h_adj = d ** 0.25 * ((res.x[0] ** 1.5 / (9.81 * S) ** 0.75) / 22.627)
        tau_g_star = (9180 * h_adj * S) / (1650 * 9.81 * d)
        ratio = tau_g_star / (0.038 * (d / D50) ** -0.65)  #
        if ratio <= 0:
            ratio = 0

        return ratio, res.x[0], h_adj

