# imports
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class FractionalTransport:
    """
    Create an instance of fractional transport calculations for a given stream reach.
    """

    def __init__(self, stream_id):

        geom_df = pd.read_csv('Input_data/hydraulic_geometry.csv')
        d_df = pd.read_csv('Input_data/grain_size.csv')

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

        # use function with hydraulic geometry data to come up with relationship width as function of depth

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

