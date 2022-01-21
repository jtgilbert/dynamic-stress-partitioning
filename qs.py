# imports
import numpy as np
import pandas as pd


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
        