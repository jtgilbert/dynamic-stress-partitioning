import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


class CreatePlots:

    def __init__(self, stream_id):

        try:
            self.df = pd.read_csv('../Outputs/{}/{}_qb.csv'.format(stream_id, stream_id))
        except:
            raise Exception('No output csv file that matches stream_id; dsp_transport must be run first')

        if not os.path.isdir('../Outputs/{}/{}_figs'.format(stream_id, stream_id)):
            os.mkdir('../Outputs/{}/{}_figs'.format(stream_id, stream_id))

        self.out_path = '../Outputs/{}/{}_figs'.format(stream_id, stream_id)

        print('Creating plots')
        self.total_yield()
        self.fractional_yield()

    def total_yield(self):

        y_dict = {i: None for i in self.df['Q'].unique()}

        for i in self.df.index:
            if y_dict[self.df.loc[i, 'Q']] is None:
                y_dict[self.df.loc[i, 'Q']] = self.df.loc[i, 'Yield (kg)']
            else:
                y_dict[self.df.loc[i, 'Q']] = y_dict[self.df.loc[i, 'Q']] + self.df.loc[i, 'Yield (kg)']

        y_vals = list(y_dict.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_vals, linewidth=1, c='k')
        ax.set_title('Total Sediment Yield', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
        ax.set_ylabel('Yield (kg)', fontsize=14, fontweight='bold')
        plt.grid(which='major', axis='y')
        plt.text(0.7*len(y_vals), 0.7*np.max(y_vals), 'Total yield: ' + str(int(np.sum(y_vals))) + ' kg', fontsize=14)
        plt.tight_layout()

        plt.savefig(self.out_path + '/total_yield.png', dpi=150)

    def fractional_yield(self):

        df1 = self.df[self.df['D'] == 0.001]
        df8 = self.df[self.df['D'] == 0.008]
        df32 = self.df[self.df['D'] == 0.032]
        df128 = self.df[self.df['D'] == 0.128]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df1['Q'], df1['qb (kg/m/s)'], label='1-2 mm')
        ax.scatter(df8['Q'], df8['qb (kg/m/s)'], label='8-12 mm')
        ax.scatter(df32['Q'], df32['qb (kg/m/s)'], label='32-48 mm')
        ax.scatter(df128['Q'], df128['qb (kg/m/s)'], label='128-192 mm')
        ax.set_title('Fractional Transport Rates', fontsize=16, fontweight='bold')
        ax.set_xlabel(r'$Q (\frac{m^3}{s})$', fontsize=14)
        ax.set_ylabel(r'$q_b (kg m^{-1}s^{-1})$', fontsize=14)

        plt.legend(fontsize=14)
        plt.tight_layout()

        plt.savefig(self.out_path + '/fractional_transport.png', dpi=150)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('stream_id', help='The name of the stream as entered in the Input_data csv files', type=str)

    args = parser.parse_args()

    CreatePlots(args.stream_id)


if __name__ == '__main__':
    main()
