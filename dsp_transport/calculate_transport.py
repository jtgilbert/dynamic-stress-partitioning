import argparse
import numpy as np
from scipy.optimize import minimize


def err(v, d, q_obs, s, w):
    h = d ** 0.25 * ((v ** 1.5 / (9.81 * s) ** 0.75) / 22.627)
    q_pred = v * h * w
    err = (q_obs - q_pred) ** 2

    return err

def ferguson_vpe(h, d84, s):  # could maybe strip depth out of this and just give a starting estimate...

    v_est = ((6.5*(h/d84)) / (((h/d84)**(5/3)+(6.5/2.5)**2)**0.5))*(9.81*h*s)**0.5
    return v_est

def calc_hi(d, q, h, d84, s, w):
    v0 = ferguson_vpe(h, d84, s)*3

    res = minimize(err, v0, args=(d, q, s, w))
    if res.x[0] <= 0.01:
        print('Very low or negative velocity solution: setting to 0.01 m/s')
        res.x[0] = 0.01
    h_adj = d ** 0.25 * ((res.x[0] ** 1.5 / (9.81 * s) ** 0.75) / 22.627)

    return h_adj

def percentiles(fractions):
    out_sizes = []
    d_percentiles = [0.5, 0.84]
    for p in d_percentiles:
        cumulative = 0
        grain_size = None
        # while cumulative < p:
        for size, fraction in fractions.items():
            if cumulative < p:
                cumulative += fraction
                grain_size = size
        for i, size in enumerate(fractions.keys()):
            if size == grain_size:
                low_size = list(fractions.keys())[i - 1] - 0.25
                low_cum = cumulative - fractions[list(fractions.keys())[i - 1]]
                high_size = grain_size - 0.25
                high_cum = cumulative
                p_out = ((low_cum / p) * low_size + (p / high_cum) * high_size) / ((low_cum / p) + (p / high_cum))
                out_sizes.append(p_out)

    return out_sizes[0], out_sizes[1]

def transport(fractions: dict, slope:float, discharge: float, depth: float, width: float, interval: int,
              twod: bool = False, lwd_factor: int = None):
    """

    :param fractions: A python dictionary {size: fraction in bed} with all size classes and the fraction of the bed
    they make up e.g. {0.004: 0.023, 0.008: 0.041}
    :param slope: The reach slope
    :param discharge: The flow discharge in m3/s
    :param depth: The average flow depth at the given discharge
    :param width: The average flow width at the given discharge
    :param interval: The length of time in seconds of the discharge measurement
    :return: A dictionary with transport rate and total transport for each size fraction
    """

    if twod is False:
        transport_rates = {}
    else:
        transport_rates = {'bed': {}, 'wall': {}}

    # find D50 and D84 from GSD
    if twod is False:
        d_sizes = percentiles(fractions)
    else:
        d_sizes = percentiles(fractions['bed'])
    d50 = 2**(-d_sizes[0]) / 1000
    d84 = 2**(-d_sizes[1]) / 1000
    roughness = d84/d50

    if twod is False:
        fractionssub = fractions
    else:
        fractionssub = fractions['bed']

    for size_phi, frac in fractionssub.items():
        size = 2**-size_phi / 1000
        if roughness <= 2:
            tau_star_coef = 0.025
        elif 2 < roughness < 3.5:
            tau_star_coef = 0.087 * np.log(roughness) - 0.034
        else:
            tau_star_coef = 0.073

        if lwd_factor == 1:
            tau_star_coef = tau_star_coef + 0.01
        if lwd_factor == 2:
            tau_star_coef = tau_star_coef + 0.02
        if lwd_factor == 3:
            tau_star_coef = tau_star_coef + 0.03

        h_i = calc_hi(size, discharge, depth, d84, slope, width)
        tau_star_crit = tau_star_coef * (size / d50) ** -0.68
        tau_star = (9810 * h_i * slope) / (1650 * 9.81 * size)

        if twod is False:
            ratio = tau_star / tau_star_crit
            if ratio < 1.8:
                wi_star = 0.0015 * ratio ** 7.5
            else:
                wi_star = 14 * (1 - (1.0386 / ratio ** 0.9)) ** 5

            # convert from wi_star to qs
            q_b = (wi_star * (frac * 100) * (9.81 * depth * slope) ** (3 / 2)) / (1.65 * 9.81)
            Qb = q_b * width
            tot = Qb * interval

            transport_rates[size_phi] = [q_b, tot]

        else:
            wall_frac = 1.9534 * (width/depth) ** -1.12  # based on Pan et al 2020 (https://www.tandfonline.com/doi/full/10.1080/00221686.2020.1818318?casa_token=E-zYFFoAlSUAAAAA%3Am-yS_bCnb25Mey6KoglKydPB-1PuhyPdIIcafqib3Kswe09LD72p4w1k4qCOEAL7XZ2OdJlVnmM)
            tau_bed = tau_star / (wall_frac + 1)
            tau_wall = wall_frac * tau_bed

            ratio_bed = tau_bed / tau_star_crit
            if ratio_bed < 1.8:
                wi_star_bed = 0.0015 * ratio_bed ** 7.5
            else:
                wi_star_bed = 14 * (1 - (1.0386 / ratio_bed ** 0.9)) ** 5

            # convert from wi_star to qs
            q_b_bed = (wi_star_bed * (frac * 100) * (9.81 * depth * slope) ** (3 / 2)) / (1.65 * 9.81)
            Qb_bed = q_b_bed * width
            tot_bed = Qb_bed * interval

            ratio_wall = tau_wall / tau_star_crit
            if ratio_wall < 1.8:
                wi_star_wall = 0.0015 * ratio_wall ** 7.5
            else:
                wi_star_wall = 14 * (1 - (1.0386 / ratio_wall ** 0.9)) ** 5

            # convert from wi_star to qs
            q_b_wall = (wi_star_wall * (fractions['wall'][size_phi] * 100) * (9.81 * depth * slope) ** (3 / 2)) / (1.65 * 9.81)
            Qb_wall = q_b_wall * (2 * depth)
            tot_wall = Qb_wall * interval

            transport_rates['bed'].update({size_phi: [q_b_bed, tot_bed]})
            transport_rates['wall'].update({size_phi: [q_b_wall, tot_wall]})

    return transport_rates

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('fractions', help='A python dictionary {size: fraction in bed} with all size classes and the'
                                          'fraction of the bed they make up e.g. {0.004: 0.023, 0.008: 0.041}',
                        type=any)
    parser.add_argument('slope', help='The reach slope', type=float)
    parser.add_argument('discharge', help='The flow discharge in m3/s', type=float)
    parser.add_argument('depth', help='The average flow depth at the given discharge', type=float)
    parser.add_argument('width', help='The average flow width at the given discharge', type=float)
    parser.add_argument('interval', help='The length of time in seconds of the discharge measurement', type=int)

    args = parser.parse_args()

    transport(args.fractions, args.slope, args.discharge, args.depth, args.width, args.interval)


if __name__ == '__main__':
    main()
