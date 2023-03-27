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
        print(f'Very low or negative velocity solution for flow: {q}, size: {d}; setting to 0.01')
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
        fractionss = fractions
    else:
        fractionss = fractions['bed']

    # adjust fraction sub to area
    tmp_vals = {}
    for key, value in fractionss.items():
        tmp_vals[key] = (value * ((2**-key / 1000) / d50) ** 2)
    denom = np.sum(list(tmp_vals.values()))

    fractionssub = {}
    for key, value in fractionss.items():
        fractionssub[key] = tmp_vals[key] / denom

    updatekeys = [1,0,-1]
    update_val = fractionssub[1] + fractionssub[0] + fractionssub[-1]
    for i in updatekeys:
        fractionssub[i] = update_val

    for size_phi, frac in fractionssub.items():
        size = 2**-size_phi / 1000
        if roughness <= 2:
            tau_star_coef = 0.029
        else:
            tau_star_coef = 0.043 * np.log(roughness) - 0.0005

        h_i = calc_hi(size, discharge, depth, d84, slope, width)

        if lwd_factor is None:
            tau_star_crit = tau_star_coef * (size / d50) ** -0.67
        elif lwd_factor == 1:
            tau_star_crit = (tau_star_coef * (size / d50) ** -0.67) + 0.01
        elif lwd_factor == 2:
            tau_star_crit = (tau_star_coef * (size / d50) ** -0.67) + 0.02
        elif lwd_factor == 3:
            tau_star_crit = (tau_star_coef * (size / d50) ** -0.67) + 0.03

        tau_star = (9810 * h_i * slope) / (1650 * 9.81 * size)

        if twod is False:
            ratio = tau_star / tau_star_crit
            min_ratio = (0.02 * (size / d50) ** -0.67) / tau_star_crit
            if ratio < min_ratio:
                ratio = 0
            if ratio < 2:
                wi_star = 0.0002 * ratio ** 13
            else:
                wi_star = 100 * (1 - (1.348 / ratio ** 2)) ** 10

            # convert from wi_star to qs
            q_b_vol = (wi_star * frac * (9.81 * depth * slope) ** (3 / 2)) / (1.65 * 9.81)
            q_b_mass = q_b_vol * 2650
            Qb = q_b_mass * width
            tot = Qb * interval

            transport_rates[size_phi] = [q_b_mass, tot]

        else:
            wall_frac = 1.9534 * (width/depth) ** -1.12  # based on Pan et al 2020 (https://www.tandfonline.com/doi/full/10.1080/00221686.2020.1818318?casa_token=E-zYFFoAlSUAAAAA%3Am-yS_bCnb25Mey6KoglKydPB-1PuhyPdIIcafqib3Kswe09LD72p4w1k4qCOEAL7XZ2OdJlVnmM)
            tau_bed = tau_star / (wall_frac + 1)
            tau_wall = wall_frac * tau_bed

            ratio_bed = tau_bed / tau_star_crit
            min_ratio_bed = (0.02 * (size / d50) ** -0.67) / tau_star_crit
            if ratio_bed < min_ratio_bed:
                ratio = 0
            if ratio_bed < 2:
                wi_star_bed = 0.0002 * ratio_bed ** 13
            else:
                wi_star_bed = 100 * (1 - (1.348 / ratio_bed ** 2)) ** 10

            # convert from wi_star to qs
            q_b_bed_vol = (wi_star_bed * frac * (9.81 * depth * slope) ** (3 / 2)) / (1.65 * 9.81)
            q_b_bed_mass = q_b_bed_vol * 2650
            Qb_bed = q_b_bed_mass * width
            tot_bed = Qb_bed * interval

            ratio_wall = tau_wall / tau_star_crit
            # if ratio_wall < 1.8:
            wi_star_wall = 0.00005 * ratio_wall ** 6
            # else:
            #     wi_star_wall = 14 * (1 - (1.0386 / ratio_wall ** 0.9)) ** 5

            # convert from wi_star to qs
            q_b_wall_vol = (wi_star_wall * fractions['wall'][size_phi] * (9.81 * depth * slope) ** (3 / 2)) / (1.65 * 9.81)
            q_b_wall_mass = q_b_wall_vol * 2650
            Qb_wall = q_b_wall_mass * (2 * depth)
            tot_wall = Qb_wall * interval

            transport_rates['bed'].update({size_phi: [q_b_bed_mass, tot_bed]})
            transport_rates['wall'].update({size_phi: [q_b_wall_mass, tot_wall]})

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

# fracs = {'bed': {1: 0.01, 0: 0.01, -1: 0.01, -2: 0.01, -3: 0.02, -3.5: 0.02, -4: 0.04, -4.5: 0.05, -5: 0.1, -5.5: 0.12,
#          -6: 0.15, -6.5: 0.18, -7: 0.08, -7.5: 0.05, -8: 0.02},
#          'wall': {1: 0.01, 0: 0.01, -1: 0.01, -2: 0.01, -3: 0.02, -3.5: 0.02, -4: 0.04, -4.5: 0.05, -5: 0.1, -5.5: 0.12,
#          -6: 0.15, -6.5: 0.18, -7: 0.08, -7.5: 0.05, -8: 0.02}}
#
# tr = transport(fracs, 0.013, 7, 0.42, 13.6, 900, twod=True)
# print(tr)