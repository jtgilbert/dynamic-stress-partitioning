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

def transport(fractions: dict, slope:float, discharge: float, depth: float, width: float, interval: int):

    transport_rates = {}

    # find D50 and D84 from GSD
    d50 = 136
    d84 = 260
    roughness = d50/d84

    for size, frac in fractions.items():
        if roughness <= 2:
            tau_star_coef = 0.025
        elif 2 < roughness < 3.5:
            tau_star_coef = 0.087 * np.log(roughness) - 0.034
        else:
            tau_star_coef = 0.073

        h_i = calc_hi(size, discharge, depth, d84, slope, width)

        tau_star_crit = tau_star_coef * (size/d50) ** -0.68
        tau_star = (9810 * h_i * slope) / (1650 * 9.81 * size)

        ratio = tau_star / tau_star_crit
        if ratio < 1.8:
            wi_star = 0.0015 * ratio ** 7.5
        else:
            wi_star = 14 * (1 - (1.0386 / ratio ** 0.9)) ** 5

        # convert from wi_star to qs
        q_b = (wi_star * (frac * 100) * (9.81 * depth * slope) ** (3 / 2)) / (1.65 * 9.81)
        Qb = q_b * width
        tot = Qb * interval

        transport_rates[size] = tot

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
