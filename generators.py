from scipy.special import i0 as bessel_i0
import numpy as np


def get_gen_mean(t, gen_params, dt=0.01):
    pi = 3.1415
    ALPHA = 5.0
    v_an = 20

    field_center = gen_params['CenterPlaceField']
    out_rate = gen_params['OutPlaceFiringRate']
    theta_phase = gen_params['OutPlaceThetaPhase'] # np.deg2rad
    R = gen_params['R']
    kappa = r2kappa(R)
    theta_freq = gen_params['ThetaFreq']

    peak_rate = gen_params['InPlacePeakRate']
    sigma_field = gen_params['SigmaPlaceField']

    precession_onset = gen_params['PrecessionOnset']
    precession_slope = gen_params['SlopePhasePrecession']
    
    t_center = field_center#/v_an

    mult4time = 2 * pi * theta_freq * 0.001
    I0 = bessel_i0(kappa)
    normalizator = out_rate / I0 * 0.001 # units: Herz/ ms # not probability of spikes during dt


    # Если нужна прецессия:
    # start_place = t - t_center - 3 * sigma_field
    # end_place = t - t_center + 3 * sigma_field
    # inplace = 0.25 * (1.0 - (start_place / (ALPHA + np.abs(start_place)))) * (
    #         1.0 + end_place / (ALPHA + np.abs(end_place)))
    
    phase = theta_phase # * (1 - inplace) - precession_onset * inplace
    
    precession = 0 #precession_slope * t * inplace

    out_firings = normalizator * np.exp(kappa * np.cos(mult4time * t + precession - phase))
    # spatial_firings = 1 + peak_rate * np.exp(-0.5 * ((t - t_center) / sigma_field)** 2)

    firings = out_firings #* spatial_firings

    return firings

def r2kappa(R):
    kappa = np.where(R < 0.53,  2 * R + (R**3) + 5 / 6 * R**5, 0.0)
    kappa = np.where(np.logical_and(R >= 0.53, R < 0.85),  -0.4 + 1.39 * R + 0.43 / (1 - R), kappa)
    kappa = np.where(R >= 0.85,  1 / (3 * R - 4 * R**2 + R**3), kappa)
    return kappa


def generators_inputs(gen_params, t):

    # mec = np.zeros(shape=(t.size, gen_pop_size), dtype=np.float32)
    # lec = np.zeros(shape=(t.size, gen_pop_size), dtype=np.float32)

    mec_mean = np.zeros(t.size, dtype=np.float32)
    lec_mean = np.zeros(t.size, dtype=np.float32)

    for i in range(t.size):
        
        mec_mean[i] = get_gen_mean(t[i], gen_params['mec'])
        lec_mean[i] = get_gen_mean(t[i], gen_params['lec'])

        # mec[i, :] = (np.random.rand(gen_pop_size) < mec_mean[i]*dt).astype(np.float32)
        # lec[i, :] = (np.random.rand(gen_pop_size) < lec_mean[i]*dt).astype(np.float32)

    return mec_mean, lec_mean
    