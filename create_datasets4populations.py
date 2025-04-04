import pandas as pd
import numpy as np
from scipy.signal.windows import parzen
from brian2 import NeuronGroup, Network, SpikeMonitor, StateMonitor
from brian2 import ms, mV, nF, uF, mS, uS, uA, pA, second, Hz, nA, nS, pF
from brian2 import defaultclock
import h5py
import os
from pprint import pprint
import myconfig
import matplotlib.pyplot as plt
from multiprocessing import Pool

PATH4SAVING = myconfig.DATASETS4POPULATIONMODELS


def randinterval(minv, maxv):
    v = np.random.uniform(minv, maxv)
    return v


def add_units(value, key):
    if key == "Cm":
        return value * pF
    if key == "k":
        return value * nS / mV
    if key == "Vrest":
        return value * mV
    if key == "Vth":
        return value * mV
    if key == "Vpeak":
        return value * mV
    if key == "Vmin":
        return value * mV
    if key == "a":
        return value * ms ** -1
    if key == "b":
        return value * nS
    if key == "d":
        return value * pA
    return value


def check_gparams(params, duration):

    t = np.arange(0, duration,  myconfig.DT) * 0.001

    g_exc = 0
    g_inh = 0
    for idx in range(1, 5):
        ge = params[f"ampl_{idx}_e"]/mS * 0.5 * ( np.cos(2*np.pi*t*params[f"omega_{idx}_e"]/Hz + params[f"phase0_{idx}_e"] ) + 1)
        gi = params[f"ampl_{idx}_i"]/mS * 0.5 * ( np.cos(2*np.pi*t*params[f"omega_{idx}_i"]/Hz + params[f"phase0_{idx}_i"] ) + 1)

        g_exc += ge
        g_inh += gi

    Erev = (params["Eexc"] / mV * g_exc + params["Einh"] / mV * g_inh) / (g_exc + g_inh)
    tau_syn = float(params['Cm'] / uF) / (g_exc + g_inh + 0.0001)

    #isupthresh = Erev > params["Vth_mean"] #/ mV)
    isupthresh = tau_syn < 10 #/ mV)

    # level = float(params["Vth_mean"])
    # print(np.sum(isupthresh))
    # plt.plot(t, Erev)
    # plt.hlines(level , 0, t[-1])
    # plt.show()
    return np.mean(isupthresh)



def run_izhikevich_neurons(params, duration, NN, filepath):

    defaultclock.dt = myconfig.DT * ms
    tau_min = 1.5 # ms
    tau_max = 20.0 # ms

    ampl_max_exc = 0.1 * float(params['Cm']/pF) / tau_min
    ampl_min_exc = 0.1 * float(params['Cm']/pF) / tau_max

    ampl_min_inh = 2.5 * ampl_min_exc
    ampl_max_inh = 2.5 * ampl_max_exc

    g_params = {
        "omega_1_e": randinterval(0.2, 2.0) * Hz,  # [0.2 2],
        "omega_2_e": randinterval(4.0, 12.0) * Hz,  # [4  12],
        "omega_3_e": randinterval(25.0, 45.0) * Hz,  # [25  45],
        "omega_4_e": randinterval(50.0, 90.0) * Hz,  # [50  90],

        "ampl_1_e": randinterval(ampl_min_exc, ampl_max_exc) * nS,  # [0.2 10],
        "ampl_2_e": randinterval(ampl_min_exc, ampl_max_exc) * nS,  # [0.2 10],
        "ampl_3_e": randinterval(ampl_min_exc, ampl_max_exc) * nS,  # [0.2 10],
        "ampl_4_e": randinterval(ampl_min_exc, ampl_max_exc) * nS,  # [0.2 10],

        "phase0_1_e": randinterval(-np.pi, np.pi),  # [-pi pi],
        "phase0_2_e": randinterval(-np.pi, np.pi),  # [-pi pi],
        "phase0_3_e": randinterval(-np.pi, np.pi),  # [-pi pi],
        "phase0_4_e": randinterval(-np.pi, np.pi),  # [-pi pi],

        "omega_1_i": randinterval(0.2, 2.0) * Hz,  # [0.2 2],
        "omega_2_i": randinterval(4.0, 12.0) * Hz,  # [4  12],
        "omega_3_i": randinterval(25.0, 45.0) * Hz,  # [25  45],
        "omega_4_i": randinterval(50.0, 90.0) * Hz,  # [50  90],

        "ampl_1_i": randinterval(ampl_min_inh, ampl_max_inh) * nS,  # [0.2 50],
        "ampl_2_i": randinterval(ampl_min_inh, ampl_max_inh) * nS,  # [0.2 50],
        "ampl_3_i": randinterval(ampl_min_inh, ampl_max_inh) * nS,  # [0.2 50],
        "ampl_4_i": randinterval(ampl_min_inh, ampl_max_inh) * nS,  # [0.2 50]

        "phase0_1_i": randinterval(-np.pi, np.pi),  # [-pi pi],
        "phase0_2_i": randinterval(-np.pi, np.pi),  # [-pi pi],
        "phase0_3_i": randinterval(-np.pi, np.pi),  # [-pi pi],
        "phase0_4_i": randinterval(-np.pi, np.pi),  # [-pi pi],
    }



    ################
    eqs = '''
    dV/dt = (k*(V - Vrest)*(V - Vth) - U + Iexc + Iinh)/Cm + sigma*xi/ms**0.5 : volt
    dU/dt = a * (b * (V - Vrest) - U) : ampere
    Iexc = gexc*(Eexc - V)            : ampere
    Iinh = ginh*(Einh - V)            : ampere
    gexc = ampl_1_e*0.5*(cos(2*pi*t*omega_1_e + phase0_1_e) + 1 ) + ampl_2_e*0.5*(cos(2*pi*t*omega_2_e + phase0_2_e) + 1 ) + ampl_3_e*0.5*(cos(2*pi*t*omega_3_e + phase0_3_e) + 1 ) + ampl_4_e*0.5*(cos(2*pi*t*omega_4_e + phase0_4_e) + 1 ) : siemens
    ginh = ampl_1_i*0.5*(cos(2*pi*t*omega_1_i + phase0_1_i) + 1 ) + ampl_2_i*0.5*(cos(2*pi*t*omega_2_i + phase0_2_i) + 1 ) + ampl_3_i*0.5*(cos(2*pi*t*omega_3_i + phase0_3_i) + 1 ) + ampl_4_i*0.5*(cos(2*pi*t*omega_4_i + phase0_4_i) + 1 ) : siemens
    Vth : volt
    '''

    params = params | g_params
    # res = check_gparams(params, duration)
    # print(res)
    #return True
    #NN = 1
    ## params["Vrest"], params["Vmin"] = params["Vmin"], params["Vrest"]
    ## params["a"] = params["a"] * 0.01

    #print(params["b"] * params["Vrest"])


    neuron = NeuronGroup(NN, eqs, method='heun', threshold='V > Vpeak', reset="V = Vmin; U = U + d", namespace=params)
    #
    neuron.V = params["Vrest"]
    neuron.U = 0 * pA
    neuron.Vth = np.random.normal(loc=params['Vth_mean'], scale=4.0, size=NN) * mV # params['Vth_mean']*mV  #

    if myconfig.IS_SAVE_V:
        M_full_V = StateMonitor(neuron, ['V', 'U'], record=np.arange(NN))


    # M_full_U = StateMonitor(neuron, 'U', record=np.arange(N))
    gexc_monitor = StateMonitor(neuron, 'gexc', record=0)
    ginh_monitor = StateMonitor(neuron, 'ginh', record=0)

    firing_monitor = SpikeMonitor(neuron)

    if myconfig.IS_SAVE_V:
        monitors = [M_full_V, gexc_monitor, ginh_monitor, firing_monitor]
    else:
        monitors = [gexc_monitor, ginh_monitor, firing_monitor]
    #monitors = [M_full_V, firing_monitor]

    net = Network(neuron)  # automatically include G and S
    net.add(monitors)  # manually add the monitors

    net.run(duration * ms) #, report='text'

    #Varr = np.asarray(M_full_V.V / mV)

    population_firing_rate, bins = np.histogram(firing_monitor.t / ms, range=[0, duration], bins=int(duration/(defaultclock.dt / ms) ))
    dbins = bins[1] - bins[0]
    population_firing_rate = population_firing_rate / NN  / (0.001 * dbins) # spikes per second




    Nspikes = np.asarray(firing_monitor.t).size
    mean_firing_rate = Nspikes/NN/(duration * 0.001)
    if  mean_firing_rate < 0.2:
        print("A lot of spikes!!!! Do not save simulation!!!!!")
        return False


    # fig, axes = plt.subplots(nrows=2)
    # axes[0].plot(M_full_V.V[0] / mV)
    # axes[1].plot(M_full_V.U[0] / pA)
    # axes[0].set_title(mean_firing_rate)
    # plt.show()
    # return True

    # ###### smoothing of population firing rate #########
    win = parzen(101)
    win = win / np.sum(win)
    population_firing_rate = np.convolve(population_firing_rate, win, mode='same')

    gexc = np.asarray(gexc_monitor.gexc / nS).astype(np.float32).ravel()
    ginh = np.asarray(ginh_monitor.ginh / nS).astype(np.float32).ravel()
    grest = myconfig.GREST

    Erev =  (params["Eexc"]/mV * gexc + params["Einh"]/mV * ginh  + params["Vrest"]/mV * grest ) / (gexc + ginh + grest)
    tau_syn = float(params['Cm']/pF) / (gexc + ginh + grest)

    file = h5py.File(filepath, mode='w')
    file.create_dataset('firing_i', data=np.asarray(firing_monitor.i).astype(np.float32).ravel() )
    file.create_dataset('firing_t', data=np.asarray(firing_monitor.t / ms).astype(np.float32).ravel() )
    file.create_dataset('firing_rate', data=population_firing_rate.astype(np.float32).ravel() )
    file.create_dataset('Erevsyn', data=Erev.ravel())
    file.create_dataset('tau_syn', data=tau_syn.ravel())
    file.attrs['dt'] = dbins

    file.create_dataset('gexc', data=gexc)
    file.create_dataset('ginh', data=ginh)

    if myconfig.IS_SAVE_V:
        file.create_dataset('V', data = np.asarray(M_full_V.V / mV))
        file.create_dataset('U', data=np.asarray(M_full_V.U / pA))

    file.close()

    return True


def create_single_type_dataset(params, path, Niter=100, duration=2000, NN=4000):


    idx = 0
    while (idx < Niter):
        filepath = '{path}/{i}.hdf5'.format(path=path, i=idx)
        if os.path.isfile(filepath):
            idx += 1
            continue

        res = run_izhikevich_neurons(params, duration, NN, filepath)
        if res:
            idx += 1

def parrallel_runner(run_params):
    for neuron_type_params in run_params:
        params, path, Niter, duration, NN = neuron_type_params
        create_single_type_dataset(params, path, Niter=Niter, duration=duration, NN=NN)

def create_all_types_dataset(all_params, NN):

    running_params = []
    for i in range(myconfig.N_THREDS ):
        running_params.append([])

    for n, (key, item) in enumerate(all_params.items()):
        path = '{path}{key}'.format(path=PATH4SAVING, key=key)

        if not os.path.isdir(path):
           os.mkdir(path)

        running_params[ int(n%myconfig.N_THREDS)].append([item, path, myconfig.NFILESDATASETS, myconfig.DURATION, NN])
        #create_single_type_dataset(item, path, Niter=myconfig.NFILESDATASETS, NN=NN)

    #pprint(running_params)
    with Pool(myconfig.N_THREDS) as parallel:
         parallel.map(parrallel_runner, running_params)


def main():
    NN = myconfig.NUMBERNEURONSINPOP
    default_params = {
        "Cm": 114, # * pF,
        "k": 1.19, # * nS / mV,
        "Vrest": -57.63, # * mV,
        "Vth_mean": 35.53, #*mV,
        "Vpeak": 21.72, # * mV,
        "Vmin": -48.7, # * mV,
        "a": 0.005, # * ms ** -1,
        "b": 0.22, # * nS,
        "d": 2, # * pA,

        "Eexc": 0 * mV,
        "Einh": -75 * mV,

        "sigma": 0.4 * mV,

    }
    filepath = myconfig.IZHIKEVICNNEURONSPARAMS
    syndata = pd.read_csv(filepath)
    syndata = syndata.fillna(-1)
    syndata = syndata.rename(columns={'Izh Vr': 'Vrest', 'Izh Vt': 'Vth_mean', 'Izh C': 'Cm', 'Izh k': 'k', 'Izh a': 'a', 'Izh b': 'b', 'Izh d': 'd', 'Izh Vpeak': 'Vpeak', 'Izh Vmin': 'Vmin',})
    syndata = syndata.drop(['CARLsim_default', 'E/I', 'Population Size', 'Refractory Period', 'rank'], axis=1)

    ###syndata["k"] *= 10

    neurons_types = pd.read_excel(myconfig.FIRINGSNEURONPARAMS, sheet_name="Sheet2", header=0)
    simutated_population_types = neurons_types[neurons_types["is_include"] == 1]["neurons"].to_list()


    all_params = {}
    for population_idx, populations_params in syndata.iterrows():
        populations_params = populations_params.to_dict()

        if not populations_params["Neuron Type"] in simutated_population_types:
            continue

        # if not populations_params["Neuron Type"] in ["CA1 Horizontal Axo-Axonic", ]: ######!!!!!!!!!!!!!!!!!!!!!
        #      continue


        neuron_opt_params = default_params.copy()
        neuron_type = populations_params["Neuron Type"]


        del populations_params["Neuron Type"]

        #pprint.pprint(populations_params)
        for key, val in populations_params.items():
             neuron_opt_params[key] = add_units(val, key)
        all_params[neuron_type] = neuron_opt_params

    create_all_types_dataset(all_params, NN)


if __name__ == '__main__':
    main()
