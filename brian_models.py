from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from brian2 import NeuronGroup, Synapses, SpikeMonitor, StateMonitor
from brian2 import ms, mV
from brian2 import defaultclock, run
import csv
import pandas as pd
from scipy.special import i0 as bessel
import sys
import pickle
from pprint import pprint
sys.setrecursionlimit(2000)


# import warnings
# warnings.filterwarnings("ignore")
# BrianLogger.initialize()
# BrianLogger.suppress_hierarchy('brian2', filter_log_file=False)
# logging.console_log_level = 'FALSE'
# logging.file_log_level = 'FALSE'
# logging.file_log = False


THETA_FREQ = 8
V_AN = 20
duration = 5/10 * second
tfinal = 5000/10 * ms
defaultclock.dt = 0.02 * ms


# import tables with parameters of neurons and synapses types
neuron_types = pd.read_csv('parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv', delimiter=',')
synapse_types = pd.read_csv('parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv')

# Load neurons and synapses lists with indexes
with (open("presimulation_files/neurons.pickle", "rb")) as openfile:
    while True:
        try:
            neurons_list0 = pickle.load(openfile)
        except EOFError:
            break
with (open("presimulation_files/connections.pickle", "rb")) as openfile:
    while True:
        try:
            synapses_list0 = pickle.load(openfile)
        except EOFError:
            break

# Just a set of all given types:
Types = set()
for neuron_ in neurons_list0:
    key = neuron_['type']
    Types.add(key)
Types = list(Types)

print(Types)


# But for testing, create another lists of neurons of random types and with random connections:

Nneur = 100
Nsyn = 500 # 0.05 of all possible

neurons_list = [] 
for n in range(Nneur):
    i = randint(len(Types))
    _type = Types[i]
    neurons_list.append({'type': _type})

#neurons_list = [{'type': 'CA1 Pyramidal'}, {'type': 'CA1 Pyramidal'}, {'type': 'CA1 Pyramidal'}, {'type': 'CA1 Axo-Axonic'}, {'type': 'CA1 Axo-Axonic'}, {'type': 'CA1 Axo-Axonic'}, {'type': 'CA1 Axo-Axonic'}, {'type': 'CA1 Axo-Axonic'}, {'type': 'CA1 O-LM'}, {'type': 'CA1 O-LM'}]

# Dictionary where key is a type and value is a list of indexes of neurons of this type in neuron_list
types_indicies = {}
for idx, neuron in enumerate(neurons_list):
    type = neuron['type']
    if type not in types_indicies.keys():
        types_indicies[type] = [idx]
    else: types_indicies[type].append(idx)

#pprint(types_indicies)


synapses_list = []
for n in range(Nsyn):
    idx1 = randint(0, Nneur) # choosing random indexes of neurons to connect
    idx2 = randint(0, Nneur)
    synapses_list.append({'post_idx': idx2, 'pre_idx': idx1, 'post_type': neurons_list[idx2]['type'], 'pre_type': neurons_list[idx1]['type']})
# print(synapses)

syn_groups = {}
for type1 in types_indicies.keys():
    for type2 in types_indicies.keys():
        syn_groups[type1+'_to_'+type2] = {'pre_indexes':[], 'post_indexes':[]}


for syn in synapses_list:
    type1 = syn['pre_type']
    type2 = syn['post_type']

    new_pre_index = types_indicies[type1].index(syn['pre_idx'])
    new_post_index = types_indicies[type2].index(syn['post_idx'])

    syn_groups[type1+'_to_'+type2]['pre_indexes'].append(new_pre_index)
    syn_groups[type1+'_to_'+type2]['post_indexes'].append(new_post_index)
    # syn['pconn']

# print(syn_groups)

eqs = """dv/dt = ((k*(v - vr)*(v - vt) - u_p + I + I_noise + 10*I_syn)/Cm)/ms : 1
         du_p/dt = (a*(b*(v-vr) - u_p))/ms  : 1
         I : 1
         I_noise : 1
         I_syn : 1 
         a : 1
         b : 1
         d : 1
         Cm : 1
         k : 1
         vt : 1
         vr : 1
         Vpeak : 1
         Vmin : 1
         refr : second
       """

# returns NeuronGroup object of needed type and population size
def get_neuron_group(type, pop_size):

    eqs = """dv/dt = ((k*(v - vr)*(v - vt) - u_p + I + I_noise + 10*Isyn)/Cm)/ms : 1
             du_p/dt = (a*(b*(v-vr) - u_p))/ms  : 1
          """
    l = len(type) + 2
    # Choose only those synapses, where post_neuron is of needed type
    new_syn_names = ['Isyn_' + s.replace(' ', '_').replace('-', '_').replace('+', '') for s in syn_groups.keys() if type in s[-l:]]
    Isyn_sum = "Isyn = " + " + ".join(new_syn_names) + " : 1"
    eqs += Isyn_sum
    eqs += """
         I : 1
         I_noise : 1
         a : 1
         b : 1
         d : 1
         Cm : 1
         k : 1
         vt : 1
         vr : 1
         Vpeak : 1
         Vmin : 1
         """
    
    for syn_name in new_syn_names:
        eqs += f"""{syn_name} : 1
         """ 
    
    # print(eqs)

    # Choose params of necessary type
    neuron_param = neuron_types[neuron_types['Neuron Type'] == type]
    
    refr = neuron_param['Refractory Period'].iloc[0]
    Group = NeuronGroup(pop_size, eqs, threshold="v>=Vpeak", reset="v=Vmin; u_p+=d", refractory= refr*ms, method="euler") #neuron['Population Size']
    Group.a = neuron_param['Izh a'].iloc[0]
    Group.b = neuron_param['Izh b'].iloc[0]
    Group.Cm = neuron_param['Izh C'].iloc[0]
    Group.d = neuron_param['Izh d'].iloc[0]
    Group.vr = neuron_param['Izh Vr'].iloc[0]
    Group.vt = neuron_param['Izh Vt'].iloc[0]
    Group.k = neuron_param['Izh k'].iloc[0]
    Group.I = 0
    Group.I_noise = 0
    Group.Vpeak = neuron_param['Izh Vpeak'].iloc[0]
    Group.Vmin = neuron_param['Izh Vmin'].iloc[0]
    return Group


print('Successfuly created neuron models for all types')

# Check models:

# C = get_neuron_group('CA1 Pyramidal', 10)
# print(C.a, C.b, C.d, C.Cm, C.vr, C.vt, C.k, C.u_p)
# M = StateMonitor(C, 'v', record=True)
# run(tfinal)
# plot(M.t/ms, M.v[0])
# plt.show()

def r2kappa(R):
    """
    recalulate kappa from R for von Misses function
    """
    if R < 0.53:
        kappa = 2 * R + R**3 + 5/6 * R**5

    elif R >= 0.53 and R < 0.85:
        kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)

    elif R >= 0.85:
        kappa = 1 / (3*R - 4*R**2 + R**3)
        
    I0 = bessel(kappa)

    return kappa, I0


ca1params = {
            "name": "ca1pyr",
            "R": 0.2,
            "freq": THETA_FREQ,
            "mean_spike_rate": 0.5,#*20, #######
            "phase": 3.14,
        }

ec3params = {
            "name": "ec3",
            "R": 0.3,
            "freq": THETA_FREQ,
            "mean_spike_rate": 1.5,
            "phase": -1.57,

            "sigma_sp": 5.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 8.0,  # spike sec in the center of field
            "sp_centers": -5.0,  # cm
        }

ca3params = {
            "name": "ca3pyr",
            "R": 0.3,
            "freq": THETA_FREQ,
            "mean_spike_rate": 0.5,#*20,##########
            "phase": 1.58,

            "sigma_sp": 8.0,  # cm
            "v_an": V_AN,  # cm/sec
            "maxFiring": 8.0,  # spike sec in the center of field
            "sp_centers": 5.0}


ca3_center = ca3params['sp_centers']/ V_AN + duration/second * 1000 / 2
ca3params['t_center'] = ca3_center
ca3params['sigma_sp'] *= V_AN

ec3_center = ec3params['sp_centers']/ V_AN + duration/second * 1000 / 2
ec3params['t_center'] = ec3_center
ec3params['sigma_sp'] *= V_AN

rates_template_mec = '{fr} * Hz * exp({kappa} * cos(2*pi*{freq}*Hz*t - {phase}) )'
rates_template_ca3 = '{fr} * Hz * exp({kappa} * cos(2*pi*{freq}*Hz*t - {phase}) ) * (1+ {maxFiring} * exp(-0.5 * ((t/ms - {t_center}) / {sigma_sp}) ** 2))'
rates_template_lec = '{fr} * Hz * exp({kappa} * cos(2*pi*{freq}*Hz*t - {phase}) ) * (1+ {maxFiring} * exp(-0.5 * ((t/ms - {t_center}) / {sigma_sp}) ** 2))'

gens_params = [ca1params, ca3params, ec3params]

for gen_param in gens_params: 
    kappa, IO = r2kappa(gen_param["R"])
    gen_param["kappa"] = kappa
    gen_param["fr"] = gen_param["mean_spike_rate"] / IO


ca3rates = rates_template_ca3.format(**ca3params) # ca1
mec_rates = rates_template_mec.format(**ca1params)
lec_rates = rates_template_lec.format(**ec3params)
N = 1000 # 1000*6

ca3 = PoissonGroup(N, rates=ca3rates)
ca3_sm = SpikeMonitor(ca3)

mec = PoissonGroup(N, rates=mec_rates)
ca1_sm = SpikeMonitor(mec)

lec = PoissonGroup(N, rates=lec_rates)
ec3_sm = SpikeMonitor(lec)

generators = [ca3, mec, lec]

print('Succesfuly created generator models')

# run(duration)

# mfr = np.asarray(ca3_sm.t/ms).size / N / (duration/second)
# firing_rate, bins = np.histogram(ca3_sm.t/ms, bins=int(duration/ms)+1, range=[0, int(duration/ms)+1])
# dbins = 0.001*(bins[1] - bins[0])
# firing_rate = firing_rate / N / dbins

# t = np.linspace(0, duration/second, 1000)
# sine = np.max(firing_rate) * 0.5 * (np.cos(2*np.pi*ca3params["freq"]*t)+1)
# fig, axes = plt.subplots(nrows=2)
# axes[0].plot(t, sine)
# axes[0].plot(0.001*bins[:-1], firing_rate)
# axes[1].scatter(ca3_sm.t/ms, ca3_sm.i, s=2)

# plt.show()



def get_synapses(pre, post, tau_inact, tau_rec, tau_facil, A_SE, U_SE, t_delay, conn_prob, indexes, syn_name):
    """
    pre -- input stimulus
    post -- target neuron
    tau_inact -- inactivation time constant
    A_SE -- absolute synaptic strength
    U_SE -- utilization of synaptic efficacy
    tau_rec -- recovery time constant
    tau_facil -- facilitation time constant (optional)
    """

    synapses_eqs = f"""
    dx/dt =  z/tau_rec   : 1 (clock-driven) # recovered
    dy/dt = -y/tau_inact : 1 (clock-driven) # active
    A_SE : 1
    U_SE : 1
    tau_inact : second
    tau_rec : second
    z = 1 - x - y : 1 # inactive
    {syn_name}_post = A_SE*y : 1 (summed)
    du/dt = -u/tau_facil : 1 (clock-driven)
    tau_facil : second
    """
    
    synapses_action = """
    u += U_SE*(1-u)
    y += u*x # important: update y first
    x += -u*x
    """

    synapses = Synapses(pre,
                        post,
                        model=synapses_eqs,
                        on_pre=synapses_action,
                        method="exponential_euler")
    
    #print('pre: ', indexes['pre_indexes'], +'\n' + 'post: ', indexes['post_indexes'])
    synapses.connect(i = indexes['pre_indexes'], j = indexes['post_indexes']) #conn_prob

    synapses.x = 1
    synapses.tau_inact = tau_inact*ms
    synapses.A_SE = A_SE*1#00
    synapses.U_SE = U_SE*1#00
    synapses.tau_rec = tau_rec*ms
    t_delay = t_delay*ms
    synapses.delay = 't_delay'
    synapses.tau_facil = tau_facil*ms

    return synapses

#'''

ready_neurons = []
neurons = {}

for _type, _indicies in types_indicies.items():
    G = get_neuron_group(_type, len(_indicies))
    ready_neurons.append(G)
    neurons[_type] = G

net = Network(collect())

net.add(ready_neurons)


#'''

ready_synapses = []

for type1 in types_indicies.keys():
    for type2 in types_indicies.keys():
        pre_group = neurons[type1]
        post_group = neurons[type2]
        values = [0, 1]
        # pconn = syn_groups[type1+'_to_'+type2]['pconn']
        data = 1 # random.choices(values, weights=[1 - pconn, pconn])
        if data == 1:

            pre_group = neurons[type1]
            post_group = neurons[type2]
            indexes = syn_groups[type1+'_to_'+type2]
            if (len(indexes['pre_indexes']) == 0) or (len(indexes['post_indexes']) == 0): continue

            # Choosing right type of synapse from synapse_types table 
            synapse_type = synapse_types[(synapse_types['Presynaptic Neuron Type'] == type1) & (synapse_types['Postsynaptic Neuron Type'] == type2)]

            # Gettting parameters of synapse and connecting via function get_synapses
            if not synapse_type.empty:

                tau_d, tau_r, tau_f = synapse_type['tau_d'].iloc[0], synapse_type['tau_r'].iloc[0], synapse_type['tau_f'].iloc[0]
                A_SE, U_SE = synapse_type['g'].iloc[0], synapse_type['u'].iloc[0]
                t_delay = synapse_type['Synaptic Delay'].iloc[0]
                conn_prob = synapse_type['Connection Probability'].iloc[0]
                syn_name = 'Isyn_'+type1+'_to_'+type2
                syn_name = syn_name.replace(' ', '_').replace('-', '_').replace('+', '')

                S = get_synapses(pre_group, post_group, tau_d, tau_r, tau_f, A_SE, U_SE, t_delay, 1, indexes, syn_name) # 1 -> conn_prob
                
                ready_synapses.append(S)
                
                print(f'Pre neuron of type {type1} and post neuron of type {type2} are connected...')
            

net.add(ready_synapses)


from neuropynamics.src.utils.plotting import (plot_cmesh, plot_signals,
                                              plot_spikes, plot_synapses)
import networkx as nx

colors = ['cyan', 'darkblue', 'darkcyan', 'deeppink',  'darkturquoise', 'deepskyblue', 'darkgreen', 'darkred', "navy", "khaki", "darksalmon", 'springgreen', 'tomato', 'wheat']


plot_synapses(ready_neurons,
              ready_synapses,
              pos_func=nx.kamada_kawai_layout,
              color_cycle=colors,
              legend="best", node_size=100)


for neuron in neurons:
    if neuron == 'CA1 Pyramidal': 
        G = neurons['CA1 Pyramidal']
        M = StateMonitor(G, 'v', record=True)
        break


M = StateMonitor(G, 'v', record=True) 
net.add(M)
net.run(tfinal)
plot(M.t/ms, M.v[0])
plt.show()


