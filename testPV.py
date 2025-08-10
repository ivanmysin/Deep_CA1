import numpy as np
import matplotlib.pyplot as plt
import izhs_lib
import pickle
import pandas as pd
import h5py
import sys
sys.path.append('../')
import myconfig
from scipy.special import i0 as bessel_i0
from scipy.stats import cauchy
import time
from progress.bar import IncrementalBar
from pprint import pprint
PI = 3.14151728
MODE = 'honest'# #'mean' # 
if MODE == 'honest': FILE_NAME = 'results_default.h5'
else: FILE_NAME = 'results_mean.h5'
from scipy.ndimage import gaussian_filter1d


from np_meanfield import MeanFieldNetwork, run_mean_field
from generators import *




class HonestNetwork:

    def __init__(self, params, dt_dim=0.01, use_input=False, **kwargs):

        self.dt = dt_dim
        self.use_input = use_input

        self.alpha = np.asarray(params['alpha'], dtype=myconfig.DTYPE )[:,np.newaxis]

        self.units = len(params['Izh Vr'])
        self.a = np.asarray( params['Izh a'], dtype=myconfig.DTYPE )
        self.b = np.asarray( params['Izh b'], dtype=myconfig.DTYPE )
        self.w_jump = np.asarray( params['Izh d'], dtype=myconfig.DTYPE )
        # self.w_jump = np.where(self.w_jump < 0, 0, self.w_jump)

        self.Iext = np.asarray( params['Iext'], dtype=myconfig.DTYPE )
        
        self.v_tr = np.asarray( params['Izh Vt'], dtype=myconfig.DTYPE )
        self.v_rest =  np.asarray( params['Izh Vr'], dtype=myconfig.DTYPE )
        self.v_peak =  np.zeros_like(self.a, dtype=myconfig.DTYPE) + 400 #   np.asarray( params['Izh Vpeak'], dtype=myconfig.DTYPE )
        self.v_reset = np.zeros_like(self.a, dtype=myconfig.DTYPE) -400 #np.asarray( params['Izh Vr'], dtype=myconfig.DTYPE ) #  #

        print(self.v_peak)
        print(self.v_reset)
        self.gsyn_max = np.asarray(params['g'], dtype=myconfig.DTYPE)
        self.tau_f = np.asarray(params['tau_f'], dtype=myconfig.DTYPE)
        self.tau_d = np.asarray(params['tau_d'], dtype=myconfig.DTYPE)
        self.tau_r = np.asarray(params['tau_r'], dtype=myconfig.DTYPE)
        self.Uinc = np.asarray(params['u'], dtype=myconfig.DTYPE)
        self.e_r = np.asarray(params['e_r'], dtype=myconfig.DTYPE)
        

        # Применяем матрицу плотности связей
        self.pconn = np.asarray(params['pconn'])
        if MODE == 'honest':
            self.gsyn_max = np.where(np.random.uniform(0, 1, (NN+Ninps, NN, pop_size)) < self.pconn, self.gsyn_max, 0)

        self.aI = params['Izh k'] # np.ones((NN, pop_size)) #  # # , dtype=myconfig.DTYPE) # 
        self.bI = params['Izh k']*(-params['Izh Vr'] - params['Izh Vt']) # -self.alpha #  # 
        self.cI = params['Izh k']*params['Izh Vr']*(-np.abs(params['Izh Vt'])) # 0
        self.Cm = params['Izh C']  #

        print(self.aI)
        print(self.bI)
        print(self.cI)
        print(self.Cm)

        # Задаем внешние токи из Коши
        self.Delta_eta = params['Delta_eta']
        self.Icauchy = np.zeros((NN, pop_size))

        for i in range(NN):
            self.Icauchy[i] = cauchy.rvs(size=(pop_size), scale=self.Delta_eta[i]) 

        
        # self.gsyn_max *= self.aI*np.abs(self.v_rest) !!!

        # print(self.gsyn_max[:,:,0])
        self.exp_tau_d = np.exp(-self.dt / self.tau_d)
        self.exp_tau_f = np.exp(-self.dt / self.tau_f)
        self.exp_tau_r = np.exp(-self.dt / self.tau_r)
        self.tau1r = np.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)

    
        synaptic_matrix_shapes = self.gsyn_max.shape
        self.state_size = [self.units, self.units, self.units, synaptic_matrix_shapes, synaptic_matrix_shapes, synaptic_matrix_shapes]

    def get_initial_state(self):

        r = np.zeros( [self.units], dtype=myconfig.DTYPE) # 
        v = np.zeros( [self.units, pop_size], dtype=myconfig.DTYPE) #+ self.v_rest
        w = np.zeros( [self.units, pop_size], dtype=myconfig.DTYPE)

        synaptic_matrix_shapes = self.gsyn_max.shape

        R = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE) + 0.5 #1/3# ones
        U = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE) + 0.5 #1/3
        A = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE) #+ 0.09 #1/3

        if MODE == 'honest': 
            pass
            v += self.v_rest
            # R+=1/3
            # U+=1/3
            # A+=1/3

        initial_state = [r, v, w, R, U, A]

        return initial_state
    


    def call(self, inputs, states):
        rates = states[0]
        v = states[1]
        w = states[2]
        R = states[3]
        U = states[4]
        A = states[5]
        

        g_syn = self.gsyn_max * A 
        Isyn = 0 #np.sum(g_syn * (self.e_r - v), axis=0) 

        
        v_prev = v
        noise = 0 #self.sigma*np.sqrt(self.dt)*np.random.randn(NN, pop_size)
        # print(Isyn)
        v = v + self.dt*(self.aI*v**2 + self.bI*v + self.cI - w + self.Icauchy + Isyn + self.Iext)/self.Cm + noise
        w = w + self.dt * (self.a*(self.b*(v - self.v_rest) - w))

        fired = (v >= self.v_peak) #& (v_prev < self.v_peak)
        v = np.where(fired, self.v_reset, v)
        w = np.where(fired, w + self.w_jump, w)

        rates = np.mean(fired, axis=1)/dt_dim # mean rate per ms
        gen_rates = np.hstack((inputs[:,0], inputs[:,1]))

        full_rates = np.hstack((rates, gen_rates))
        firing_prob = full_rates[:, np.newaxis, np.newaxis] * dt_dim

    
    

        a_ = A * self.exp_tau_d
        r_ = 1 + (R - 1 + self.tau1r * A) * self.exp_tau_r  - self.tau1r * A
        u_ = U * self.exp_tau_f

        released_mediator = U * r_ * firing_prob

        U = u_ + self.Uinc * (1 - u_) * firing_prob
        A = a_ + released_mediator
        R = r_ - released_mediator

        output = full_rates * 1000  # impulses per second # /self.dt
        # print('full_rates', full_rates.shape)

        Isyn_group = 0 #np.mean(Isyn, axis=1)
        Amean = 0 #np.mean(np.mean(A, axis = 2), axis=0)

        return output, [rates, v, w, R, U, A, Amean, Isyn_group]
    

    def predict0(self, inputs, initial_states=None, batch_size=1000):

        num_steps = inputs.shape[1]
        # steps4save = num_steps/10

        bar = IncrementalBar('Countdown', max = num_steps)

        if initial_states is None:
            states = self.get_initial_state()
        else:
            states = initial_states

        # Состояния не возвращаются, а записываются в h5 батчами

        
        
        with h5py.File(FILE_NAME, 'w') as hf:

            hf.create_dataset('v', (num_steps, NN, pop_size), maxshape=(None, NN, pop_size), dtype=np.float32)
            hf.create_dataset('rate', (num_steps, NN+Ninps), maxshape=(None, NN+Ninps), dtype=np.float32)
            hf.create_dataset('Isyn', (num_steps, NN), maxshape=(None, NN), dtype=np.float32)
            hf.create_dataset('Amean', (num_steps, NN), maxshape=(None, NN), dtype=np.float32)
            # hf.create_dataset('w', (num_steps, NN, pop_size), maxshape=(None, NN, pop_size), dtype=np.float32)
            # hf.create_dataset('R', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32) #, pop_size
            # hf.create_dataset('U', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32)
            # hf.create_dataset('A', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32)
            
            for i in range(0, inputs.shape[1], batch_size):
                batch = inputs[:, i:i+batch_size]
                
                # Вычисляем и записываем по батчам
                for j in range(batch.shape[1]):

                    step = i + j
                    output, hist_states = self.call(batch[:, j], states) 

                    # if step//10 ==0:
                    
                    # сохраняем данные

                    hf['rate'][step] = output #hist_states[0]  # rates (NN)
                    hf['v'][step] = hist_states[1]
                    # hf['Isyn'][step] = hist_states[7]
                    # hf['Amean'][step] = hist_states[6]

                    # hf['w'][step] = hist_states[2]      # w (NN x pop_size)
                    # hf['R'][step] = hist_states[3]      # R (NN x NN x pop_size)
                    # hf['U'][step] = hist_states[4]      # U
                    # hf['A'][step] = hist_states[5]      # A

                    states = hist_states
                    
                    # освобождаем память
                    del output
                    del hist_states

                    bar.next()
                    
                # принудительная запись на диск
                hf.flush()
            bar.finish()



    def predict(self, inputs, initial_states=None):

        num_steps = inputs.shape[1]
        rates = np.zeros((num_steps, NN+Ninps), dtype=np.float32)

        bar = IncrementalBar('Countdown', max = num_steps)

        if initial_states is None:
            states = self.get_initial_state()
        else:
            states = initial_states
    
        for i in range(0, inputs.shape[1]):
                        
            output, hist_states = self.call(inputs[:, i], states) 

            rates[i] = output

            states = hist_states
            
            bar.next()
                
            # принудительная запись на диск
        bar.finish()

        return rates
    
print('Class is ready')





######################################################################
if __name__ == '__main__':


    neuron_types = pd.read_csv('parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv', delimiter=',')
    synapse_types = pd.read_csv('parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv')


    with (open("parameters/params.pickle", "rb")) as openfile:
        while True:
            try:
                params_list = pickle.load(openfile)
            except EOFError:
                break


    types_from_table = {
        0: 'CA1 Pyramidal',
        1: 'CA1 Pyramidal',
        2: 'CA1 Axo-Axonic',
        3: 'CA1 Basket',
        4: 'CA1 Basket CCK+',
        5: 'CA1 Bistratified',
        6: 'CA1 Ivy',
        7: 'CA1 Neurogliaform',
        8: 'CA1 O-LM',
        9: 'CA1 Perforant Path-Associated',
        10: 'CA1 Interneuron Specific R-O',
        11: 'CA1 Interneuron Specific RO-O',
        12: 'CA1 Trilaminar'
    }

    types_basket = {
        0: 'CA1 Basket',
        1: 'CA1 Basket'
    }



    NN = 2 
    Ninps = 2
    if MODE == 'honest': 
        pop_size = 2000 # Количество нейронов в каждой популяции
        dt_dim = 0.01  # ms

    elif MODE == 'mean': 
        pop_size = 1 
        dt_dim = 0.01  # ms


    izh_params = {
        # фиксированные параметры, заменяются на соответствующие типам
        "Izh C": 0,  # * pF,
        "Izh Vr": 0,  # * mV,
        "Izh Vt": 0,  # *mV, 
        "Izh Vpeak": 0,  # * mV,
        "Izh Vmin": 0,
        "Izh a": 0, 
        "Izh b": 0, 
        "Izh d": 0, 
        "Izh k": 0, 

        # оптимизируемые, из net_params
        "Iext": np.zeros((NN, pop_size), dtype=np.float32), # + net_params['I_ext'][:, np.newaxis], # pA
        "Delta_eta": np.zeros(NN, dtype=np.float32) # net_params['Delta_eta']
    }



    #'''
    for key, val in izh_params.items():
        # не изменяемые параметры, из таблицы

        izh_params[key] = np.zeros((NN, pop_size), dtype=np.float32) 
        if key == 'Izh C': izh_params[key] = np.ones((NN, pop_size), dtype=np.float32) # емкости где нет связи - единицы
        for i in range(NN):
            type = types_basket[i]  # types_from_table !!!
            neuron_param = neuron_types[neuron_types['Neuron Type'] == type]

            if not neuron_param.empty:
                # оптимизированные переводим в размерные из безразмерных
                if key in ('Iext', 'Delta_eta'):
                    I = val[i]
                    k = neuron_param['Izh k'].iloc[0]
                    V_R = neuron_param['Izh Vr'].iloc[0]
                    Idim = I*(k * abs(V_R)**2) 
                    izh_params[key][i] = Idim

                # остальные из таблицы
                else:
                    izh_params[key][i] = neuron_param[key].iloc[0]

    

    izh_params['alpha'] = np.zeros((NN), dtype=np.float32)
    izh_params['alpha'][0] = 0.38344756
    izh_params['alpha'][1] = 0.38344756 # net_params['alpha']
    # print('alpha pre', izh_params['alpha'].shape)

    izh_params['Iext'][0] = 800
    izh_params['Iext'][1] = 800

    izh_params['Delta_eta'][0] = 80
    izh_params['Delta_eta'][1] = 80

    izh_params['Izh C'][1] = 114
    izh_params['Izh C'][0] = 114

    # izh_params['a'] = np.zeros((NN, pop_size)) + 0.0083115
    # izh_params['b'] = np.zeros((NN, pop_size)) + 0.0320795

    ## synaptic variables
    syn_params = {
        'g': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32), 
        'tau_d': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32), 
        'tau_r': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32), 
        'tau_f': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32), 
        'u': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32), 
        'e_r': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32), 
        'pconn': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32), 
    }

    for key, val in syn_params.items():
        # не изменяемые параметры, из таблицы
        syn_params[key] = np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) 

        for i in range(NN):
            for j in range(NN):
                pre_type = types_basket[i] 
                post_type = types_basket[j]
                syn_param = synapse_types[(synapse_types['Presynaptic Neuron Type'] == pre_type) & (synapse_types['Postsynaptic Neuron Type'] == post_type)]

                if not syn_param.empty:
                    if key == 'e_r':
                        syn_params[key][i][j] = -80
                    elif key == 'pconn':
                        syn_params[key][i][j] = 0
                    elif key == 'tau_d':
                        syn_params[key][i][j] = 6.02
                    elif key == 'tau_f':
                        syn_params[key][i][j] = 21.0
                    elif key == 'tau_r':
                        syn_params[key][i][j] = 359.8
                    else:
                        syn_params[key][i][j] = syn_param[key].iloc[0]


    syn_params['g'] = np.zeros((NN+Ninps,NN, pop_size), dtype=np.float32)
    syn_params['g'][0,1] = 20
    syn_params['g'][1,0] = 15

    syn_params['pconn'][1,0] = 1
    syn_params['pconn'][0,1] = 1

    syn_params['e_r'] = np.zeros((NN+Ninps, NN, pop_size)) -75

    syn_params['Uinc'] = np.zeros((NN+Ninps, NN, pop_size)) + 0.27

    # syn_params['tau_d'][0] = np.zeros(NN+Ninps)


    syn_params['tau_d'] = np.where(syn_params['tau_d'] == 0.0, 80.0, syn_params['tau_d'])
    syn_params['tau_f'] = np.where(syn_params['tau_f'] == 0.0, 100.0, syn_params['tau_f'])
    syn_params['tau_r'] = np.where(syn_params['tau_r'] == 0.0, 100.0, syn_params['tau_r'])



    gen_params = {'mec': params_list['generator_params'][0],
                  'lec': params_list['generator_params'][1]}


    params = izh_params | gen_params | syn_params 

    pprint(params)

    # Параметры готовы



    
    # Запуск
    #'''

    
    duration = 1000.0
    t = np.arange(0, duration, dt_dim, dtype=np.float32)
    t = t.reshape(1, -1, 1)
    t = t.ravel()

    dt_mean = 0.01
    t_mean = np.arange(0, duration, dt_mean, dtype=np.float32)
    
    

    firings_inputs = np.zeros(shape=(1, t.size, Ninps), dtype=np.float32)
    # mec_inputs, lec_inputs = generators_inputs(gen_params, t)
    # firings_inputs[:,:,0] = mec_inputs
    # firings_inputs[:,:,1] = lec_inputs


    # plt.plot(t, mec_inputs)
    # plt.show()


    mean_rates = run_mean_field(params, duration, dt_mean)

    # params['Iext'] += 4800
    # params['Izh C'] /= 8

    model = HonestNetwork(params, dt_dim=dt_dim, use_input=True)

    # init_states = model.get_initial_state()
    # one_step = model.call(firings_inputs[:,0], init_states)

    
    
    rates = model.predict(firings_inputs) # , hist_states
             
    firing_rate_honest = rates[:,0]
    smoothed_rate = gaussian_filter1d(firing_rate_honest, sigma=220)
    plt.plot(t, smoothed_rate, label='honest')
    plt.plot(t_mean, mean_rates[:,0], label='mean_field')
    plt.legend()

    plt.show()




