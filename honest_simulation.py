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



class HonestNetwork:

    def __init__(self, params, dt_dim=0.01, use_input=False, **kwargs):

        self.dt = dt_dim
        self.use_input = use_input
        # self.pop_size = params('pop_size')


        self.units = len(params['Izh Vr'])
        self.a = np.asarray( params['Izh a'], dtype=myconfig.DTYPE )
        self.b = np.asarray( params['Izh b'], dtype=myconfig.DTYPE )
        self.w_jump = np.asarray( params['Izh d'], dtype=myconfig.DTYPE )

        self.Iext = np.asarray( params['Iext'], dtype=myconfig.DTYPE )*2.5
        self.sigma = params['sigma']
        
        self.v_tr = np.asarray( params['Izh Vt'], dtype=myconfig.DTYPE )
        self.v_rest = np.asarray( params['Izh Vr'], dtype=myconfig.DTYPE )
        self.v_peak = np.asarray( params['Izh Vpeak'], dtype=myconfig.DTYPE )

        self.gsyn_max = np.asarray(params['g'], dtype=myconfig.DTYPE)
        self.tau_f = np.asarray(params['tau_f'], dtype=myconfig.DTYPE)
        self.tau_d = np.asarray(params['tau_d'], dtype=myconfig.DTYPE)
        self.tau_r = np.asarray(params['tau_r'], dtype=myconfig.DTYPE)
        self.Uinc = np.asarray(params['u'], dtype=myconfig.DTYPE)
        self.e_r = np.asarray(params['e_r'], dtype=myconfig.DTYPE)

        # Применяем матрицу плотности связей
        self.pconn = np.asarray(params['pconn'])
        self.gsyn_max = np.where(np.random.uniform(0, 1, (NN+Ninps, NN, pop_size)) < self.pconn, self.gsyn_max, 0)

        self.aI = params['Izh k'] # 1
        self.bI = params['Izh k']*(-params['Izh Vr'] - params['Izh Vt']) # 98
        self.cI = params['Izh k']*params['Izh Vr']*params['Izh Vt'] #2320
        self.Cm = params['Izh C'] # 40

        # Задаем внешние токи из Коши
        DeltaI = params['Delta_eta']
        self.Icauchy = np.zeros((NN, pop_size))

        for i in range(NN):
            self.Icauchy[i] = cauchy.rvs(size=(pop_size), scale=DeltaI[i]) 

        
        self.exp_tau_d = np.exp(-self.dt / self.tau_d)
        self.exp_tau_f = np.exp(-self.dt / self.tau_f)
        self.exp_tau_r = np.exp(-self.dt / self.tau_r)
        self.tau1r = np.where(self.tau_d != self.tau_r, self.tau_d / (self.tau_d - self.tau_r), 1e-13)

    
        synaptic_matrix_shapes = self.gsyn_max.shape
        self.state_size = [self.units, self.units, self.units, synaptic_matrix_shapes, synaptic_matrix_shapes, synaptic_matrix_shapes]

    def get_initial_state(self):

        r = np.zeros( [self.units, pop_size], dtype=myconfig.DTYPE)
        v = np.zeros( [self.units, pop_size], dtype=myconfig.DTYPE) + self.v_rest
        w = np.zeros( [self.units, pop_size], dtype=myconfig.DTYPE)

        synaptic_matrix_shapes = self.gsyn_max.shape

        R = np.ones( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        U = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)
        A = np.zeros( synaptic_matrix_shapes, dtype=myconfig.DTYPE)

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
        Isyn = np.sum(g_syn * (self.e_r - v), axis=0) # [:,:,np.newaxis]
        # print('Isyn.shape', Isyn.shape)

        # Ar = np.sum(np.mean(A, axis=2), axis = 0)
        # Rr = np.sum(np.mean(R, axis=2), axis = 0)
        # Ur = np.sum(np.mean(U, axis=2), axis = 0)
        
        v_prev = v
        # noise = self.sigma*np.sqrt(self.dt)*np.random.randn(NN, pop_size)
        # print('v_prev ', v[0])
        v = v + self.dt*(self.aI*v**2 + self.bI*v + self.cI - w + self.Icauchy + Isyn + self.Iext)/self.Cm #+ noise
        # print('v', v[0])
        w = w + self.dt * (self.a*(self.b*(v + 58.0) - w))
        

        fired = (v_prev < self.v_peak) & (v >= self.v_peak)
        v = np.where(fired, self.v_rest, v)
        w = np.where(fired, w + self.w_jump, w)
        rates = np.mean(fired, axis=1)/dt_dim

        # print('rates.shape', rates.shape)

        gen_rates = np.hstack((inputs[:,0], inputs[:,1]))

        # print('mec.shape',inputs[:,0].shape )
        # print('gen_rates.shape', gen_rates.shape)

        rates = np.hstack((rates, gen_rates))
        firing_prob = rates[:, np.newaxis, np.newaxis]*dt_dim # [:, np.newaxis, np.newaxis]


        #rates, v, w = self.runge_kutta_step(rates, v, w, g)

        # if self.use_input:
        #     inputs = inputs.T * 0.001 * self.dt
        #     firing_probs = np.concatenate( [firing_probs, inputs], axis=0)


        # print('rates.shape', rates.shape)
        # print('fired.shape', fired.shape)


        a_ = A * self.exp_tau_d
        r_ = 1 + (R - 1 + self.tau1r * A) * self.exp_tau_r  - self.tau1r * A
        u_ = U * self.exp_tau_f

        # print('r_.shape', r_.shape)
        # print('firing_prob.shape', firing_prob.shape)
        # print('U.shape', U.shape)

        released_mediator = U * r_ * firing_prob

        # print('released_mediator', released_mediator.shape)

        U = u_ + self.Uinc * (1 - u_) * firing_prob
        A = a_ + released_mediator
        R = r_ - released_mediator

        # print('R.shape', R.shape)

        output = rates /self.dt  # v

        return output, [rates, v, w, R, U, A]
    

    def predict(self, inputs, initial_states=None, batch_size=1000):

        num_steps = inputs.shape[1]

        bar = IncrementalBar('Countdown', max = num_steps)

        if initial_states is None:
            states = self.get_initial_state()
        else:
            states = initial_states

        # Состояния не возвращаются, а записываются в h5 батчами
        
        with h5py.File('results.h5', 'w') as hf:

            hf.create_dataset('v', (num_steps, NN, pop_size), maxshape=(None, NN, pop_size), dtype=np.float32)
            hf.create_dataset('rate', (num_steps, NN+Ninps), maxshape=(None, NN+Ninps), dtype=np.float32)
            # hf.create_dataset('w', (num_steps, NN, pop_size), maxshape=(None, NN, pop_size), dtype=np.float32)
            # hf.create_dataset('R', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32) #, pop_size
            # hf.create_dataset('U', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32)
            # hf.create_dataset('A', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32)
            
            for i in range(0, inputs.shape[1], batch_size):
                batch = inputs[:, i:i+batch_size]
                
                # Вычисляем и записываем по батчам
                for j in range(batch.shape[1]):

                    step = i + j
                    output, hist_states = self.call(batch[:, j], states) # , hist_states
                    
                    # сохраняем данные

                    hf['rate'][step] = hist_states[0]  # rates (NN)
                    hf['v'][step] = hist_states[1]
                    # hf['w'][step] = hist_states[2]      # w (NN x pop_size)
                    # hf['R'][step] = hist_states[3]      # R (NN x NN x pop_size)
                    # hf['U'][step] = hist_states[4]      # U
                    # hf['A'][step] = hist_states[5]      # A

                    states = hist_states
                    
                    # освобождаем память
                    del output
                    del hist_states
                    bar.next()
                    # time.sleep()
                    
                # принудительная запись на диск
                hf.flush()
            bar.finish()



    # Версия возвращающая все значения, но не сохраняющая их
    def predict0(self, inputs, time_axis=1, initial_states=None):
        if initial_states is None:
            states = self.get_initial_state()
        else:
            states = initial_states
        outputs = []
        hist_states = []
        for idx in range(inputs.shape[time_axis]):
            inp = inputs[:, idx, :]

            output = self.call(inp, states) #, states

            outputs.append(output)

            for s in states:
                hist_states.append(s)

        outputs = np.stack(outputs)

        hist_states = np.stack(hist_states)
        print('outputs', outputs.shape)
        h_states = []
        for s_idx in range(len(states)):
            s = hist_states[s_idx::len(states)]
            #print(s[0].shape, s[-1].shape)

            s = np.stack(s)

            h_states.append(s)
        return outputs, h_states
    
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

    from pprint import pprint

    pprint(params_list)

    types_from_table = {
        0: 'CA1 Pyramidal',
        1: 'CA1 Pyramidal',
        2: 'CA1 Axo-Axonic',
        3: 'CA1 Basket',
        4: 'CA1 Basket CCK+',
        5: 'CA1 Bistratified',
        6: 'CA1 Ivy',
        7: 'CA1 Neuragliaform',
        8: 'CA1 O-LM',
        9: 'CA1 Perforant Path-Associated',
        10: 'CA1 Interneuron Specific R-O',
        11: 'CA1 Interneuron Specific RO-O',
        12: 'CA1 Trilaminar'
    }


    #''' если нужно попробовать на случайной меньшей сети

    # Nneur = 200
    # Nsyn = 1000 # 0.05 of all possible

    # neurons_list = [] 
    # for n in range(Nneur):
    #     i = np.random.randint(len(Types))
    #     _type = Types[i]
    #     neurons_list.append({'type': _type})

    # print(neurons_list[0]['type'])


    # synapses_list = []
    # for n in range(Nsyn):
    #     idx1 = np.random.randint(0, Nneur) # choosing random indexes of neurons to connect
    #     idx2 = np.random.randint(0, Nneur)
    #     synapses_list.append({'post_idx': idx2, 'pre_idx': idx1, 'post_type': neurons_list[idx2]['type'], 'pre_type': neurons_list[idx1]['type']})
    
    

    # print(len(neurons_list), len(synapses_list))



    NN = len(params_list['net_params']['I_ext'])
    net_params = params_list['net_params']
    Ninps = 2
    pop_size = 1000 # Количество нейронов в каждой популяции
    gen_pop_size = 100000


    izh_params = {
        # фиксированные параметры, заменяются на соответствующие типам
        "Izh C": 114,  # * pF,
        "Izh Vr": -57.63,  # * mV,
        "Izh Vt": -35.53,  # *mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
        "Izh Vpeak": 21.72,  # * mV,
        "Izh a": 0, #np.zeros((NN, pop_size), dtype=np.float32),# + net_params['a'][:, np.newaxis], # 0.005,  # * ms ** -1,
        "Izh b": 0, #np.zeros((NN, pop_size), dtype=np.float32),# + net_params['b'][:, np.newaxis], # 0.22,  # * mS,
        "Izh d": 0, #np.zeros((NN, pop_size), dtype=np.float32),# + net_params['w_jump'][:, np.newaxis], # 2,  # * pA,
        "Izh k": 0, #np.zeros((NN, pop_size), dtype=np.float32),# + net_params['alpha'][:, np.newaxis], # 1.0,

        "Iext": np.zeros((NN, pop_size), dtype=np.float32) + net_params['I_ext'][:, np.newaxis], # pA
        "sigma": 1,
        "Delta_eta": net_params['Delta_eta']
    }


    #'''
    for key, val in izh_params.items():
        # не изменяемые параметры, из таблицы
        if key not in ('Delta_eta', 'sigma'):

            izh_params[key] = np.zeros((NN, pop_size), dtype=np.float32) 
            if key == 'Izh C': izh_params[key] = np.ones((NN, pop_size), dtype=np.float32) # емкости где нет связи - единицы
            # if key == 'sigma': izh_params[key] = np.zeros((NN, pop_size), dtype=np.float32)+0.1

            for i in range(NN):
                type = types_from_table[i] #neurons_list[i]['type']
                neuron_param = neuron_types[neuron_types['Neuron Type'] == type]

                if not neuron_param.empty:
                    if key == 'Iext':
                        I = val[i]
                        k = neuron_param['Izh k'].iloc[0]
                        V_R = neuron_param['Izh Vr'].iloc[0]
                        Idim = I*(k * abs(V_R)**2) #izhs_lib.anti_transform_I(I[i,:], k, V_R)
                        izh_params[key][i,:] = Idim
                    else:
                        izh_params[key][i] = neuron_param[key].iloc[0]


    print(izh_params['Iext'])

    ## synaptic variables
    syn_params = {
        'g': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) + net_params['gsyn_max'][:,:, np.newaxis], # 200.0,
        'tau_d': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) + net_params['tau_d'][:,:, np.newaxis], # 6.02,
        'tau_r': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) + net_params['tau_r'][:,:, np.newaxis], # 359.8,
        'tau_f': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) + net_params['tau_f'][:,:, np.newaxis], # 21.0,
        'u': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) + net_params['Uinc'][:,:, np.newaxis], # 0.25,
        'e_r': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) + net_params['e_r'][:,:, np.newaxis], # 0.0 net_params['e_r'], #
        'pconn': np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) + net_params['pconn'][:,:, np.newaxis]
    }

    print(net_params['gsyn_max'].shape)
    print(net_params['I_ext'].shape)


    gen_params = {'mec': params_list['generator_params'][0],
                  'lec': params_list['generator_params'][1]}


    params = izh_params | gen_params | syn_params 

    # Параметры готовы



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
        normalizator = out_rate / I0 * 0.001  # units: Herz # not probability of spikes during dt


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
    

    def generators_inputs(gen_params, dt):

        mec = np.zeros(shape=(t.size, gen_pop_size), dtype=np.float32)
        lec = np.zeros(shape=(t.size, gen_pop_size), dtype=np.float32)

        mec_mean = np.zeros(t.size, dtype=np.float32)
        lec_mean = np.zeros(t.size, dtype=np.float32)

        for i in range(t.size):
            
            mec_mean[i] = get_gen_mean(t[i], gen_params['mec'])
            lec_mean[i] = get_gen_mean(t[i], gen_params['lec'])

            # mec[i, :] = (np.random.rand(gen_pop_size) < mec_mean[i]*dt).astype(np.float32)
            # lec[i, :] = (np.random.rand(gen_pop_size) < lec_mean[i]*dt).astype(np.float32)

        return mec_mean, lec_mean
       



    # Запуск

    # with h5py.File('results.h5', 'w') as f:
    #     pass

    #'''

    dt_dim = 0.1  # ms
    duration = 500.0
    t = np.arange(0, duration, dt_dim, dtype=np.float32)
    t = t.reshape(1, -1, 1)
    t = t.ravel()


    firings_inputs = np.zeros(shape=(1, t.size, Ninps), dtype=np.float32)

    mec_inputs, lec_inputs = generators_inputs(gen_params, dt_dim)

    # mec_inputs_mean = np.mean(mec_inputs, axis=1)

    firings_inputs[:,:,0] = mec_inputs
    firings_inputs[:,:,1] = lec_inputs

    # print(mec_inputs.shape)

    # plt.plot(t, mec_inputs_mean)
    # plt.show()



    model = HonestNetwork(params, dt_dim=dt_dim, use_input=True)

    # init_states = model.get_initial_state()
    # one_step = model.call(firings_inputs[:,0], init_states)

    #
    rates = model.predict(firings_inputs) # , hist_states


    #'''
    with h5py.File('results.h5', 'r') as f:
        num_steps = f['v'].shape[0]
        num_groups = f['v'].shape[1]
        num_neurons = f['v'].shape[2]
            
        voltages = np.zeros((num_steps, num_groups, num_neurons))
        
        # Чтение конкретного нейрона 
        group_idx = 3
        neuron_idx = 32
        v_single = f['v'][:, group_idx, neuron_idx]  # Читаем только один нейрон
        firing_rate = f['rate'][:, group_idx]
        

    t = t.ravel()

    from scipy.ndimage import gaussian_filter1d

    smoothed_rate = gaussian_filter1d(firing_rate, sigma=60)


    # fig, axes = plt.subplots(nrows=1)
    # axes[0].plot(t, firing_rate)
    # axes[0].set_title(f"Group {group_idx}, Частота разрядов")

    plt.plot(t, smoothed_rate    ) # 
    #plt.plot(t, v_single    )
    plt.title(f"Group {group_idx}, Частота разрядов")

    # axes[1].plot(t, v_single)
    # axes[1].set_title (f"Neuron {neuron_idx}, Мембранный потенциал")
    
    plt.xlabel("Время (мс)")

    plt.show()

    # '''
