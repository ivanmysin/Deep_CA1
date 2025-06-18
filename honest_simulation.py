import numpy as np
import matplotlib.pyplot as plt
import izhs_lib
import pickle
import pandas as pd
import h5py
import sys
sys.path.append('../')
import myconfig


class HonestNetwork:

    def __init__(self, params, dt_dim=0.01, use_input=False, **kwargs):

        self.dt = dt_dim
        self.use_input = use_input
        # self.pop_size = params('pop_size')


        self.units = len(params['Izh Vr'])
        self.a = np.asarray( params['Izh a'], dtype=myconfig.DTYPE )
        self.b = np.asarray( params['Izh b'], dtype=myconfig.DTYPE )
        self.w_jump = np.asarray( params['Izh d'], dtype=myconfig.DTYPE )

        self.Iext = np.asarray( params['Iext'], dtype=myconfig.DTYPE )
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
        self.pconn = np.asarray(params['pconn'])

        self.aI = params['Izh k'] # 1
        self.bI = params['Izh k']*(-params['Izh Vr'] - params['Izh Vt']) # 98
        self.cI = params['Izh k']*params['Izh Vr']*params['Izh Vt'] #2320
        self.Cm = params['Izh C'] # 40

        # conn01 = np.random.random((pop_size,)) < params['pconn'][0,1]
        # conn10 = np.random.random((pop_size,)) < params['pconn'][1,0]
        

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
        Isyn = np.sum(g_syn * (self.e_r - v), axis=0)

        # Ar = np.sum(np.mean(A, axis=2), axis = 0)
        # Rr = np.sum(np.mean(R, axis=2), axis = 0)
        # Ur = np.sum(np.mean(U, axis=2), axis = 0)
        
        v_prev = v
        noise = self.sigma*np.sqrt(self.dt)*np.random.randn(NN, pop_size)
        # print('v_prev ', v[0])
        v = v + self.dt*(self.aI*v**2 + self.bI*v + self.cI - w + Isyn + self.Iext)/self.Cm + noise
        # print(((self.aI*v**2 + self.bI*v + self.cI - w + Isyn + self.Iext)/self.Cm)[0])
        # print('v', v[0])
        w = w + self.dt * (self.a*(self.b*(v + 58.0) - w))
        

        fired = (v_prev < self.v_peak) & (v >= self.v_peak)
        v = np.where(fired, self.v_rest, v)
        w = np.where(fired, w + self.w_jump, w)
        rates = np.mean(fired, axis=1)
        # print(rates)


        #rates, v, w = self.runge_kutta_step(rates, v, w, g)

        # if self.use_input:
        #     inputs = inputs.T * 0.001 * self.dt
        #     firing_probs = np.concatenate( [firing_probs, inputs], axis=0)


        with np.errstate(divide='ignore', invalid='ignore'):
            try:
                inv_tau_1r = np.divide(1, (self.tau_d - self.tau_r), where=((self.tau_d - self.tau_r)>=0.1))
                tau1r = np.exp(-self.dt * inv_tau_1r)
                tau1r[(self.tau_d - self.tau_r)<0.1] = 0

                inv_tau_d = np.divide(1, self.tau_d, where=(self.tau_d >= 0.1))
                exp_tau_d = np.exp(-self.dt * inv_tau_d)
                exp_tau_d[self.tau_d < 0.1] = 0

                inv_tau_r = np.divide(1, self.tau_r, where=(self.tau_r >= 0.1))
                exp_tau_r = np.exp(-self.dt * inv_tau_r)
                exp_tau_r[self.tau_r < 0.1] = 0

                inv_tau_f = np.divide(1, self.tau_f, where=(self.tau_f >= 0.1))
                exp_tau_f = np.exp(-self.dt * inv_tau_f)
                exp_tau_f[self.tau_f < 0.1] = 0
            except RuntimeWarning: print(self.tau_d, self.tau_r, self.tau_f)


        # exp_tau_d = np.exp(-self.dt / self.tau_d)
        # exp_tau_f = np.exp(-self.dt / self.tau_f)
        # exp_tau_r = np.exp(-self.dt / self.tau_r)


        a_ = A * exp_tau_d
        r_ = 1 + (R - 1 + tau1r * A) * exp_tau_r  - tau1r * A
        u_ = U * exp_tau_f

        released_mediator = U * r_ * fired

        U = u_ + self.Uinc * (1 - u_) * fired
        A = a_ + released_mediator
        R = r_ - released_mediator

        output = rates /self.dt  # v

        return output, [rates, v, w, R, U, A]
    

    def predict(self, inputs, initial_states=None, batch_size=1000):

        num_steps = inputs.shape[1]

        if initial_states is None:
            states = self.get_initial_state()
        else:
            states = initial_states

        # Состояния не возвращаются, а записываются в h5 батчами
        
        with h5py.File('results.h5', 'w') as hf:
        # расширяемые датасеты
            hf.create_dataset('v', (num_steps, NN, pop_size), maxshape=(None, NN, pop_size), dtype=np.float32)
            hf.create_dataset('rate', (num_steps, NN), maxshape=(None, NN), dtype=np.float32)
            hf.create_dataset('w', (num_steps, NN, pop_size), maxshape=(None, NN, pop_size), dtype=np.float32)
            hf.create_dataset('R', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32)
            hf.create_dataset('U', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32)
            hf.create_dataset('A', (num_steps, NN+Ninps, NN, pop_size), maxshape=(None, NN+Ninps, NN, pop_size), dtype=np.float32)
            
            for i in range(0, inputs.shape[1], batch_size):
                batch = inputs[:, i:i+batch_size]
                
                # Вычисляем и записываем по батчам
                for j in range(batch.shape[1]):

                    step = i + j
                    # print(step)
                    output, hist_states = self.call(batch[:, j], states) # , hist_states
                    # print('output ', output[0])
                    
                    # сохраняем данные

                    hf['rate'][step] = hist_states[0]  # rates (NN)
                    hf['v'][step] = hist_states[1]
                    hf['w'][step] = hist_states[2]      # w (NN x pop_size)
                    hf['R'][step] = hist_states[3]      # R (NN x NN x pop_size)
                    hf['U'][step] = hist_states[4]      # U
                    hf['A'][step] = hist_states[5]      # A

                    states = hist_states
                    
                    # освобождаем память
                    del output
                    del hist_states
                    
                # принудительная запись на диск
                hf.flush()



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

    # Load neurons and synapses lists with indexes
    with (open("presimulation_files/neurons.pickle", "rb")) as openfile:
        while True:
            try:
                neurons_list = pickle.load(openfile)
            except EOFError:
                break

    with (open("presimulation_files/connections.pickle", "rb")) as openfile:
        while True:
            try:
                synapses_list = pickle.load(openfile)
            except EOFError:
                break

    # Just a set of all given types:
    Types = set()
    for neuron_ in neurons_list:
        key = neuron_['type']
        Types.add(key)
    Types = list(Types)


    #''' если нужно попробовать на случайной меньшей сети

    Nneur = 200
    Nsyn = 1000 # 0.05 of all possible

    neurons_list = [] 
    for n in range(Nneur):
        i = np.random.randint(len(Types))
        _type = Types[i]
        neurons_list.append({'type': _type})

    print(neurons_list[0]['type'])


    synapses_list = []
    for n in range(Nsyn):
        idx1 = np.random.randint(0, Nneur) # choosing random indexes of neurons to connect
        idx2 = np.random.randint(0, Nneur)
        synapses_list.append({'post_idx': idx2, 'pre_idx': idx1, 'post_type': neurons_list[idx2]['type'], 'pre_type': neurons_list[idx1]['type']})
    
    #'''

    # print(len(neurons_list), len(synapses_list))


    NN = len(neurons_list)
    Ninps = 3
    pop_size = 2 # Количество нейронов в каждой популяции

    dim_izh_params = {
        "Izh C": 114,  # * pF,
        "Izh Vr": -57.63,  # * mV,
        "Izh Vt": -35.53,  # *mV, # np.random.normal(loc=-35.53, scale=4.0, size=NN) * mV,  # -35.53*mV,
        "Izh Vpeak": 21.72,  # * mV,
        "Izh a": 0.005,  # * ms ** -1,
        "Izh b": 0.22,  # * mS,
        "Izh d": 2,  # * pA,
        "Izh k": 1.0,

        "Iext": 200,  # pA
        "sigma": 10,
    }


    izh_params = dim_izh_params # | cauchy_dencity_params


    for key, val in izh_params.items():
        # начальные пустые матрицы
        izh_params[key] = np.zeros((NN, pop_size), dtype=np.float32) 
        # if key == 'Izh C': izh_params[key] = np.ones((NN, pop_size), dtype=np.float32) # емкости где нет связи - единицы

        # Ток не задан в параметрах neurons_params, так что здесь все одинаковые просто 
        # (добавлю нужные значения как получу оптимальные параметры)
        if key == 'Iext': izh_params[key] = np.zeros((NN, pop_size), dtype=np.float32)+100
        if key == 'sigma': izh_params[key] = np.zeros((NN, pop_size), dtype=np.float32)+0.1

        for i in range(NN):
            type = neurons_list[i]['type']
            neuron_param = neuron_types[neuron_types['Neuron Type'] == type]

            if not neuron_param.empty and (key not in ('Iext', 'sigma')):
                izh_params[key][i] = neuron_param[key].iloc[0]



    ## synaptic static variables
    syn_params = {
        'g': 200.0,
        'tau_d': 6.02,
        'tau_r': 359.8,
        'tau_f': 21.0,
        'u': 0.25,
        'e_r': 0.0
    }

    for key, val in syn_params.items():
        syn_params[key] = np.zeros((NN+Ninps, NN, pop_size), dtype=np.float32) # , pop_size)
        print(f'preparing {key}')
        
        for syn in synapses_list:
            pre_type = syn['pre_type']
            post_type = syn['post_type']

            pre_idx = syn['pre_idx']
            post_idx = syn['post_idx']

            synapse_type = synapse_types[(synapse_types['Presynaptic Neuron Type'] == pre_type) & (synapse_types['Postsynaptic Neuron Type'] == post_type)]
            neuron_type = neuron_types[neuron_types['Neuron Type'] == type]


            if not synapse_type.empty:
                # Устанавливаем равновесный потенциал синапса, в зависимости от того - возбуждающий или тромозный пресинапс
                if key == 'e_r':
                    if neuron_type['E/I'].iloc[0] == 'e': syn_params['e_r'][pre_idx][post_idx] = 0.0
                    else: syn_params['e_r'][pre_idx][post_idx] = -80.0
                else:
                    syn_params[key][pre_idx][post_idx] = synapse_type[key].iloc[0]
            else: syn_params[key][pre_idx][post_idx] = 0.0

        print(key + ' is ready')


            # print(f'synapse is ready {pre_idx, post_idx}')

    # print(syn_params)

    # pconn пока не используется
    pconn = np.ones(shape=(NN+Ninps, NN, pop_size), dtype=np.float32)
    syn_params['pconn'] = pconn

    params = izh_params| syn_params

    # Параметры готовы



    # перед запуском на всякий очищаем то куда записывать будем

    with h5py.File('results.h5', 'a') as f:  # 'a' - режим редактирования
        if 'v' in f:
            del f['v']  


    # Запуск

    dt_dim = 0.1  # ms
    duration = 100.0
    t = np.arange(0, duration, dt_dim, dtype=np.float32)
    t = t.reshape(1, -1, 1)

    firings_inputs = np.zeros(shape=(1, t.size, Ninps), dtype=np.float32)


    model = HonestNetwork(params, dt_dim=dt_dim, use_input=True)
    rates = model.predict(firings_inputs) # , hist_states



    with h5py.File('results.h5', 'r') as f:
        num_steps = f['v'].shape[0]
        num_groups = f['v'].shape[1]
        num_neurons = f['v'].shape[2]
            
        voltages = np.zeros((num_steps, num_groups, num_neurons))
        
        # Чтение конкретного нейрона 
        group_idx = 0
        neuron_idx = 0 
        v_single = f['v'][:, group_idx, neuron_idx]  # Читаем только один нейрон
        firing_rate = f['rate'][:, group_idx]
        

    t = t.ravel()


    fig, axes = plt.subplots(nrows=2)
    axes[0].plot(t, firing_rate)
    axes[0].set_title(f"Group {group_idx}, Частота разрядов")

    axes[1].plot(t, v_single)
    axes[1].set_title (f"Neuron {neuron_idx}, Мембранный потенциал")
    
    plt.xlabel("Время (мс)")

    plt.show()