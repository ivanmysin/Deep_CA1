import numpy as np
import myconfig
from myutils import get_net_params, get_gen_params
from np_meanfield import MeanFieldNetwork, SpatialThetaGenerators
import matplotlib.pyplot as plt

model_path = './outputs/big_models/big_model_3759.keras'

params = get_net_params(model_path)
generators_params = get_gen_params(model_path)

dt = myconfig.DT

generators = SpatialThetaGenerators(generators_params)
tnp = np.arange(0, 800, dt, dtype=np.float32).reshape(1, -1, 1)

generators_firings = generators.call(tnp)


# model = MeanFieldNetwork(params, dt_dim=dt, use_input=True)
#
tnp = tnp.ravel()
# #plt.plot(model.I_ext)
#
#
# # plt.plot(model.gsyn_max.ravel())
#
# npfirings, states = model.predict(generators_firings)
#
# initial_states = [s[-1] for s in states]
#
# model.v_threshold = 10000
# npfirings, states = model.predict(generators_firings, initial_states=initial_states)
#
# npfirings = npfirings.reshape(-1, npfirings.shape[-1])


generators_firings = generators_firings.reshape(-1, generators_firings.shape[-1])

plt.plot(tnp, generators_firings)
plt.show()

