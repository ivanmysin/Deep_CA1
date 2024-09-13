import myconfig
import sys
sys.path.append(myconfig.SCRIPTS4PARAMSGENERATION)

import cell_anatom_pos
import interneurons_pos
import external_generators
import set_connections
import create_datasets4populations
import dl_deep_pop_model

print("Generation neurons parameters")
cell_anatom_pos.main()
interneurons_pos.main()
external_generators.main()

print("Generation synapses parameters")
set_connections.main()

print("Run comuputation datasets for population models")
create_datasets4populations.main()

print("Run fitting of population models")
dl_deep_pop_model.main()