

DTYPE = "float32"

N_THREDS = 8
############## directorires  #################
SCRIPTS4PARAMSGENERATION = "./parameters/"
PRETRANEDMODELS = "./pretrained_models/" # Path to DL models of populations
STRUCTURESOFNET = "./presimulation_files/"  # Path to files with parameters of full net
DATASETS4POPULATIONMODELS = "./population_datasets/"  # Path to files with datasets for traning models of populations

IZHIKEVICNNEURONSPARAMS = './parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv'
TSODYCSMARKRAMPARAMS = './parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv'
FIRINGSNEURONPARAMS = "./parameters/neurons_parameters.xlsx"


##############################################
####### global parameters for simulation #####
NUMBERNEURONSINPOP = 4000
NFILESDATASETS = 120
DT = 0.1 # time step, ms

IS_SAVE_V = False
GREST = 0.0000000001
DURATION = 2000 # ms

##############################################
######### fit models of populations ##########
NEPOCHES = 200
TRAIN2TESTRATIO = 0.7
VERBOSETRANINGPOPMODELS = 2
BATCHSIZE = 100

########### Net creation ######################
PCONN_THRESHOLD = 0.01



#####
ThetaFreq = 7.0
