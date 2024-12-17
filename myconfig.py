import os


RUNMODE = 'DEBUG' #'DEBUG'
DTYPE = "float32"

N_THREDS = 8
############## directorires  #################
SCRIPTS4PARAMSGENERATION = "./parameters/"
PRETRANEDMODELS = "./pretrained_models/" # Path to DL models of populations
STRUCTURESOFNET = "./presimulation_files/"  # Path to files with parameters of full net

DATASETS4POPULATIONMODELS = "/media/sdisk/Deep_CA1/population_datasets/"  # Path to files with datasets for traning models of populations
if not os.path.isdir(DATASETS4POPULATIONMODELS):
    DATASETS4POPULATIONMODELS = "./population_datasets/"



IZHIKEVICNNEURONSPARAMS = './parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv'
TSODYCSMARKRAMPARAMS = './parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv'
FIRINGSNEURONPARAMS = "./parameters/neurons_parameters.xlsx"


##############################################
####### global parameters for simulation #####
NUMBERNEURONSINPOP = 4000
NFILESDATASETS = 1000
DT = 0.1 # time step, ms

IS_SAVE_V = False
GREST = 0.0000000001
DURATION = 2000 # ms

##############################################
######### fit models of populations ##########
NEPOCHES = 200
TRAIN2TESTRATIO = 0.9
VERBOSETRANINGPOPMODELS = 0
BATCHSIZE = 200
BATCH_LEN_PART = 1.0 # 0.25
########### Net creation ######################
PCONN_THRESHOLD = 0.01



#####
ThetaFreq = 7.0

##############################################
######### fit whole model of CA1 net #########
TRACK_LENGTH = 400 # cm
ANIMAL_VELOCITY = 20 # cm/sec
N_TIMESTEPS = 100

EPOCHES_ON_BATCH = 3
