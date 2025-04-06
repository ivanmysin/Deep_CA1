import os


RUNMODE = 'RELEASE' # 'DEBUG' #
DTYPE = "float32"

N_THREDS = 1
############## directorires  #################
SCRIPTS4PARAMSGENERATION = "./parameters/"
PRETRANEDMODELS = "./pretrained_models/" # Path to DL models of populations
STRUCTURESOFNET = "./presimulation_files/"  # Path to files with parameters of full net

DATASETS4POPULATIONMODELS = "/media/sdisk/Deep_CA1/new_population_datasets/"  # Path to files with datasets for traning models of populations
if not os.path.isdir(DATASETS4POPULATIONMODELS):
    DATASETS4POPULATIONMODELS = "./population_datasets/"

OUTPUTSPATH = "/media/sdisk/Deep_CA1/outputs/"  # Path to files with datasets for traning models of populations
if not os.path.isdir(OUTPUTSPATH):
    OUTPUTSPATH = "./outputs/"

if not os.path.isdir(OUTPUTSPATH):
    os.mkdir(OUTPUTSPATH)

OUTPUTSPATH_FIRINGS = OUTPUTSPATH + 'firings/'
if not os.path.isdir(OUTPUTSPATH_FIRINGS):
    os.mkdir(OUTPUTSPATH_FIRINGS)
OUTPUTSPATH_MODELS = OUTPUTSPATH + 'big_models/'
if not os.path.isdir(OUTPUTSPATH_MODELS):
    os.mkdir(OUTPUTSPATH_MODELS)
OUTPUTSPATH_PLOTS = OUTPUTSPATH + 'plots/'
if not os.path.isdir(OUTPUTSPATH_PLOTS):
    os.mkdir(OUTPUTSPATH_PLOTS)

IZHIKEVICNNEURONSPARAMS = './parameters/DG_CA2_Sub_CA3_CA1_EC_neuron_parameters06-30-2024_10_52_20.csv'
TSODYCSMARKRAMPARAMS = './parameters/DG_CA2_Sub_CA3_CA1_EC_conn_parameters06-30-2024_10_52_20.csv'
FIRINGSNEURONPARAMS = "./parameters/neurons_parameters.xlsx"


##############################################
####### global parameters for simulation #####
NUMBERNEURONSINPOP = 4000
NFILESDATASETS = 1000
DT = 0.5 ##!! 0.1 # time step, ms

IS_SAVE_V = True
GREST = 0.0000000001
DURATION = 2000 # ms

##############################################
######### fit models of populations ##########
NEPOCHES = 2000
TRAIN2TESTRATIO = 0.9
VERBOSETRANINGPOPMODELS = 0
BATCHSIZE = 200
BATCH_LEN_PART = 1
########### Net creation ######################
PCONN_THRESHOLD = 0.05



#####
ThetaFreq = 8.0

##############################################
######### fit whole model of CA1 net #########
DV_MIN = 0
DV_MAX = 200


TRACK_LENGTH = 250 # cm
ANIMAL_VELOCITY = 20 # cm/sec
N_TIMESTEPS = 240

EPOCHES_ON_BATCH = 50
EPOCHES_FULL_T = 100
