import os

#Dataframe building
VERBOSE = True
STANDARDIZE = True
USE_LOG_SCALE = False
TICKS_PER_BEAT = 960
BPM = 60
MAX_DURATION = 2048 #1468
MAX_DELTA = 2048 #960
NOTES_LENGTH = 64 # define the number of notes a piece of music can have

#Paths
NOTES_CONVERSION_TABLE_PATH = os.path.join("data","notes_conversion_table.txt")
DATAFRAME_PATH = os.path.join("data","dataframe.pkl")
STATISTICS_PATH = os.path.join("data","statistics.pkl")
DATASET_PATH = os.path.join("data","Maestro dataset")
WEIGHTS_PATH = "weights"
UNET_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"unet_{}.h5")

#Diffusion
BETA_START = 1e-4
BETA_END = 0.02
NOISE_STEPS = 1000
N_HEADS = 2
TIME_EMBEDDING_SIZE = 256

#Training
RUN_EAGERLY = True #False
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 128

#Saving/Loading
SAVE_MODEL = True
LOAD_MODEL = True
EPOCHS_BEFORE_SAVE = 50

#Sampling
N_SAMPLES = 2
SAMPLES_PATH = os.path.join("data","samples","sample_epoch{}_{}.mid")
