import os

#Dataframe building
VERBOSE = True
STANDARDIZE = True
TICKS_PER_BEAT = 960
BPM = 120
NOTES_LENGTH = 256 # define the number of notes a piece of music can have

#Paths
NOTES_CONVERSION_TABLE_PATH = os.path.join("data","notes_conversion_table.txt")
DATAFRAME_PATH = os.path.join("data","dataframe.pkl")
DATASET_PATH = os.path.join("data","Classical music dataset")
WEIGHTS_PATH = "weights"
UNET_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH,"unet_{}.h5")

#Diffusion
BETA_START = 1e-4
BETA_END = 0.02
NOISE_STEPS = 1000
N_HEADS = 1
TIME_EMBEDDING_SIZE = 256

#Training
RUN_EAGERLY = True #False
LEARNING_RATE = 1e-4 #3e-6
EPOCHS = 100
BATCH_SIZE = 4

#Saving/Loading
SAVE_MODEL = False
LOAD_MODEL = True
EPOCHS_BEFORE_SAVE = 100

#Sampling
N_SAMPLES = 1
SAMPLES_PATH = os.path.join("data","samples","sample_epoch{}_{}.mid")
