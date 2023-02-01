import pickle

import numpy as np
import tensorflow as tf

from src.midi.notes import Notes
from src.preprocessing.dataframe_builder import DataframeHandler
import src.preprocessing.numerical_processing as numerical_processing

from src.model.callbacks import SaveUpdateStepCallback
from src.model.model_handler import ModelHandler

from config import *

def load_dataset(notes):

    dataframe_handler = DataframeHandler(DATASET_PATH, notes, VERBOSE)
    dataframe = dataframe_handler.get()

    if USE_LOG_SCALE:

        if VERBOSE:
            print("Converting pitches to log scale")

        numerical_processing.convert_to_log_scale(dataframe, "Frequencies")

    if VERBOSE:
        print("Converting ticks to beats")

    numerical_processing.convert_ticks_to_beats(dataframe, "Durations")
    numerical_processing.convert_ticks_to_beats(dataframe, "Deltas")

    if STANDARDIZE:

        if VERBOSE:
            print("Standardizing dataframe")

        max_freq, min_freq = numerical_processing.normalize_data(dataframe, "Frequencies")
        #mean_freq, std_freq = numerical_processing.standardize_data(dataframe, "Frequencies")
        max_dur, min_dur = numerical_processing.normalize_data(dataframe, "Durations")
        max_del, min_del = numerical_processing.normalize_data(dataframe, "Deltas")

        normalization_dict = {
        "max_freq":max_freq,
        "min_freq":min_freq,
        "max_dur":max_dur,
        "min_dur":min_dur,
        "max_del":max_del,
        "min_del":min_del
        }

        with open(STATISTICS_PATH, 'wb') as handle:
            pickle.dump(normalization_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        normalization_dict = dict()

    frequencies = np.stack(dataframe["Frequencies"], axis=0) #(Batch size, W)
    durations = np.stack(dataframe["Durations"], axis=0) #(Batch size, W)
    deltas = np.stack(dataframe["Deltas"], axis=0) #(Batch size, W)

    dataset = np.stack((frequencies, durations, deltas), axis=1) #(Batch size, 3, W)
    dataset = np.expand_dims(dataset, axis=-1) #(Batch size, 3, W, 1)

    return dataset, normalization_dict

if __name__ == '__main__':

    notes = Notes()

    dataset, normalization_dict = load_dataset(notes)

    if VERBOSE:
        print("Dataset preprocessed and loaded successfully")
        print("There are {} samples in the dataset".format(len(dataset)))

    input_shape = (3, NOTES_LENGTH, 1)

    model_handler = ModelHandler(notes, input_shape, N_HEADS, TIME_EMBEDDING_SIZE,
                                BETA_START, BETA_END, NOISE_STEPS, normalization_dict,
                                load_model=LOAD_MODEL)
    model_handler.build_model()

    callback = SaveUpdateStepCallback(model_handler)

    remaining_epochs = EPOCHS - model_handler.current_step

    model_handler.model.fit(x=dataset,
                            batch_size=BATCH_SIZE,
                            callbacks=[callback],
                            epochs=remaining_epochs)
