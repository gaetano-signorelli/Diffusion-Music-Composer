import os
import pickle
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, callbacks, losses, optimizers

from src.midi.notes import Notes
from src.preprocessing.dataframe_builder import DataframeHandler
import src.preprocessing.numerical_processing as numerical_processing

from src.model.callbacks import SaveUpdateStepCallback
from src.model.model_handler import ModelHandler
from src.model.unet import UNet

from config import *

def parse_arguments():

    parser = argparse.ArgumentParser(description='Diffusion Music Composer Trainer')
    parser.add_argument('model', type=str, help='Model to be trained: must be one between "frequency", "duration" and "delta"')

    args = parser.parse_args()

    model_name = args.model

    if not model_name in ["frequency", "duration", "delta"]:
        raise Exception ("Model must be one between \"frequency\", \"duration\" and \"delta\"")

    return model_name

def load_dataset(notes):

    dataframe_handler = DataframeHandler(DATASET_PATH, notes, VERBOSE)
    dataframe = dataframe_handler.get()

    if VERBOSE:
        print("Converting pitches to log scale")

    if USE_LOG_SCALE:
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
        #mean_dur, std_dur = numerical_processing.standardize_data(dataframe, "Durations")
        max_del, min_del = numerical_processing.normalize_data(dataframe, "Deltas")
        #mean_del, std_del = numerical_processing.standardize_data(dataframe, "Deltas")

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

    frequencies = np.expand_dims(frequencies, axis=1) #(Batch size, 1, W)
    frequencies = np.expand_dims(frequencies, axis=-1) #(Batch size, 1, W, 1)

    durations = np.expand_dims(durations, axis=1) #(Batch size, 1, W)
    durations = np.expand_dims(durations, axis=-1) #(Batch size, 1, W, 1)

    deltas = np.expand_dims(deltas, axis=1) #(Batch size, 1, W)
    deltas = np.expand_dims(deltas, axis=-1) #(Batch size, 1, W, 1)

    return frequencies, durations, deltas, normalization_dict

def train_frequency_diffusion_model(notes, input_shape, frequencies, normalization_dict):

    model_handler = ModelHandler(notes, input_shape, N_HEADS, TIME_EMBEDDING_SIZE,
                                BETA_START, BETA_END, NOISE_STEPS, normalization_dict,
                                load_model=LOAD_MODEL)
    model_handler.build_model()

    callback = SaveUpdateStepCallback(model_handler)

    remaining_epochs = EPOCHS - model_handler.current_step

    model_handler.model.fit(x=frequencies,
                            batch_size=BATCH_SIZE,
                            callbacks=[callback],
                            epochs=remaining_epochs)

def train_prediction_model(input_shape, frequencies, target, save_path):

    model = UNet(input_shape, N_HEADS, None)

    inputs = Input(shape=input_shape)
    model(inputs)

    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = losses.MeanSquaredError()
    model.compile(loss=loss, optimizer=optimizer, run_eagerly=RUN_EAGERLY)

    if VERBOSE:
        model.summary()

    checkpoint_callback = callbacks.ModelCheckpoint(
                            save_path,
                            monitor="loss",
                            verbose=1,
                            save_best_only=False,
                            save_weights_only=True,
                            mode="auto",
                            save_freq="epoch"
                            )

    model.fit(x=frequencies,
            y=target,
            batch_size=BATCH_SIZE,
            callbacks=[checkpoint_callback],
            epochs=TIMES_PREDICTIONS_EPOCHS)

if __name__ == '__main__':

    notes = Notes()

    frequencies, durations, deltas, normalization_dict = load_dataset(notes)

    if VERBOSE:
        print("Dataset preprocessed and loaded successfully")
        print("There are {} samples in the dataset".format(len(frequencies)))

    input_shape = (1, NOTES_LENGTH, 1)

    model_name = parse_arguments()

    if model_name=="frequency":
        train_frequency_diffusion_model(notes, input_shape, frequencies, normalization_dict)

    elif model_name=="duration":
        path = os.path.join(WEIGHTS_PATH_DURATION, "unet.h5")
        train_prediction_model(input_shape, frequencies, durations, path)

    elif model_name=="delta":
        path = os.path.join(WEIGHTS_PATH_DELTA, "unet.h5")
        train_prediction_model(input_shape, frequencies, deltas, path)
