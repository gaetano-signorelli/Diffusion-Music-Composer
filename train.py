import numpy as np

from src.midi.notes import Notes
import DataframeHandler from src.preprocessing.dataframe_builder
import src.preprocessing.numerical_preprocessing as numerical_preprocessing

from config import *

def load_dataset(notes):

    dataframe_handler = DataframeHandler(DATASET_PATH, notes, VERBOSE)
    dataframe = dataframe_handler.get()

    if VERBOSE:
        print("Converting pitches to log scale")

    numerical_preprocessing.convert_to_log_scale(dataframe, "Frequencies")

    if VERBOSE:
        print("Converting seconds to milliseconds")

    numerical_preprocessing.convert_ms_to_s(dataframe, "Durations")
    numerical_preprocessing.convert_ms_to_s(dataframe, "Deltas")

    if STANDARDIZE:

        if VERBOSE:
            print("Standardizing dataframe")

        mean_freq, std_freq = numerical_preprocessing.standardize_data(dataframe, "Frequencies")
        mean_dur, std_dur = numerical_preprocessing.standardize_data(dataframe, "Durations")
        mean_del, std_del = numerical_preprocessing.standardize_data(dataframe, "Deltas")

        #TODO save means and stds for inference

    frequencies = np.stack(dataframe["Frequencies"], axis=0) #(Batch size, W)
    durations = np.stack(dataframe["Durations"], axis=0) #(Batch size, W)
    deltas = np.stack(dataframe["Deltas"], axis=0) #(Batch size, W)

    dataset = np.stack((frequencies, durations, deltas), axis=-1) #(Batch size, W, 3)
    dataset = np.expand_dims(dataset, axis=1) #(Batch size, 1, W, 3)

    return dataset

if __name__ == '__main__':

    notes = Notes()

    dataset = load_dataset(notes)
