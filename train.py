import numpy as np

from src.midi.notes import Notes
from src.preprocessing.dataframe_builder import DataframeHandler
import src.preprocessing.numerical_processing as numerical_processing

from config import *

def load_dataset(notes):

    dataframe_handler = DataframeHandler(DATASET_PATH, notes, VERBOSE)
    dataframe = dataframe_handler.get()

    if VERBOSE:
        print("Converting pitches to log scale")

    numerical_processing.convert_to_log_scale(dataframe, "Frequencies")

    if VERBOSE:
        print("Converting seconds to milliseconds")

    numerical_processing.convert_ms_to_s(dataframe, "Durations")
    numerical_processing.convert_ms_to_s(dataframe, "Deltas")

    if STANDARDIZE:

        if VERBOSE:
            print("Standardizing dataframe")

        max_freq = numerical_processing.normalize_data(dataframe, "Frequencies")
        #mean_freq, std_freq = numerical_processing.standardize_data(dataframe, "Frequencies")
        mean_dur, std_dur = numerical_processing.standardize_data(dataframe, "Durations")
        mean_del, std_del = numerical_processing.standardize_data(dataframe, "Deltas")

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
    print(dataset.shape)
