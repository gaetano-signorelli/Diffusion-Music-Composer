import os
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np

from src.midi.midi_converter import MidiDataExtractor

from config import DATAFRAME_PATH, NOTES_LENGTH, MAX_DURATION, MAX_DELTA

class DataframeHandler:

    def __init__(self, dataset_path, notes, verbose=True):

        self.dataset_path = dataset_path
        self.notes = notes
        self.verbose = verbose

        self.dataframe = None

    def get(self):

        if self.dataframe is None:
            self.dataframe, save = self.__load()

            if save:
                self.save()

        return self.dataframe

    def save(self, path=DATAFRAME_PATH):

        if self.verbose:
            print("Saving dataframe to file", path)

        self.dataframe.to_pickle(path)

    def __load(self):

        if os.path.exists(DATAFRAME_PATH):
            dataframe = pd.read_pickle(DATAFRAME_PATH)
            save = False

            if self.verbose:
                print("Loaded dataframe from file")

        else:

            if self.verbose:
                print("Dataframe not found in memory: building started")

            dataframe = self.__build()
            save = True

            if self.verbose:
                print("Dataframe built successfully")

        return dataframe, save

    def __build(self):

        dataframe_rows = []

        for file in tqdm(os.listdir(self.dataset_path)):
            if file.endswith(".mid") or file.endswith(".midi"):
                file_path = os.path.join(self.dataset_path, file)

                try:
                    data_extractor = MidiDataExtractor(file_path, self.notes)
                    note_datas = data_extractor.get_data()

                    frequencies, durations, deltas = self.__collect_data(note_datas)

                    if not self.__is_valid_data(frequencies, durations, deltas):
                        continue

                    chunk_frequencies = self.__split_data(frequencies)
                    chunk_durations = self.__split_data(durations)
                    chunk_deltas = self.__split_data(deltas)

                    assert len(chunk_frequencies)==len(chunk_durations)
                    assert len(chunk_durations)==len(chunk_deltas)

                    for i in range(len(chunk_frequencies)):

                        row = {
                        "File name": file,
                        "Frequencies": chunk_frequencies[i],
                        "Durations": chunk_durations[i],
                        "Deltas": chunk_deltas[i]
                        }

                        dataframe_rows.append(row)

                except:
                    continue

        dataframe = pd.DataFrame(dataframe_rows)

        return dataframe

    def __is_valid_data(self, frequencies, durations, deltas):

        if (frequencies==None).any() or (durations==None).any() or (deltas==None).any():
            return False

        if (np.isnan(frequencies)).any() or (np.isnan(durations)).any() or (np.isnan(deltas)).any():
            return False

        if (durations > MAX_DURATION).any() or (deltas > MAX_DELTA).any():
            return False

        return True

    def __collect_data(self, note_datas):

        frequencies = []
        durations = []
        deltas = []

        for note_data in note_datas:
            frequencies.append(note_data.frequency)
            durations.append(note_data.duration)
            deltas.append(note_data.delta)

        frequencies = np.array(frequencies)
        durations = np.array(durations)
        deltas = np.array(deltas)

        return frequencies, durations, deltas

    def __split_data(self, data):

        len_data = data.shape[-1]

        n_chunks = len_data//NOTES_LENGTH
        max_len = n_chunks * NOTES_LENGTH

        truncated_data = data[0:max_len]

        if n_chunks > 0:
            splitted_data = np.array_split(truncated_data, n_chunks)

        else:
            splitted_data = []

        return splitted_data
