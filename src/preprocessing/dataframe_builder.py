import os
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np

from src.midi.midi_converter import MidiDataExtractor

from config import DATAFRAME_PATH, NOTES_LENGTH

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
                self.__save()

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
            if file.endswith(".mid"):
                file_path = os.path.join(self.dataset_path, file)

                data_extractor = MidiDataExtractor(file_path, self.notes)
                note_datas = data_extractor.get_data()

                frequencies, durations, deltas = self.__collect_data(note_datas)

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

        dataframe = pd.DataFrame(dataframe_rows)

        return dataframe

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
        chunks = int(len_data/NOTES_LENGTH)

        splitted_data = np.array_split(data, chunks)

        last_data = splitted_data[-1]
        if (NOTES_LENGTH != last_data.shape[-1]):
            splitted_data = splitted_data[:-1]

        return splitted_data
