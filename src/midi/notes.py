import numpy as np

import config

class Note:

    def __init__(self, number, name, frequency):

        self.number = number
        self.name = name
        self.frequency = frequency

class Notes:

    def __init__(self):

        self.notes, self.frequencies = self.__instantiate_notes()

    def get_frequency(self, note_number):
        return self.notes[note_number].frequency

    def get_name(self, note_number):
        return self.notes[note_number].name

    def get_closest_note(self, frequency):

        dist = np.sum((frequencies - frequency)**2)
        return np.argmin(dist)

    def __instantiate_notes(self):

        notes = dict()
        frequencies = []

        with open(config.NOTES_CONVERSION_TABLE_PATH, "r") as f:
            lines = f.readlines()
            for line in lines:
                values = line.split("\t")

                number = int(values[0])
                name = values[1]
                frequency = float(values[2])

                frequencies.append(frequency)
                notes[number] = Note(number, name, frequency)

        return notes, np.asarray(frequencies)
