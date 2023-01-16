import config

class Note:

    def __init__(self, number, name, frequency):

        self.number = number
        self.name = name
        self.frequency = frequency

class Notes:

    def __init__(self):

        self.notes = self.__instantiate_notes()

    def get_frequency(note_number):
        return notes[note_number].frequency

    def __instantiate_notes(self):

        notes = dict()

        with open(config.NOTES_CONVERSION_TABLE_PATH, "r") as f:
            lines = f.readlines()
            for line in lines:
                values = line.split("\t")
                number = int(values[0])
                name = values[1]
                frequency = float(values[2])
                notes[number] = Note(number, name, frequency)

        return notes
