from mido import MidiFile
from src.midi.notes import Notes

ON = "note_on"
OFF = "note_off"

class NoteData:

    def __init__(self):

        self.frequency = None
        self.duration = None
        self.delta = None

    @property
    def is_complete(self):
        return self.frequency!=None and self.duration!=None and self.delta!=None

    def set_frequency(self, frequency):
        self.frequency = frequency

    def set_delta(self, delta):
        self.delta = delta

    def set_duration(self, duration):
        self.duration = duration

class MidiDataExtractor:

    def __init__(self, file_name, notes):

        self.file_name = file_name
        self.notes = notes

        self.midi = MidiFile(file_name, clip=True)

        self.note_datas = None

    def get_data(self):

        if self.note_datas is None:
            self.note_datas = self.__extract_data()

        return self.note_datas

    def __extract_data(self):

        main_track = self.__get_main_track()
        messages = self.__get_notes_messages(main_track)

        note_datas = []

        open_note_datas = dict()
        current_notes_durations = dict()

        last_delta = 0

        for msg in messages:

            note_number = msg.note
            last_delta += msg.time

            for note in current_notes_durations:
                current_notes_durations[note] += msg.time

            if msg.type==ON:
                note_data = NoteData()
                note_data.set_frequency(self.notes.get_frequency(note_number))
                note_data.set_delta(last_delta)
                last_delta = 0

                current_notes_durations[note_number] = 0
                open_note_datas[note_number] = note_data
                note_datas.append(note_data)

            else:
                duration = current_notes_durations.pop(note_number)
                note_data = open_note_datas.pop(note_number)
                note_data.set_duration(duration)

        return note_datas

    def __get_main_track(self):

        if len(self.midi.tracks)==1:
            return self.midi.tracks[0]

        main_track = self.midi.tracks[0]

        for track in self.midi.tracks[1:]:
            if len(track)>len(main_track):
                main_track = track

        return main_track

    def __get_notes_messages(self, track):

        messages = [msg for msg in track if msg.type==ON or msg.type==OFF]

        return messages

class MidiDataBuilder:

    def __init__(self, frequencies, durations, deltas, notes):

        self.frequencies = frequencies
        self.durations = durations
        self.deltas = deltas

        self.notes = notes

    def build_and_save(self, file_name):

        pitches = [self.notes.get_closest_note(frequency) for frequency in frequencies]

        #TODO: build messages (Notes on, notes off, durations and deltas)

        #TODO: build midi from messages, instrument ecc.

        #TODO: save midi in memory
