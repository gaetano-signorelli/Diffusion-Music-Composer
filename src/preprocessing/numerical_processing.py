import numpy as np

from config import TICKS_PER_BEAT, STANDARDIZE, USE_LOG_SCALE

def convert_to_log_scale(dataframe, column):

    dataframe[column] = dataframe[column].apply(
    lambda sequence: np.log10(sequence)
    )

def convert_ticks_to_beats(dataframe, column):

    dataframe[column] = dataframe[column].apply(
    lambda sequence: sequence / float(TICKS_PER_BEAT)
    )

def get_standard_statistics(dataframe, column):

    data = np.concatenate(dataframe[column], axis=None)
    mean = np.mean(data)
    std = np.std(data)

    return mean, std

def get_normal_statistics(dataframe, column):

    data = np.concatenate(dataframe[column], axis=None)
    max = np.amax(data)
    min = np.amin(data)

    return max, min

def standardize_data(dataframe, column):

    mean, std = get_standard_statistics(dataframe, column)

    dataframe[column] = dataframe[column].apply(
    lambda sequence: (sequence - mean) / std
    )

    return mean, std

def normalize_data(dataframe, column, negative_range=True):

    max, min = get_normal_statistics(dataframe, column)

    if negative_range:
        dataframe[column] = dataframe[column].apply(
        lambda sequence: 2 * ((sequence-min)/(max-min)) - 1
        )

    else:
        dataframe[column] = dataframe[column].apply(
        lambda sequence: (sequence-min)/(max-min)
        )

    return max, min

def convert_to_pitch_scale(sequence):
    return 10**sequence

def convert_beats_to_ticks(sequence):
    return np.int_(sequence * TICKS_PER_BEAT)

def inverse_standardize_data(sequence, mean, std):
    return (sequence * std) + mean

def inverse_normalize_data(sequence, max, min, negative_range=True):

    if negative_range:
        return ((sequence + 1) / 2 * (max-min)) + min

    else:
        return (sequence * (max-min)) + min

def sample_to_midi_values(sample, max_freq, min_freq,
                        max_dur, min_dur, max_del, min_del):

    sequence = sample[0] #(Notes length, 3)
    sequence = np.transpose(sequence) #(3, Notes length)

    frequencies = sequence[0]
    durations = sequence[1]
    deltas = sequence[2]

    if STANDARDIZE:
        frequencies = inverse_normalize_data(frequencies, max_freq, min_freq)
        durations = inverse_normalize_data(durations, max_dur, min_dur)
        deltas = inverse_normalize_data(deltas, max_del, min_del)

    #durations = convert_beats_to_ticks(durations)
    #deltas = convert_beats_to_ticks(deltas)

    if USE_LOG_SCALE:
        frequencies = np.clip(frequencies, np.log10(min_freq), np.log10(max_freq))
        frequencies = convert_to_pitch_scale(frequencies)

    else:
        frequencies = np.clip(frequencies, min_freq, max_freq)

    durations = np.clip(durations, min_dur, max_dur)
    deltas = np.clip(deltas, min_del, max_del)

    return frequencies, durations, deltas
