import numpy as np

from config import TICKS_PER_BEAT, STANDARDIZE

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
                        mean_dur, std_dur, mean_del, std_del):

    sequence = np.squeeze(sample)

    frequencies = sequence[0]
    durations = sequence[1]
    deltas = sequence[2]

    if STANDARDIZE:
        frequencies = inverse_normalize_data(frequencies, max_freq, min_freq)
        durations = inverse_standardize_data(durations, mean_dur, std_dur)
        deltas = inverse_standardize_data(deltas, mean_del, std_del)

    #durations = convert_beats_to_ticks(durations)
    #deltas = convert_beats_to_ticks(deltas)

    frequencies = np.clip(frequencies, np.log10(8.176), np.log10(12543.850))
    frequencies = convert_to_pitch_scale(frequencies)

    durations = np.clip(np.int_(durations), 1, None)
    deltas = np.clip(np.int_(deltas), 0, None)

    return frequencies, durations, deltas
