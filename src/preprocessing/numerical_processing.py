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

def get_statistics(dataframe, column):

    data = np.concatenate(dataframe[column], axis=None)
    mean = np.mean(data)
    std = np.std(data)
    max = np.amax(data)

    return mean, std, max

def standardize_data(dataframe, column):

    mean, std, max = get_statistics(dataframe, column)

    dataframe[column] = dataframe[column].apply(
    lambda sequence: (sequence - mean) / std
    )

    return mean, std

def normalize_data(dataframe, column, negative_range=True):

    mean, std, max = get_statistics(dataframe, column)

    if negative_range:
        dataframe[column] = dataframe[column].apply(
        lambda sequence: 2 * (sequence / max) - 1
        #lambda sequence: 2* ((sequence-min)/(max-min)) - 1
        )

    else:
        dataframe[column] = dataframe[column].apply(
        lambda sequence: sequence / max
        )

    return max

def convert_to_pitch_scale(sequence):
    return 10**sequence

def convert_beats_to_ticks(sequence):
    return np.int_(sequence * TICKS_PER_BEAT)

def inverse_standardize_data(sequence, mean, std):
    return (sequence * std) + mean

def inverse_normalize_data(sequence, max, negative_range=True):

    if negative_range:
        return (sequence + 1) / 2 * max
        #return ((sequence + 1) / 2 *(mix-min)) + min

    else:
        return sequence * max

def sample_to_midi_values(sample, max_freq, mean_dur, std_dur, mean_del, std_del):

    sequence = sample[0] #(Notes length, 3)
    sequence = np.transpose(sequence) #(3, Notes length)

    frequencies = sequence[0]
    durations = sequence[1]
    deltas = sequence[2]

    if STANDARDIZE:
        frequencies = inverse_normalize_data(frequencies, max_freq)
        durations = inverse_standardize_data(durations, mean_dur, std_dur)
        deltas = inverse_standardize_data(deltas, mean_del, std_del)

    #durations = convert_beats_to_ticks(durations)
    #deltas = convert_beats_to_ticks(deltas)

    frequencies = np.clip(frequencies, None, np.log10(12543.850))
    frequencies = convert_to_pitch_scale(frequencies)

    durations = np.clip(np.int_(durations), 1, None)
    deltas = np.clip(np.int_(deltas), 0, None)

    return frequencies, durations, deltas
