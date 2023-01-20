import numpy as np

def convert_to_log_scale(dataframe, column):

    dataframe[column] = dataframe[column].apply(
    lambda sequence: np.log10(sequence)
    )

def convert_ms_to_s(dataframe, column):

    dataframe[column] = dataframe[column].apply(
    lambda sequence: sequence / 1000.0
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

def normalize_data(dataframe, column):

    mean, std, max = get_statistics(dataframe, column)

    dataframe[column] = dataframe[column].apply(
    lambda sequence: sequence / max
    )

    return max

def convert_to_pitch_scale(sequence):
    return 10**sequence

def convert_s_to_ms(sequence):
    return int(sequence * 1000)

def inverse_standardize_data(sequence, mean, std):
    return (sequence * std) + mean

def inverse_normalize_data(sequence, max):
    return sequence * max
