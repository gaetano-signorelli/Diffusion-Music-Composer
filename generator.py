import argparse
import pickle

from src.midi.notes import Notes
from src.model.model_handler import ModelHandler

from config import *

def parse_arguments():

    parser = argparse.ArgumentParser(description='Diffusion sampling generation')
    parser.add_argument('save_folder', type=str, help='Path to folder to save results in')
    parser.add_argument('--n', type=int, help='Number of songs to generate', default=8)

    args = parser.parse_args()

    return args

def read_normalization_dict():

    with open(STATISTICS_PATH, 'rb') as handle:
        normalization_dict = pickle.load(handle)

    return normalization_dict

if __name__ == '__main__':

    notes = Notes()

    normalization_dict = read_normalization_dict()

    args = parse_arguments()

    n_samples = args.n
    save_path = args.save_folder

    input_shape = (1, NOTES_LENGTH, 3)

    model_handler = ModelHandler(notes, input_shape, N_HEADS, TIME_EMBEDDING_SIZE,
                                BETA_START, BETA_END, NOISE_STEPS, normalization_dict,
                                load_model=True)
    model_handler.build_model()

    model_handler.save_samples(n_samples=n_samples, save_path=save_path)
