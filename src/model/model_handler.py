import os
import re
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input

from src.model.diffusion import Diffusion
from src.preprocessing.numerical_processing import sample_to_midi_values
from src.midi.midi_converter import MidiDataBuilder

from config import *

class ModelHandler:

    def __init__(self, notes, input_shape, n_heads, time_embedding_size,
                beta_start, beta_end, noise_steps, normalization_dict,
                load_model=None, weights_path=None, verbose=True):

        self.notes = notes
        self.input_shape = input_shape
        self.n_heads = n_heads
        self.time_embedding_size = time_embedding_size
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.load_model = load_model
        self.weights_path = weights_path
        self.verbose = verbose

        self.max_freq = normalization_dict.get("max_freq")
        self.min_freq = normalization_dict.get("min_freq")
        self.max_dur = normalization_dict.get("max_dur")
        self.min_dur = normalization_dict.get("min_dur")
        self.max_del = normalization_dict.get("max_del")
        self.min_del = normalization_dict.get("min_del")

        self.model = None

        self.optimizer = None

        self.current_step = 0

        if self.load_model is None:
            self.load_model = LOAD_MODEL

        if self.weights_path is None:
            self.weights_path = WEIGHTS_PATH

    def build_model(self):

        if self.model is None:

            self.model = Diffusion(self.input_shape, self.n_heads, self.time_embedding_size,
                                    self.beta_start, self.beta_end, self.noise_steps)

            self.initialize_model()

            if self.verbose:
                self.model.unet_model.summary()

    def initialize_model(self):

        pieces_inputs = Input(shape=self.input_shape)
        times_inputs = Input(shape=(1,))
        inputs = [pieces_inputs, times_inputs]

        self.model(inputs)

        if self.load_model:
            last_weights_path = self.get_weights()

            if last_weights_path is not None:

                self.load_weights(last_weights_path)
                if self.verbose:
                    print("Weights loaded")
                    print("Restored backup from step {}".format(self.current_step))

            elif self.verbose:
                print("WARNING: Weights not found: initializing model with random weights")
                print("Ignore this warning if this is the first training or a test")

        self.optimizer = Adam(learning_rate=LEARNING_RATE)

        self.model.compile(self.optimizer, run_eagerly=RUN_EAGERLY)

    def get_weights(self):

        last_weights = None

        pattern_unet_weights = re.compile("unet_\d+.h5")

        weights_files = os.listdir(self.weights_path)
        weights_files.sort(reverse=True)

        if weights_files is not None:
            for file in weights_files:

                if pattern_unet_weights.match(file) and last_weights is None:
                    last_weights = os.path.join(self.weights_path, file)
                    self.current_step = int(last_weights[-9:-3])
                    break

        return last_weights

    def save_weights(self):

        #weights = self.model.get_network_weights()

        current_step = str(self.current_step).zfill(6)
        save_path = UNET_WEIGHTS_PATH.format(current_step)

        #np.save(save_path, weights)

        self.model.save_weights(save_path)

        if self.verbose:
            print()
            print("Weights saved")

    def load_weights(self, last_weights_path):

        #last_weights = np.load(last_weights_path, allow_pickle=True)

        #self.model.set_network_weights(last_weights)

        self.model.load_weights(last_weights_path)

    def update_current_step(self):

        self.current_step += 1

    def save_samples(self, n_samples=None, save_path=None):

        if self.verbose:
            print("\nGenerating samples...")

        if n_samples is None:
            n_samples = N_SAMPLES

        samples = self.model.sample(n_samples)

        samples = samples.numpy()

        for i, sample in enumerate(samples):

            frequencies, durations, deltas = sample_to_midi_values(
            sample, self.max_freq, self.min_freq, self.max_dur, self.min_dur,
            self.max_del, self.min_del
            )

            midi_builder = MidiDataBuilder(frequencies, durations, deltas, self.notes)

            if save_path is None:
                file_path = SAMPLES_PATH.format(self.current_step, i+1)
            else:
                file_path = os.path.join(save_path, "sample_n_{}.mid").format(i+1)

            midi_builder.build_and_save(file_path)

        if self.verbose:
            print("Samples saved")
