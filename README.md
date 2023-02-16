# Diffusion-Music-Composer

This project is an attempt to generate music by means of a [**diffusion**](https://arxiv.org/abs/2006.11239) model, using a straightforward approach. It is based on **tensorflow** and the **MIDI** audio format.

## Methods

- The main goal of this repository is to sample new audio files based on the MIDI standard: the model has to create pieces of music by producing *pitches*, their *durations* and their time distances (*deltas*).

- As core model for learning how to correctly denoise inputs, a modified (see later) [**U-Net**](https://arxiv.org/abs/1505.04597)] architecture has been adopted. Different tecniques have also been tried, but without any useful improvement, such as learning the inputs separately (and then sampling pitches and subsequently durations and deltas conditioned on them) or adopting larger networks. They can be found as test branches.

- Inputs and outputs are made up of a triple *pitches-durations-deltas*, with a fixed length *L*. Training inputs have been extracted from the MIDI files by the following operations:

  1. Only tracks including certain instruments have been selected: **Piano** for most of the experiments;
  2. A main track has been extracted by choosing the longest one;
  3. Corrupted tracks or tracks having durations or deltas exceeding a specified threshold have been removed;
  4. Each track has been split into segments of length *L*, discarding the last one when shorter;
  5. All the three inputs have been normalized in the range [0,1] by dividng them for their respective maximum in the training dataset.
  
- Outputs have been reconstructed by sampling from the model, clipping in the [0,1] range and multiplying for their maximum found in the dataset.
