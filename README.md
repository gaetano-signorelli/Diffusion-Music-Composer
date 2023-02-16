# Diffusion-Music-Composer

This project is an attempt to generate music by means of a [**diffusion**](https://arxiv.org/abs/2006.11239) model, using a straightforward approach. It is based on **tensorflow** and the **MIDI** audio format.

## Methods

- The main goal of this repository is to sample new audio files based on the MIDI standard: the model has to create pieces of music by producing *pitches*, their *durations* and their time distances (*deltas*).

- As core model for learning how to correctly denoise inputs, a modified (see later) [**U-Net**](https://arxiv.org/abs/1505.04597) architecture has been adopted. Different tecniques have also been tried, but without any useful improvement, such as learning the inputs separately (and then sampling pitches and subsequently durations and deltas conditioned on them) or adopting larger networks. They can be found as test branches.

- Inputs and outputs are made up of a triple of features *pitches-durations-deltas*, with a fixed length *L*. Training inputs have been extracted from the MIDI files by the following operations:

  1. Only tracks including certain instruments have been selected: **piano** for most of the experiments;
  2. A main track has been extracted by choosing the longest one;
  3. Corrupted tracks or tracks having durations or deltas exceeding a specified threshold have been removed;
  4. Each track has been split into segments of length *L*, discarding the last one when shorter;
  5. All the three features have been normalized in the range [0,1] by dividing them for their respective maximum in the training dataset. Durations and deltas have been converted from *ticks* to *beats* before normalizing.
  
- Outputs have been reconstructed by sampling from the model, clipping in the [0,1] range and multiplying for their maximum found in the dataset.

## Novelties and advantages

This project introduces some differences compared to other works, from which it could be possible to benefit. Here is a list of the main points:

- Usually, music generation based upon the MIDI format relies on Transformers-based approaches, as notes cannot be directly represented in a continous space, thus discouraging the use of diffusion-models in favour of tokens-oriented architectures. One way to solve this problem has been tackled in [**Symbolic Music Generation with Diffusion Models**](https://arxiv.org/abs/2103.16091), where notes have been compressed into a continous latent space exploiting *Music VAE embeddings*, hence adding a nested model.
The method proposed here is simpler: each pitch is encoded by its corresponding **frequency**. This looks like a more natural way to solve the problem, also safeguarding the mathematical relations between the notes, that could allow for a better learning.

- As previously pointed out, this model, differently from many others, is able not only to produce pitches, but also their durations and distances. So, it can **determine the rythm** of the music and, moreover, it can learn to **play notes simultaneously**.

- The entire pipeline could be easily extended, by increasing the height dimension of the inputs, to include **multiple instruments** at a time.

- This U-Net has been changed so that all the convolutions are **causal** (each set of features depends only on the previous ones), as well as the self-attention modules, with the inclusion of the **spatial positional encoding**. Furthermore, the kernel size varies accordingly with the compression level and embeddings are only halved along the width dimension each time.

- Each convolution has been enhanced by introducing a [**Squeeze and Excitation** layer](https://arxiv.org/abs/1709.01507).

- The model is quite **lightweight** (around *12M* parameters) and can learn quickly.

## Results

Five different datasets have been tested to explore the differences in results: *Classical music*, *Maestro dataset*, *Final Fantasy music collection*, *Free midi dataset*, *LAKH dataset*. Links to download them can be found in *data/datasets*.

Increasing the number of samples brought sensible performance improvements, with the [**LAKH dataset**](https://colinraffel.com/projects/lmd/) scoring the best, thanks to its *178K* MIDI files. Training has been conducted using a length of *L=128* notes for *200K* steps, with a batch size of *64* (20 epochs) and a learning rate of $10^{-4}$ (first 150K steps) and $10^{-6}$ (last 50k steps). Weights can be found under the *weights* folder, and they are automatically loaded when synthesizing new songs.

Here are some samples generated by the model, converted into *mp3* format and normalized:

### Training on Maestro dataset

https://user-images.githubusercontent.com/51027023/219242747-609603ed-c3b7-4c52-9860-b8b4ca7fddc5.mp4

### Training on Final Fantasy dataset

https://user-images.githubusercontent.com/51027023/219243601-b06fa6cc-d421-412f-befc-67fb1f9820ae.mp4

https://user-images.githubusercontent.com/51027023/219243858-94721518-2f0a-4c91-9158-edbab582d064.mp4

### Training on LAKH dataset

https://user-images.githubusercontent.com/51027023/219243986-c7003efa-4e98-483b-ac16-9af8f1c3b7cb.mp4

https://user-images.githubusercontent.com/51027023/219244055-07c4b4f8-635a-4d63-8cf4-44fabfc0c93f.mp4

\
Results coming from the latest weights (LAKH dataset) are generally better, as it can be noticed. Overall, they sound very differently compared to those generated by means of other state-of-the-art techniques, which also produce samples of a higher quality, in terms of rhytm and musicality. However, some results are still quite interesting and there is room to make big improvements along this different path.

## Generate new samples

The model is capable of sampling infinitely many unique pieces of music, with a fixed length of *128* notes. In order to synthesize new songs, run the `generator.py` script:

`python generator.py "save_folder"`

There is only one optional argument:
- `--n` : set the number of songs to sample (to be chosen accordingly to GPU's memory, default=8)

Results are saved into the specified folder as *"sample i.mid"*, where *i* ranges from 1 to the requested number of images.

## Train the model

Training the model from scratch is as simple as editing the hyperparameters inside the *config* file. The preset that can be currently found is the one that has been adopted for the purpose of this work. Though, the model can be re-trained on whichever dataset.

To start a new training session (or recover from the last training epoch), run the command:

`python train.py`

Training parameters can be adjusted by accessing the file *config.py*.
