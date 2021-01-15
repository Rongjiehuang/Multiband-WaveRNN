# Multiband-WaveRNN


Pytorch implementation of MultiBand-WaveRNN model from
[Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)
[DURATION INFORMED ATTENTION NETWORK FOR MULTIMODAL SYNTHESIS](https://arxiv.org/abs/1909.01700)


# Installation

Ensure you have:

* Python >= 3.6
* [Pytorch 1 with CUDA](https://pytorch.org/)

Then install the rest with pip:

> pip install -r requirements.txt

# How to Use

### Training your own Models
Download the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) Dataset.

Edit **hparams.py**, point **wav_path** to your dataset and run:

> python preprocess.py

or use preprocess.py --path to point directly to the dataset
___

Here's my recommendation on what order to run things:

1 - Train WaveRNN with:

> python train_wavernn.py --gta

NB: You can always just run train_wavernn.py without --gta if you're not interested in TTS.

2 - Generate Sentences with both models using:

> python gen_wavernn.py



# Chinese singing voice samples

<!--
<ruby>猜不透是哪里出了错<rt style="font-size: 15px;">cāi bú tòu shì nǎ lǐ chū le cuò</rt></ruby>
-->

<table><thead>
<tr>
<th style="text-align: center">Recording</th>
<th style="text-align: center">WaveRNN</th>
<th style="text-align: center"><b>MultiBand WaveRNN</b></th>
<th style="text-align: center">Parallel WaveGAN</th>
<th style="text-align: center">FB MelGAN</th>
<th style="text-align: center">HIFI Singer</th>
<th style="text-align: center">NewVocoder</th>
</tr></thead><tbody>
<tr>
<td style="text-align: center"><audio controls style="width: 150px;"><source src="audio
_demo/Recording_700k_steps_2_target.wav" type="audio/wav"></audio></td>

<td style="text-align: center"></td>

<td style="text-align: center"><audio controls style="width: 150px;"><source src="audio
_demo/MBWaveRNN_700k_steps_2_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>




<td style="text-align: center"></td>
<td style="text-align: center"></td>
<td style="text-align: center"></td>
<td style="text-align: center"></td>
</tr>
</tbody></table>
<br>

# References
* [DURATION INFORMED ATTENTION NETWORK FOR MULTIMODAL SYNTHESIS](https://arxiv.org/abs/1909.01700)
* [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

# Acknowlegements

* [https://github.com/fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
