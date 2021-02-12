# Multiband-WaveRNN


Pytorch implementation of MultiBand-WaveRNN model from
[Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)
[DURATION INFORMED ATTENTION NETWORK FOR MULTIMODAL SYNTHESIS](https://arxiv.org/abs/1909.01700)

# Issues
RAW mode, Unbatched generation supported.
Welcome for your contribution to implement MOL mode.

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

> python train_wavernn.py 

2 - Generate Sentences with both models using:

> python gen_wavernn.py



# Speech

## Mandarin

<table>
    <thead>
    <th style="text-align: center">Speaker</th>
    <th style="text-align: center">Recording</th>
    <th style="text-align: center">WaveRNN</th>
    <th style="text-align: center">Parallel WaveGAN</th>
    <th style="text-align: center">FB MelGAN</th>
    <th style="text-align: center">SingVocoder</th>
    </thead>
    <tbody>
        <tr>
            <th>#1</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/gt/009951.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/WaveRNN/009951.npy__740k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/pwg/009951_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/melgan/009951_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/singvocoder/009951_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#2</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/gt/009952.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/WaveRNN/009952.npy__740k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/pwg/009952_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/melgan/009952_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/singvocoder/009952_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#3</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/gt/009953.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/WaveRNN/009953.npy__740k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/pwg/009953_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/melgan/009953_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/singvocoder/009953_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#4</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/gt/009954.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/WaveRNN/009954.npy__740k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/pwg/009954_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/melgan/009954_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/singvocoder/009954_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#5</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/gt/009955.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/WaveRNN/009955.npy__740k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/pwg/009955_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/melgan/009955_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/mandarin/singvocoder/009955_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody> 
</table>


## English

<table>
    <thead>
    <th style="text-align: center">Speaker</th>
    <th style="text-align: center">Recording</th>
    <th style="text-align: center">WaveRNN</th>
    <th style="text-align: center">Parallel WaveGAN</th>
    <th style="text-align: center">FB MelGAN</th>
    <th style="text-align: center">SingVocoder</th>
    </thead>
    <tbody>
        <tr>
            <th>#1</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/gt/p225_353.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/WaveRNN/p225_353.npy__500k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/pwg/p225_353_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/melgan/p225_353_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/singvocoder/p225_353_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#2</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/gt/p226_361.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/WaveRNN/p226_361.npy__500k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/pwg/p226_361_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/melgan/p226_361_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/singvocoder/p226_361_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#3</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/gt/p227_393.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/WaveRNN/p227_393.npy__500k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/pwg/p227_393_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/melgan/p227_393_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/singvocoder/p227_393_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#4</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/gt/p229_381.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/WaveRNN/p229_381.npy__500k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/pwg/p229_381_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/melgan/p229_381_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/singvocoder/p229_381_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody>
    <tbody>
        <tr>
            <th>#5</th>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/gt/p230_407.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/WaveRNN/p230_407.npy__500k_steps_gen_NOT_BATCHED.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/pwg/p230_407_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/melgan/p230_407_gen.wav" type="audio/wav"></audio></td>
            <td style="text-align: center"><audio controls style="width: 150px;"><source src="audio_demo/speech/English/singvocoder/p230_407_gen.wav" type="audio/wav"></audio></td>
        </tr>
    </tbody> 
</table>



# References
* [DURATION INFORMED ATTENTION NETWORK FOR MULTIMODAL SYNTHESIS](https://arxiv.org/abs/1909.01700)
* [Efficient Neural Audio Synthesis](https://arxiv.org/abs/1802.08435v1)

# Acknowlegements

* [https://github.com/fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
