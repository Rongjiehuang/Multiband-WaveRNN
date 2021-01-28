
# CONFIG -----------------------------------------------------------------------------------------------------------#

# Here are the input and output data paths (Note: you can override wav_path in preprocess.py)
wav_path = 'dataset/unseen_women'
data_path = 'preprocess/unseen_women'

# model ids are separate - that way you can use a new tts with an old wavernn and vice versa
# NB: expect undefined behaviour if models were trained on different DSP settings
voc_model_id = 'RAW_woman_nopad'
tts_model_id = 'tts_not_need'

# set this to True if you are only interested in WaveRNN
ignore_tts = True


# DSP --------------------------------------------------------------------------------------------------------------#

# Settings for all models
sample_rate = 24000
n_fft = 512
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 128                    # 12.5ms - in line with Tacotron 2 paper
win_length = 512                   # 50ms - same reason as above
fmin = 50
min_level_db = -120
ref_level_db = 20
bits = 9                            # bit depth of signal
mu_law = True                       # Recommended to suppress noise if using raw bits in hp.voc_mode below
peak_norm = False                   # Normalise to the peak of each wav file


# WAVERNN / VOCODER ------------------------------------------------------------------------------------------------#


# Model Hparams
voc_mode = 'RAW'                    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
voc_rnn_dims = 512
voc_fc_dims = 512
voc_compute_dims = 128
voc_res_out_dims = 128
voc_res_blocks = 5
version = 2.0
#########################  MultiBand-WaveRNN   #########################
voc_multiband = True
voc_upsample_factors = (4, 4, 2)   # multiply to be hop_length/4
#########################  MultiBand-WaveRNN   #########################

# Training
voc_batch_size = 32
voc_init_lr = 1e-4
voc_final_lr = 1e-5
voc_checkpoint_every = 20_000
voc_gen_at_checkpoint = 5           # number of samples to generate at each checkpoint
voc_total_steps = 1_000_000         # Total number of training steps
voc_test_samples = 50               # How many unseen samples to put aside for testing
voc_pad = 2                         # this will pad the input so that the resnet can 'see' wider than input length
voc_seq_len = hop_length * 5      # must be a multiple of hop_length
voc_clip_grad_norm = 4             # set to None if no gradient clipping needed
voc_pad_val = -5                    # this is the minimum of mel features

# Generating / Synthesizing
voc_gen_batched = False              # very fast (realtime+) single utterance batched generation
voc_target = 5500                 # target number of samples to be generated in each batch entry
voc_overlap = 275                   # number of samples for crossfading between batches


# TACOTRON/TTS -----------------------------------------------------------------------------------------------------#


# Model Hparams
tts_embed_dims = 256                # embedding dimension for the graphemes/phoneme inputs
tts_encoder_dims = 128
tts_decoder_dims = 256
tts_postnet_dims = 128
tts_encoder_K = 16
tts_lstm_dims = 512
tts_postnet_K = 8
tts_num_highways = 4
tts_dropout = 0.5
tts_cleaner_names = ['english_cleaners']
tts_stop_threshold = -3.4           # Value below which audio generation ends.
                                    # For example, for a range of [-4, 4], this
                                    # will terminate the sequence at the first
                                    # frame that has all values < -3.4

# Training

tts_schedule = [(7,  1e-3,  10_000,  32),   # progressive training schedule
                (5,  1e-4, 100_000,  32),   # (r, lr, step, batch_size)
                (2,  1e-4, 180_000,  16),
                (2,  1e-4, 350_000,  8)]

tts_max_mel_len = 1250              # if you have a couple of extremely long spectrograms you might want to use this
tts_bin_lengths = True              # bins the spectrogram lengths before sampling in data loader - speeds up training
tts_clip_grad_norm = 1.0            # clips the gradient norm to prevent explosion - set to None if not needed
tts_checkpoint_every = 2_000        # checkpoints the model every X steps
# TODO: tts_phoneme_prob = 0.0              # [0 <-> 1] probability for feeding model phonemes vrs graphemes


# ------------------------------------------------------------------------------------------------------------------#

