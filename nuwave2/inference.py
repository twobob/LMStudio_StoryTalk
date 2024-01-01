from lightning_model import NuWave2
from omegaconf import OmegaConf as OC
import os
import argparse
import datetime
from glob import glob
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import matplotlib.pyplot as plt
from utils.stft import STFTMag
import numpy as np
from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import resample_poly

import random


def save_stft_mag(wav, fname):
    # Assuming 'wav' is your waveform tensor of shape [B, T] or [T]
    # If wav is a batch of waveforms, select the first one for visualization
    if len(wav.shape) > 1:
        wav = wav[0]

    stft_mag = STFTMag(nfft=1024, hop=256)
    # Compute the magnitude spectrum using the STFTMag class
    mag = stft_mag(wav.detach().cpu())  # Ensure wav is on CPU for numpy conversion

    # Convert the magnitude spectrum to decibels
    mag_db = rosa.amplitude_to_db(mag.numpy(), ref=np.max)

    # Check if mag_db is 2D and reshape if necessary
    if len(mag_db.shape) == 1:
        mag_db = mag_db.reshape(1, -1)  # Reshape to 2D if necessary

    # Visualization
    fig = plt.figure(figsize=(9, 3))
    plt.imshow(mag_db, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar()
    plt.xlabel('Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    fig.savefig(fname, format='png')
    plt.close()

model = None
hparams = None
noise_schedule = None

def infer(checkpoint, wav_file, sample_rate=22050, steps=8, gt=False, device='cuda'):
    global model
    global hparams
    global noise_schedule
    if model is None:
        hparams = OC.load(r'.\nuwave2\hparameter.yaml')
        os.makedirs(hparams.log.test_result_dir, exist_ok=True)

        if steps is None or steps == 8:
            steps = 8
            noise_schedule = eval(hparams.dpm.infer_schedule)
        else:
            noise_schedule = None

        model = NuWave2(hparams).to(device)
        model.eval()
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'] if 'EMA' not in checkpoint else ckpt)

    highcut = int(sample_rate) // 2
    nyq = 0.5 * hparams.audio.sampling_rate
    hi = highcut / nyq

    if gt:
        wav, _ = rosa.load(wav_file, sr=hparams.audio.sampling_rate, mono=True)
        wav /= np.max(np.abs(wav))
        wav = wav[:len(wav) - len(wav) % hparams.audio.hop_length]

        order = 8
        sos = cheby1(order, 0.05, hi, btype='lowpass', output='sos')
        wav_l = sosfiltfilt(sos, wav)

        # Downsample to the low sampling rate
        wav_l = resample_poly(wav_l, highcut * 2, hparams.audio.sampling_rate)
        # Upsample to the original sampling rate
        wav_l = resample_poly(wav_l, hparams.audio.sampling_rate, highcut * 2)

        if len(wav_l) < len(wav):
            wav_l = np.pad(wav, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
        elif len(wav_l) > len(wav):
            wav_l = wav_l[:len(wav)]
    else:
        
        wav, _ = rosa.load(wav_file, sr=int(sample_rate), mono=True)
        wav /= np.max(np.abs(wav))

        # Upsample to the original sampling rate
        wav_l = resample_poly(wav, hparams.audio.sampling_rate, int(sample_rate))
        wav_l = wav_l[:len(wav_l) - len(wav_l) % hparams.audio.hop_length]

    fft_size = hparams.audio.filter_length // 2 + 1
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[:int(hi * fft_size)] = 1

    wav = torch.from_numpy(wav).unsqueeze(0).to(device)
    wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(device)
    band = band.unsqueeze(0).to(device)

    wav_recon, wav_list = model.inference(wav_l, band, steps, noise_schedule)

    wav = torch.clamp(wav, min=-1, max=1 - torch.finfo(torch.float16).eps)
    save_stft_mag(wav, os.path.join(hparams.log.test_result_dir, f'wav.png'))
    if gt:
        swrite(os.path.join(hparams.log.test_result_dir, f'wav.wav'),
               hparams.audio.sampling_rate, wav[0].detach().cpu().numpy())
    else:
        swrite(os.path.join(hparams.log.test_result_dir, f'wav.wav'),
               sample_rate, wav[0].detach().cpu().numpy())

    wav_l = torch.clamp(wav_l, min=-1, max=1 - torch.finfo(torch.float16).eps)
    save_stft_mag(wav_l, os.path.join(hparams.log.test_result_dir, f'wav_l.png'))
    swrite(os.path.join(hparams.log.test_result_dir, f'wav_l.wav'),
           hparams.audio.sampling_rate, wav_l[0].detach().cpu().numpy())

    wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps)
    save_stft_mag(wav_recon, os.path.join(hparams.log.test_result_dir, f'result.png'))
    swrite(os.path.join(hparams.log.test_result_dir, f'result.wav'),
           hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())

    # Uncomment the following code if you want to save each step of the reconstruction
    # for i in range(len(wav_list)):
    #     wav_recon_i = torch.clamp(wav_list[i], min
    return os.path.join(hparams.log.test_result_dir, f'result.wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        required=True,
                        help="Checkpoint path")
    parser.add_argument('-i',
                        '--wav',
                        type=str,
                        default=None,
                        help="audio")
    parser.add_argument('--sr',
                        type=int,
                        required=True,
                        help="Sampling rate of input audio")
    parser.add_argument('--steps',
                        type=int,
                        required=False,
                        help="Steps for sampling")
    parser.add_argument('--gt', action="store_true",
                        required=False, help="Whether the input audio is 48 kHz ground truth audio.")
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        required=False,
                        help="Device, 'cuda' or 'cpu'")

    args = parser.parse_args()
    #torch.backends.cudnn.benchmark = False
    hparams = OC.load('hparameter.yaml')
    os.makedirs(hparams.log.test_result_dir, exist_ok=True)
    if args.steps is None or args.steps == 8:
        args.steps = 8
        noise_schedule = eval(hparams.dpm.infer_schedule)
    else:
        noise_schedule = None
    model = NuWave2(hparams).to(args.device)
    model.eval()
    stft = STFTMag()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if not('EMA' in args.checkpoint) else ckpt)

    highcut = args.sr // 2
    nyq = 0.5 * hparams.audio.sampling_rate
    hi = highcut / nyq

    if args.gt:
        wav, _ = rosa.load(args.wav, sr=hparams.audio.sampling_rate, mono=True)
        wav /= np.max(np.abs(wav))
        wav = wav[:len(wav) - len(wav) % hparams.audio.hop_length]

        order = 8
        sos = cheby1(order, 0.05, hi, btype='lowpass', output='sos')
        wav_l = sosfiltfilt(sos, wav)

        # downsample to the low sampling rate
        wav_l = resample_poly(wav_l, highcut * 2, hparams.audio.sampling_rate)
        # upsample to the original sampling rate
        wav_l = resample_poly(wav_l, hparams.audio.sampling_rate, highcut * 2)

        if len(wav_l) < len(wav):
            wav_l = np.pad(wav, (0, len(wav) - len(wav_l)), 'constant', constant_values=0)
        elif len(wav_l) > len(wav):
            wav_l = wav_l[:len(wav)]
    else:
        wav, _ = rosa.load(args.wav, sr=args.sr, mono=True)
        wav /= np.max(np.abs(wav))

        # upsample to the original sampling rate
        wav_l = resample_poly(wav, hparams.audio.sampling_rate, args.sr)
        wav_l = wav_l[:len(wav_l) - len(wav_l) % hparams.audio.hop_length]

    fft_size = hparams.audio.filter_length // 2 + 1
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[:int(hi * fft_size)] = 1

    wav = torch.from_numpy(wav).unsqueeze(0).to(args.device)
    wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(args.device)
    band = band.unsqueeze(0).to(args.device)

    wav_recon, wav_list = model.inference(wav_l, band, args.steps, noise_schedule)

    wav = torch.clamp(wav, min=-1, max=1 - torch.finfo(torch.float16).eps)
    save_stft_mag(wav, os.path.join(hparams.log.test_result_dir, f'wav.png'))
    if args.gt:
        swrite(os.path.join(hparams.log.test_result_dir, f'wav.wav'),
               hparams.audio.sampling_rate, wav[0].detach().cpu().numpy())
    else:
        swrite(os.path.join(hparams.log.test_result_dir, f'wav.wav'),
               args.sr, wav[0].detach().cpu().numpy())

    wav_l = torch.clamp(wav_l, min=-1, max=1 - torch.finfo(torch.float16).eps)
    save_stft_mag(wav_l, os.path.join(hparams.log.test_result_dir, f'wav_l.png'))
    swrite(os.path.join(hparams.log.test_result_dir, f'wav_l.wav'),
           hparams.audio.sampling_rate, wav_l[0].detach().cpu().numpy())

    wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps)
    save_stft_mag(wav_recon, os.path.join(hparams.log.test_result_dir, f'result.png'))
    swrite(os.path.join(hparams.log.test_result_dir, f'result.wav'),
           hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())

    # for i in range(len(wav_list)):
    #     wav_recon_i = torch.clamp(wav_list[i], min=-1, max=1-torch.finfo(torch.float16).eps)
    #     save_stft_mag(wav_recon_i, os.path.join(hparams.log.test_result_dir, f'result_{i}.png'))
    #     swrite(os.path.join(hparams.log.test_result_dir, f'result_{i}.wav'),
    #            hparams.audio.sampling_rate, wav_recon_i[0].detach().cpu().numpy())

