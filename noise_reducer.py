from pydub import AudioSegment
import noisereduce as nr
import numpy as np
import os

def clean_and_backup_audio(file_path, proportional_noise_decrease=0.4):
    # Generate backup file path
    base, extension = os.path.splitext(file_path)
    backup_file_path = f"{base}_pre_clean{extension}"

    # Backup the original file
    os.replace(file_path, backup_file_path)

    # Load audio file from the backup
    audio = AudioSegment.from_file(backup_file_path)

    # Convert audio to numpy array
    samples = np.array(audio.get_array_of_samples())

    # Reduce noise
    reduced_noise = nr.reduce_noise(samples, freq_mask_smooth_hz=666, prop_decrease=proportional_noise_decrease, time_constant_s=1.5, sr=audio.frame_rate)

    # Convert reduced noise signal back to audio
    reduced_audio = AudioSegment(
        reduced_noise.tobytes(), 
        frame_rate=audio.frame_rate, 
        sample_width=audio.sample_width, 
        channels=audio.channels
    )

    # Save reduced audio to the original file path
    reduced_audio.export(file_path, format=extension.lstrip('.'))