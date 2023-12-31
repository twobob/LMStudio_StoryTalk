from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
import os
import ast

import time

TEMP_PATH = r'R:\temp_segment.wav' # r'./temp_segment.wav'

GIBBERISH_FILE = "incomprehensible_segments.txt"

DEBUG_PRINTS = False

model = None

def transcribe_segment(model, file_path, start_sec, end_sec):

    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Extract the specific segment
    start_ms = start_sec * 1000
    end_ms = end_sec * 1000
    segment_audio = audio[start_ms:end_ms]

    # Export segment to a temporary file
    temp_file_path = TEMP_PATH
    segment_audio.export(temp_file_path, format='wav')

    # Transcribe the temporary segment file
    segment_result = model.transcribe(temp_file_path)
    return segment_result

def transcribe_and_identify_incomprehensible_segments(file_path, threshold=0.6):
    global model
    # Initialize Whisper model
    model_size = "large-v2"
    if model is None:
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")

    # Transcribe the audio file to get segments
    segments, _ = model.transcribe(file_path, beam_size=1)
    #print (f"{len(segments)} segments to process")

    # Initialize variables to track incomprehensible segments
    incomprehensible_segments = []

    # Process each segment
    for segment in tqdm(segments, desc="Processing segments", ncols=70):
        # Reevaluate each segment as an entire file
        segments_inside , segment_info = transcribe_segment(model, file_path, segment.start, segment.end)

        # Check if the language probability for this segment is below the threshold
        if segment_info.language_probability < threshold:
            incomprehensible_segments.append((segment.start, segment.end))

    if len(incomprehensible_segments):
        print (f"{len(incomprehensible_segments)} incomprehensible_segments")

    # Determine the directory of the input file
    dir_path = os.path.dirname(file_path)

    # Determine the output file path
    output_file_path = os.path.join(dir_path, GIBBERISH_FILE)

    # Write the list of incomprehensible segments to the output file
    with open(output_file_path, 'w') as f:
        for segment in incomprehensible_segments:
            f.write(f"{segment}\n")

    #print(f"Incomprehensible segments saved to: {output_file_path}")

    return len(incomprehensible_segments)

def find_nearest_zero_crossing(audio_segment, position, window_size=1000):
    # Ensure position is an integer
    position = int(position)

    # Get raw audio data
    samples = np.array(audio_segment.get_array_of_samples())

    # Define the search window
    start = max(0, position - window_size // 2)
    end = min(len(samples), position + window_size // 2)

    # Find zero crossing in the window
    zero_crossings = np.where(np.diff(np.sign(samples[start:end])))[0]

    if len(zero_crossings) > 0:
        return start + zero_crossings[0]
    else:
        return position

def truncate_incomprehensible_segments(file_path, segments, save_elided_as_wav=False):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    removed_audio = AudioSegment.silent(duration=0) 

    # Process each segment in reverse order to avoid messing up the timings
    for start_time, end_time in reversed(segments):
        # Convert times to milliseconds
        start_time_ms = start_time * 1000
        end_time_ms = end_time * 1000

        # Find nearest zero crossings
        start = find_nearest_zero_crossing(audio, start_time_ms)
        end = find_nearest_zero_crossing(audio, end_time_ms)

        # Collect the segment to be removed
        removed_segment = audio[start:end]

        # Add the removed segment to the removed_audio if it's not empty
        if removed_segment:
            removed_audio += removed_segment

        # Remove the segment from the original audio
        audio = audio[:start] + audio[end:]

    # Save the edited audio file
    edited_file_path = file_path.rsplit(".", 1)[0] + "_edited.wav"
    elided_file_path = file_path.rsplit(".", 1)[0] + "_elided.wav"
    audio.export(edited_file_path, format="wav")
    if save_elided_as_wav:
        removed_audio.export(elided_file_path, format="wav")

    return edited_file_path


def process_audio(file_path, threshold = 0.7, save_elided_as_wav=False):
    # start timing
    start_time = time.time()

    result = transcribe_and_identify_incomprehensible_segments(file_path, threshold=threshold)

    # report split time here
    split_time = time.time()
    if DEBUG_PRINTS:
        print("Time taken for transcribe_and_identify_incomprehensible_segments: ", split_time - start_time)

    # Determine the directory of the input file
    dir_path = os.path.dirname(file_path)

    # Determine the output file path
    output_file_path = os.path.join(dir_path, GIBBERISH_FILE)

    # Write the list of incomprehensible segments to the output file
    with open(output_file_path, 'r') as f:
        incomprehensible_segments = [ast.literal_eval(line.strip()) for line in f]

    edited_file_path = truncate_incomprehensible_segments(file_path, incomprehensible_segments, save_elided_as_wav)
    print("Edited audio saved as:", edited_file_path)

    # report split time here along with a total of run
    end_time = time.time()
    if DEBUG_PRINTS:
        if len(incomprehensible_segments):
            print("Time taken for truncate_incomprehensible_segments: ", end_time - split_time)
        print("Total time taken: ", end_time - start_time)

    return result


#file_path = r"C:\Users\new\dev\tts\20231225_002833Valkyre\test.wav"
#process_audio(file_path)