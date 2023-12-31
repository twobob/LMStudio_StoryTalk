if __name__ == '__main__':

    from datetime import datetime
    #pip install nltk
    # python -m nltk.downloader punkt
    import nltk

    #pip install tqdm
    from tqdm import tqdm
    import gibberish_extractor

    import os
    #pip install pydub
    from pydub import AudioSegment

    import re
    import subprocess

    import os
    import time

    # use pip to install torch nightly with cuda support
    # install commands here https://pytorch.org/get-started/locally/
    # example:  
    # pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
    #
    import torch

    import sys

    # Directory to be added
    path_to_add = "nuwave2"

    # Add the directory to sys.path
    if path_to_add not in sys.path:
        sys.path.append(path_to_add)
    import nuwave2
    from nuwave2 import inference

    # pip install noisereduce
    from noise_reducer import clean_and_backup_audio

    # pip install pedalboard
    # pip install TTS
    # pip install numpy and anything else that complains about being missing when you first run this

    ################################################################################################################################

    RENDER_EVERY_SENTENCE = True
    TEST_GIBBERISH = True
    TEST_EVERY_SENTENCE_FOR_GIBBERISH = True
    TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH = 5
    GIBBERISH_DETECTION_THRESHOLD = 0.85 # Lower detects less gibberrish. 1.0 would contstantly detect gibberish and cause failure  [0.3 - 0.9]
    SPEAKER_SPEED = 0.9
    UPSAMPLE = True
    NOISE_REDUCTION_PROPORTION = 0.4
    VOICE_TO_USE = "coqui_voices\\basso.wav"

    def contains_substring(main_string, substring):
        return substring in main_string  
    # Function to create the batch file if it doesn't exist
    def create_batch_file_if_not_exist(batch_file_path):
        try:
            # Check if the file exists
            with open(batch_file_path, "x") as file:
                file.write(batch_content)
                return f"Batch file 'concat.bat' created successfully."
        except FileExistsError:
            return f"Batch file 'concat.bat' already exists."
    # Run the batch file as a subprocess
    def run_batch_file(batch_file_path):
        try:
            output = subprocess.run(batch_file_path, capture_output=True, text=True, check=True)
            return f"Batch file executed successfully. Output:\n{output.stdout}"
        except subprocess.CalledProcessError as e:
            return f"An error occurred while executing the batch file: {e.output}"


    def convert_wav_to_mp3(wav_file, fx_file, mp3_file, folder='audio'):
        os.makedirs(folder, exist_ok=True)
        
    
        fx_file = os.path.join(folder, fx_file)
        mp3_file = os.path.join(folder, mp3_file)

        # Open an audio file for reading:
        with AudioFile(wav_file) as f:
        
            # Open an audio file to write to:
            with AudioFile(fx_file, 'w', f.samplerate, f.num_channels) as o:
            
                # Read one second of audio at a time, until the file is empty:
                while f.tell() < f.frames:
                    chunk = f.read(f.samplerate)
                    
                    # Run the audio through our pedalboard:
                    effected = board(chunk, f.samplerate, reset=False)
                    
                    # Write the output to our output file:
                    o.write(effected)

        subprocess.call(['ffmpeg', '-i', fx_file, '-b:a', '256k', '-ar', '44100', mp3_file])

    def save_to_file(text, filename, folder):
        os.makedirs(folder, exist_ok=True)
        out_file = os.path.join(folder, filename)
        with open(out_file, 'w') as f:
            f.write(text.__str__())

    def find_story_files_without_mp3(filename):
        global RENDER_EVERY_SENTENCE 
        global TEST_GIBBERISH 
        global TEST_EVERY_SENTENCE_FOR_GIBBERISH
        global TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH 
        global GIBBERISH_DETECTION_THRESHOLD 
        global SPEAKER_SPEED
        global UPSAMPLE 
        global NOISE_REDUCTION_PROPORTION 
        global VOICE_TO_USE 

        story_files_without_mp3 = {}

        # Walk through each directory in the current directory
        for root, dirs, files in os.walk('.'):
            # If 'story.txt' is in the files and 'story.mp3' is not
            if filename in files and 'story.mp3' not in files:
                # Add the file and its directory to the dictionary
                story_files_without_mp3[os.path.join(root, filename)] = root
            if 'narration_config.txt' in files and 'story.mp3' not in files:
                config_file_path = os.path.join(root, 'narration_config.txt')
                config = read_config_from_file(config_file_path)                

                # If the configuration file exists, use it to override the global settings

                RENDER_EVERY_SENTENCE = config.get('RENDER_EVERY_SENTENCE', RENDER_EVERY_SENTENCE) if config else True
                TEST_GIBBERISH = config.get('TEST_GIBBERISH', TEST_GIBBERISH) if config else True
                TEST_EVERY_SENTENCE_FOR_GIBBERISH = config.get('TEST_EVERY_SENTENCE_FOR_GIBBERISH', TEST_EVERY_SENTENCE_FOR_GIBBERISH) if config else True
                TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH = config.get('TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH', TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH) if config else 5
                GIBBERISH_DETECTION_THRESHOLD = config.get('GIBBERISH_DETECTION_THRESHOLD', GIBBERISH_DETECTION_THRESHOLD) if config else 0.85
                SPEAKER_SPEED = config.get('SPEAKER_SPEED', SPEAKER_SPEED) if config else 0.9
                UPSAMPLE = config.get('UPSAMPLE', UPSAMPLE) if config else True
                NOISE_REDUCTION_PROPORTION = config.get('NOISE_REDUCTION_PROPORTION', NOISE_REDUCTION_PROPORTION) if config else 0.4
                VOICE_TO_USE = config.get('VOICE_TO_USE', VOICE_TO_USE) if config else "coqui_voices\\basso.wav"

        return story_files_without_mp3

    def remove_multiple_backslashes(input_str):
        # Convert the string to ASCII tokens
        ascii_tokens = [ord(char) for char in input_str]

        backslash_char = 92
        # Process the ASCII tokens to replace multiple consecutive chr(92) with a single one
        processed_tokens = []
        prev_token = None
        for token in ascii_tokens:
            if token == backslash_char and prev_token == backslash_char:
                continue
            processed_tokens.append(token)
            prev_token = token

        # Convert the processed ASCII tokens back to a string
        return ''.join(chr(token) for token in processed_tokens)

    def replace_with_case(word, replacement, text):
        def replacer(match):
            if match.group().isupper():
                return replacement.upper()
            elif match.group().islower():
                return replacement.lower()
            elif match.group()[0].isupper():
                return replacement.capitalize()
            else:
                return replacement

        word_pattern = r'\b' + word + r'\b'
        return re.sub(word_pattern, replacer, text, flags=re.IGNORECASE)

    def tidy_up_sentence_formatting(sentence):
        for test_str in ["\\n"]:
            if contains_substring(sentence, test_str):                          
                sentence = sentence.replace(test_str, "")

        for test_str in ["\"', ","\", ", "', "]:
            if sentence.startswith(test_str):
                sentence = sentence.replace(test_str, "")

        for test_str in ["\"\""]:
            if contains_substring(sentence, test_str):                          
                sentence = sentence.replace(test_str, "\"")      

        # Replace n backslashes with one backslash     
        ## Not required, further investigation showed one chr(92) despite the UI showing multiple \\\ \\       
        # sentence = remove_multiple_backslashes(sentence)
                                        
        #engine cant say this word correctly very well.
        for test_str in ["chaotic,","chaotic ","chaotic!","chaotic?","chaotic.","chaotic\""]:
            if contains_substring(sentence, test_str):                          
                sentence = sentence.replace("chaotic", f"crazy")

        #engine cant say this word correctly very well.
        # Replacing "inhale" and "exhale" considering case sensitivity
        sentence = replace_with_case("Inhale", "Breathe in", sentence)
        sentence = replace_with_case("Exhale", "Breathe out", sentence)
        sentence = replace_with_case("inhale", "breathe in", sentence)
        sentence = replace_with_case("exhale", "breathe out", sentence)    
        sentence = replace_with_case("sinuses", "airways", sentence)
      

        #engine cant say this word correctly very well.
        for test_str in ["Genuine", "genuine,","genuine ","genuine!","genuine?","genuine.","genuine\""]:
            if contains_substring(sentence, test_str):                          
                sentence = sentence.replace("genuine", f"real") 
                sentence = sentence.replace("Genuine", "Real" )               
        
        return sentence


    def read_config_from_file(config_file_path):
        config = {}
        try:
            with open(config_file_path, 'r') as file:
                for line in file:
                    name, value = line.strip().split('=')
                    value = value.strip() 
                    if value.lower() == 'true':
                        config[name] = True
                    elif value.lower() == 'false':
                        config[name] = False
                    elif value.isdigit():
                        config[name] = int(value)
                    else:
                        try:
                            config[name] = float(value)
                        except ValueError:
                            config[name] = value
        except FileNotFoundError:
            pass
        return config

################################################################################


    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"



    from pedalboard import Pedalboard, Chorus, Reverb, PitchShift, Limiter, Gain
    from pedalboard.io import AudioFile

    from TTS.api import TTS
    print(TTS().list_models())

    '''
            Args:
                model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
                model_path (str, optional): Path to the model checkpoint. Defaults to None.
                config_path (str, optional): Path to the model config. Defaults to None.
                vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
                vocoder_config_path (str, optional): Path to the vocoder config. Defaults to None.
                progress_bar (bool, optional): Whether to pring a progress bar while downloading a model. Defaults to True.
                DEPRECATED: gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
    '''
    #C:\Users\new\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2\config.json
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", config_path="./tts_config.json", progress_bar=True).to(device)


    # Make a Pedalboard object, containing multiple audio plugins:
    board = Pedalboard([#PitchShift(semitones = -1.0), 
                        Gain(gain_db = -1.5),
                        Reverb(room_size = 0.06, damping = 0.4, wet_level = 0.15, dry_level = 0.7, width = .9, freeze_mode = 0.0),
                        Gain(gain_db = -1.5)
                        #Limiter(threshold_db = -6.0,  release_ms = 100.0)]
                        ])


    story_files = find_story_files_without_mp3('story.txt')
    
    for file, directory in story_files.items():
    # Read the configuration from the file in the current directory
        config_file_path = os.path.join(directory, 'narration_config.txt')
        config = read_config_from_file(config_file_path)

        # If the configuration file exists, use it to override the global settings
        if config:
            RENDER_EVERY_SENTENCE = config.get('RENDER_EVERY_SENTENCE', RENDER_EVERY_SENTENCE)
            TEST_GIBBERISH = config.get('TEST_GIBBERISH', TEST_GIBBERISH)
            TEST_EVERY_SENTENCE_FOR_GIBBERISH = config.get('TEST_EVERY_SENTENCE_FOR_GIBBERISH', TEST_EVERY_SENTENCE_FOR_GIBBERISH)
            TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH = config.get('TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH', TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH)
            GIBBERISH_DETECTION_THRESHOLD = config.get('GIBBERISH_DETECTION_THRESHOLD', GIBBERISH_DETECTION_THRESHOLD)
            SPEAKER_SPEED = config.get('SPEAKER_SPEED', SPEAKER_SPEED)
            UPSAMPLE = config.get('UPSAMPLE', UPSAMPLE)
            NOISE_REDUCTION_PROPORTION = config.get('NOISE_REDUCTION_PROPORTION', NOISE_REDUCTION_PROPORTION)
            VOICE_TO_USE = config.get('VOICE_TO_USE', VOICE_TO_USE)
    
    
    story_files = find_story_files_without_mp3('story.txt')

    for file, directory in story_files.items():
        print(f"File: {file}, Directory: {directory}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        prefix = os.path.basename(directory)


        filename = f"{prefix}_output_{timestamp}.wav"
        fx_filename = f"{prefix}_output_{timestamp}_fx.wav"
        mp3_filename = f"{prefix}_output_{timestamp}.mp3"

        #filename = f"{nickname}_output_{timestamp}.wav"
        #fx_filename = f"{nickname}_output_{timestamp}_fx.wav"
        #mp3_filename = f"{nickname}_output_{timestamp}.mp3"                

        if not any(f.endswith('_fx.wav') for f in os.listdir(directory)):
            with open(file=file, mode="r", encoding="utf-8") as chapter_text_file:
                discard_starts = [
                    "### Instruction:",
                    "### User",
                    "###User",
                    "Place:",
                    "Time:",
                    "### Assistant",
                    "[PROMPT]",
                    "[End scene.]",
                    "The story value",
                    "Story value charge:",
                    "\nThis scene",
                    "A/N: This scene",
                    "Story value:",
                    "\nThe End",
                    "Outcome:",
                    "Mood:",
                    "The End."
                ]

                text = chapter_text_file.read()
                sentences = nltk.sent_tokenize(text)
                filtered_sentences = [sentence for sentence in sentences if not any(sentence.startswith(s) for s in discard_starts)]

                if len(sentences) > len(filtered_sentences):
                    print(f"{ len(sentences) - len(filtered_sentences) } sentences filtered")

                # Split sentences that are longer than 250 characters
                split_sentences = []
                severed_sentences = 0
                for sentence in filtered_sentences:
                    while len(sentence) > 250:
                        # Find the last period or comma in the first 250 characters
                        match = re.search('[.,]', sentence[:250][::-1])
                        severed_sentences = severed_sentences + 1
                        if match:
                            split_point = 250 - match.start()
                            split_sentences.append(sentence[:split_point])
                            sentence = sentence[split_point:]
                        else:
                            # If no period or comma is found, split at the 250th character
                            split_sentences.append(sentence[:250])
                            sentence = sentence[250:]
                    split_sentences.append(sentence)
                
                if severed_sentences > 0:
                    print (f'Split {severed_sentences} sentences to fit the en 250 char barrier')

                chapter_text = '\n'.join(split_sentences)

                ''' def tts_to_file(
                        self,
                        text: str,
                        speaker: str = None,
                        language: str = None,
                        speaker_wav: str = None,
                        emotion: str = None,
                        speed: float = 1.0,
                        pipe_out=None,
                        file_path: str = "output.wav",
                        split_sentences: bool = True,
                        **kwargs,
                    ):
                '''

                if RENDER_EVERY_SENTENCE:
                    # Create a subfolder for the rendered sentences
                    rendered_folder = os.path.join(os.path.dirname(chapter_text_file.name), "rendered_sentences")
                    os.makedirs(rendered_folder, exist_ok=True)

                    # Render each sentence and save it to the subfolder
                    for i, sentence in enumerate(tqdm(split_sentences, desc="Processing sentence", ncols=70)):
                        sentence_filename = os.path.join(rendered_folder, f"sentence_{i}.wav")

                        if not os.path.exists(sentence_filename):

                            sentence = tidy_up_sentence_formatting(sentence)      

                            if TEST_EVERY_SENTENCE_FOR_GIBBERISH and TEST_GIBBERISH:
                                # assume gibberish to force a recreation attempt in the event of found gibberish.
                                total_gibberish = 1 
                                attempts = 0                               
                                while total_gibberish > 0 and attempts < TOTAL_ATTEMPTS_TO_MAKE_ONE_SENTENCE_WITHOUT_GIBBERSISH:
                                    #make the file
                                    tts.tts_to_file(text=sentence, 
                                                    speed=SPEAKER_SPEED, 
                                                    speaker_wav=VOICE_TO_USE, 
                                                    split_sentences=False, 
                                                    language="en", 
                                                    file_path=sentence_filename)                                           
                                    #test it for validity (confidence that it is "a known language to whisper")
                                    total_gibberish = gibberish_extractor.process_audio(sentence_filename, GIBBERISH_DETECTION_THRESHOLD)


                            else:
                                tts.tts_to_file(text=sentence, 
                                                speed=SPEAKER_SPEED, 
                                                speaker_wav=VOICE_TO_USE, 
                                                split_sentences=False, 
                                                language="it", 
                                                file_path=sentence_filename)
                                
                            if UPSAMPLE:

                                input_file = sentence_filename
                                checkpoint = "C:\\Users\\new\\dev\\tts\\nuwave2\\nuwave2_02_16_13_epoch=629.ckpt"
                                sample_rate = 24000

                                new_file = inference.infer(checkpoint=checkpoint,wav_file=sentence_filename, sample_rate=sample_rate, steps=8, gt=False, device=device)

                                #swap result to old output  
                                
                                edited_file_path = sentence_filename.rsplit(".", 1)[0] + "_edited.wav"
                                os.replace( r"C:\Users\new\dev\tts\nuwave2\test_sample\result\result.wav", edited_file_path)
                                clean_and_backup_audio(edited_file_path, NOISE_REDUCTION_PROPORTION)   
                                
                    # Concatenate all rendered sentences into a single audio file
                    concatenated_audio = AudioSegment.empty()
                    for i in range(len(split_sentences)):
                        sentence_filename = os.path.join(rendered_folder, f"sentence_{i}_edited.wav") if TEST_GIBBERISH else os.path.join(rendered_folder, f"sentence_{i}.wav")
                        try:
                            sentence_audio = AudioSegment.from_wav(sentence_filename)
                        except:
                            deletion_audio  = os.path.join(rendered_folder, f"sentence_{i}.wav")    
                            if os.path.exists(deletion_audio):
                                os.remove(deletion_audio)
                                print("seed wav deleted successfully.")
                            else:
                                print("Seed wav does not exist.")

                        concatenated_audio += sentence_audio

                    # Save the concatenated audio to the original filename
                    concatenated_audio.export(filename, format="wav")

                
                    # Optionally, remove the subfolder with the rendered sentences


                convert_wav_to_mp3(filename, fx_filename, mp3_filename, directory)

        else:
            print(f"skipped {filename}")
    print(f"generation completed")