if __name__ == '__main__':# Example: reuse your existing OpenAI setup

    # Chat with an intelligent assistant in your terminal
    from openai import OpenAI

    from openai import OpenAI
    from datetime import datetime
    from RealtimeTTS import TextToAudioStream, SystemEngine, CoquiEngine
    import logging

    import subprocess
    import svo

    from pedalboard import Pedalboard, Chorus, Reverb, PitchShift, Limiter, Gain
    from pedalboard.io import AudioFile

    # Make a Pedalboard object, containing multiple audio plugins:
    board = Pedalboard([#PitchShift(semitones = -1.0), 
                        Reverb(room_size = 0.06, damping = 0.4, wet_level = 0.15, dry_level = 0.7, width = .9, freeze_mode = 0.0),
                        Gain(gain_db = -3.0)
                        #Limiter(threshold_db = -6.0,  release_ms = 100.0)]
                        ])

    def convert_wav_to_mp3(wav_file, fx_file, mp3_file):
        #fx( "C:\\Users\\new\\dev\tts\\"+wav_file, "C:\\Users\\new\\dev\\tts\\"+mp3_file)

        # Open an audio file for reading, just like a regular file:
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

    logging.basicConfig(level=logging.INFO)  

    engine = CoquiEngine(level=logging.INFO, voices_path=r"C:\Users\new\dev\tts\coqui_voices",voice=r"simon_master.wav",  full_sentences=False, overlap_wav_len=4096, speed=.85, thread_count=3,temperature=.75  )
    stream = TextToAudioStream(engine)

    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    system_str = "an award winning story teller. You ASSIDUOUSLY avoid CLICHE PHRASES. You detest worn-out overused tropes, instead using the phrases less travelled."

    content_str = f"write me the SECOND chapter of a story , Here is Chapter one "+'''
    The most relaxing story of all time designed to get you sleep. Begins on a cool grassed verge in a wood, the sun was beginning to wane in the face of winters steel glare ...
    '''+f"\n A 4000 word short story:  Drifting with nature"

    history = [
        {"role": "system", "content": f"You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful. {system_str}"},
        {"role": "user", "content": content_str},
    ]


    from goat_storytelling_agent.storytelling_agent import StoryAgent
    from goat_storytelling_agent.plan import Plan


    backend_uri = "http://localhost:1234/v1"
    writer = StoryAgent(backend_uri, backend="hf", form='novel')
    novel_scenes = writer.generate_story('calming mediatation narrative for a soothing spiritual retreat')

    output = novel_scenes

    #message, book_spec = writer.init_book_spec(topic='calming mediatation narrative for a soothing spiritual retreat')
    #print(book_spec)

    #messages, plan = writer.create_plot_chapters(book_spec)
    #print(Plan.plan_2_str(plan))

    #output =Plan.plan_2_str(plan)


    while output != '' :
        #completion = client.chat.completions.create(
        #    model="local-model", # this field is currently unused
        #    messages=history,
        #    temperature=0.8,
        #    stream=True,
        #    frequency_penalty=1.3
        #)

        #new_message = {"role": "assistant", "content": ""}
        
        #for chunk in completion:
        #    if chunk.choices[0].delta.content:
        #        print(chunk.choices[0].delta.content, end="", flush=True)
        #        new_message["content"] += chunk.choices[0].delta.content

        #history.append(new_message)

        #output = new_message["content"]

        print(output)

        #print(svo.svo_question(output))

        stream.feed(output)
        #stream.play_async()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stream.engine.engine_name}_output_{timestamp}.wav"
        stream.play(log_synthesized_text=True, buffer_threshold_seconds=0.5, output_wavfile=filename)

        fx_filename = f"{stream.engine.engine_name}_output_{timestamp}_fx.wav"
        mp3_filename = f"{stream.engine.engine_name}_output_{timestamp}.mp3"
        convert_wav_to_mp3(filename, fx_filename, mp3_filename)

        # Uncomment to see chat history
        # import json
        # gray_color = "\033[90m"
        # reset_color = "\033[0m"
        # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
        # print(json.dumps(history, indent=2))
        # print(f"\n{'-'*55}\n{reset_color}")

        print()
        #history.append({"role": "user", "content": input("> ")})

        history.append({"role": "user", "content": output })


    #engine = SystemEngine() # replace with your TTS engine
    #stream = TextToAudioStream(engine)

    #prefix = r"### Instruction:\n" # "[INST]" # 
    #suffix = r"\n### Response:\n" # "[/INST]" #

    # Point to the local server
    #client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    #content_str = f"{prefix} write me the SECOND chapter of a story , Here is Chapter one "+'''
    #The most relaxing story of all time designed to get you sleep. Begins on a cool grassed verge in a wood, the sun was beginning to wane in the face of winters steel glare ...
    #'''+f"\n CHAPTER TWO:  Drifting with nature {suffix}"

    #while True:

    #    completion = client.chat.completions.create(
    #    model="local-model", # this field is currently unused
    #    messages=[
    #        {"role": "system", "content": "You're an award winning story teller. You ASSIDUOUSLY avoid CLICHE PHRASES. You detest worn-out overused phrases, instead using the phrases less travelled."},
    #        {"role": "user", "content": content_str}
    #    ],

    #    temperature=0.9,
    #    )

    #    output = completion.choices[0].message.content

    #    print(output)

    #    stream.feed(output)
        #stream.play_async()
    #    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #    filename = f"{stream.engine.engine_name}_output_{timestamp}.wav"
    #    stream.play(log_synthesized_text=True, buffer_threshold_seconds=0.3, output_wavfile=filename)

        #stream.play(log_synthesized_text=True, buffer_threshold_seconds=0.3)

    #    content_str = content_str +  completion.choices[0].message.content