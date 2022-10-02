# Copyright (c) 2022 Savoir-faire Linux Inc.
# This code is licensed under MIT license
import whisper
import time

class SpeechRecognizer:
    SAMPLE_RATE=16000

    def init(self, model = 'medium', task = 'translate'):
        print(f'Loading Whisper model...')
        self.model = whisper.load_model(model)
        self.audio_options = whisper.DecodingOptions(task = task)

    def process_audio(self, audio):
        duration = len(audio)/float(SpeechRecognizer.SAMPLE_RATE)
        # print(f'Processing audio block of {duration} s')
        start_time = time.time()
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        #_, probs = model.detect_language(mel)
        #print(f'Detected languages: {probs}')
        result = whisper.decode(self.model, mel, self.audio_options)
        took = time.time() - start_time
        if took > duration:
            print(f'Whisper took {took} seconds to analyse {duration} seconds!')
        if result.no_speech_prob < .5 and not result.text.startswith('Thank you for watching') and not result.text.startswith('Thanks for watching'):
            return result.language, result.text
        else:
            return None, ''
