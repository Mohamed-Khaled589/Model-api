import ModelInterfaces
import torch
import numpy as np
import librosa


class NeuralASR(ModelInterfaces.IASRModel):
    audio_transcript = None

    def __init__(self, model: torch.nn.Module, decoder) -> None:
        super().__init__()
        self.model = model
        self.decoder = decoder  # Decoder from CTC-outputs to transcripts

    def getTranscript(self) -> str:
        """Get the transcripts of the process audio"""
        assert(self.audio_transcript != None,
               'Can get audio transcripts without having processed the audio')
        return self.audio_transcript

    def processAudio(self, speech):
        """Process the audio"""

        input_values = self.decoder(speech, return_tensors="pt", sampling_rate=16000).input_values
        logits = self.model(input_values).logits

        pred_ids = torch.argmax(logits, axis=-1)

        self.audio_transcript = self.decoder.decode(pred_ids[0])


class NeuralTTS(ModelInterfaces.ITextToSpeechModel):
    def __init__(self, model: torch.nn.Module, sampling_rate: int) -> None:
        super().__init__()
        self.model = model
        self.sampling_rate = sampling_rate

    def getAudioFromSentence(self, sentence: str) -> np.array:
        with torch.inference_mode():
            audio_transcript = self.model.apply_tts(texts=[sentence],
                                                    sample_rate=self.sampling_rate)[0]

        return audio_transcript