import torch
import numpy as np
import models as mo
import WordMetrics
import WordMatching as wm
import ModelInterfaces as mi
import AIModels
import RuleBasedModels
from string import punctuation
import time


def getTrainer(model_name: str):

    device = torch.device('cpu')

    model, decoder = mo.getASRModel(model_name)
    model = model.to(device)
    model.eval()
    asr_model = AIModels.NeuralASR(model, decoder)
    phonem_converter = RuleBasedModels.EngPhonemConverter()

    trainer = PronunciationTrainer(asr_model, phonem_converter)

    return trainer


class PronunciationTrainer:
    current_transcript: str
    current_ipa: str
    current_recorded_audio: torch.Tensor
    current_recorded_transcript: str

    current_words_pronunciation_accuracy = []
    categories_thresholds = np.array([80, 75, 55])

    sampling_rate = 16000

    def __init__(self, asr_model: mi.IASRModel, word_to_ipa_coverter: mi.ITextToPhonemModel) -> None:
        self.asr_model = asr_model
        self.ipa_converter = word_to_ipa_coverter

    def getTranscriptAndWordsLocations(self, audio_length_in_samples: int):

        audio_transcript = self.asr_model.getTranscript()
        return audio_transcript

    ##################### ASR Functions ###########################

    def processAudioForGivenText(self, recordedAudio: torch.Tensor = None, real_text=None):

        start = time.time()
        real_text = real_text.lower()
        recording_transcript, recording_ipa = self.getAudioTranscript(
            recordedAudio)

        print('Time for NN to transcript audio: ', str(time.time()-start))

        start = time.time()
        real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices = self.matchSampleAndRecordedWords(
            real_text, recording_transcript)
        print('Time for matching transcripts: ', str(time.time()-start))

        pronunciation_accuracy, current_words_pronunciation_accuracy = self.getPronunciationAccuracy(
            real_and_transcribed_words)  # _ipa

        pronunciation_categories = self.getWordsPronunciationCategory(
            current_words_pronunciation_accuracy)

        result = {
        'recording_transcript': recording_transcript,
        'real_and_transcribed_words': real_and_transcribed_words,
        'recording_ipa': recording_ipa,
        'real_and_transcribed_words_ipa': real_and_transcribed_words_ipa,
        'pronunciation_accuracy': pronunciation_accuracy,
        'pronunciation_categories':pronunciation_categories}
        return result

    def getAudioTranscript(self, recordedAudio: torch.Tensor = None):
        current_recorded_audio = recordedAudio

        # current_recorded_audio = self.preprocessAudio(
        #     current_recorded_audio)

        self.asr_model.processAudio(current_recorded_audio)

        current_recorded_transcript = self.getTranscriptAndWordsLocations(
            current_recorded_audio)
        current_recorded_transcript = current_recorded_transcript.lower()
        current_recorded_ipa = self.ipa_converter.convertToPhonem(
            current_recorded_transcript)

        return current_recorded_transcript, current_recorded_ipa

    ##################### END ASR Functions ###########################

    ##################### Evaluation Functions ###########################
    def matchSampleAndRecordedWords(self, real_text, recorded_transcript):
        words_estimated = recorded_transcript.split()

        if real_text is None:
            words_real = self.current_transcript[0].split()
        else:
            words_real = real_text.split()

        mapped_words, mapped_words_indices = wm.get_best_mapped_words(
            words_estimated, words_real)

        real_and_transcribed_words = []
        real_and_transcribed_words_ipa = []
        for word_idx in range(len(words_real)):
            if word_idx >= len(mapped_words)-1:
                mapped_words.append('-')
            real_and_transcribed_words.append(
                (words_real[word_idx], mapped_words[word_idx])
            )
            real_and_transcribed_words_ipa.append((self.ipa_converter.convertToPhonem(words_real[word_idx]),
                                                   self.ipa_converter.convertToPhonem(mapped_words[word_idx])))
        return real_and_transcribed_words, real_and_transcribed_words_ipa, mapped_words_indices

    def getPronunciationAccuracy(self, real_and_transcribed_words_ipa) -> float:
        total_mismatches = 0.
        number_of_phonemes = 0.
        current_words_pronunciation_accuracy = []
        for pair in real_and_transcribed_words_ipa:

        # problem
            real_without_punctuation = self.removePunctuation(pair[0]).lower()
            number_of_word_mismatches = WordMetrics.edit_distance_python(
                real_without_punctuation, self.removePunctuation(pair[1]).lower())
            total_mismatches += number_of_word_mismatches
            number_of_phonemes_in_word = len(real_without_punctuation)
            number_of_phonemes += number_of_phonemes_in_word

            current_words_pronunciation_accuracy.append(float(
                number_of_phonemes_in_word-number_of_word_mismatches)/number_of_phonemes_in_word*100)

        percentage_of_correct_pronunciations = (
            number_of_phonemes-total_mismatches)/number_of_phonemes*100

        return np.round(percentage_of_correct_pronunciations), current_words_pronunciation_accuracy

    def removePunctuation(self, word: str) -> str:
        return ''.join([char for char in word if char not in punctuation])

    def getWordsPronunciationCategory(self, accuracies) -> list:
        categories = []

        for accuracy in accuracies:
            categories.append(
                self.getPronunciationCategoryFromAccuracy(accuracy))

        return categories

    def getPronunciationCategoryFromAccuracy(self, accuracy) -> str:
        return str(np.argmin(abs(self.categories_thresholds-accuracy)))

    def preprocessAudio(self, audio: torch.tensor) -> torch.tensor:
        audio = audio-torch.mean(audio)
        audio = audio/torch.max(torch.abs(audio))
        return audio
