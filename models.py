import torch
import torch.nn as nn
import pickle
from transformers import Wav2Vec2Processor, HubertForCTC


def getASRModel(model_name: str) -> nn.Module:

    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    return (model, processor)


def getTTSModel(language: str) -> nn.Module:
    speaker = 'lj_16khz'  # 16 kHz
    model = torch.hub.load(repo_or_dir='snakers4/silero-models',
                           model='silero_tts',
                           language=language,
                           speaker=speaker)

    return model