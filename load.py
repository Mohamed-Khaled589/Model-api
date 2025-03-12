import torchaudio

def load_audio():
    read,t=torchaudio.load("file.ogg")  
    return read,t