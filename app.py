import base64
import wave
from flask import Flask, request, jsonify
from flask_cors import CORS
import torchaudio
import numpy as np
import load


import os
from torchaudio.transforms import Resample
import pronunciationTrainer

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Resampling
transform = Resample(orig_freq=44100, new_freq=16000)

# Load Trainer
trainer = pronunciationTrainer.getTrainer('hubert')

@app.route("/", methods=["GET"])
def index():
    return jsonify({"name": "First Data"})

@app.route("/upload", methods=["POST"])
def upload():
    
    # ✅ Extract file name and text from request JSON
    data = request.get_json()
    if "file_name" not in data or "real_text" not in data:
        return jsonify({"error": "Missing file_name or real_text"}), 400
    
    
    bytes=data["bytes"]

    # Decode base64 audio data
    file_path = data["file_name"]
    with open("file.ogg", "wb") as f:
        f.write(base64.b64decode(bytes))
        
    waveform, sample_rate =load.load_audio()

    # ✅ Apply any transformation if needed (e.g., resampling)
    waveform = transform(waveform)
    waveform = waveform.numpy()
    audio = np.squeeze(waveform)

    # ✅ Process audio with your ML model
    real_text = data["real_text"]
    result = trainer.processAudioForGivenText(audio, real_text)

    print("Processed result:", result)
    return jsonify(result),200





if __name__ == "__main__":
    app.run(debug=True)
