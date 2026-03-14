import os
import re
import subprocess
import tempfile

import imageio_ffmpeg
import joblib
import numpy as np
import torch
import whisper
import whisper.audio
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app = Flask(__name__)


# Use imageio_ffmpeg to decode audio without needing a system ffmpeg install
_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

def load_audio(path, sr=16000):
    cmd = [_ffmpeg, "-nostdin", "-threads", "0", "-i", path,
           "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# Patch Whisper to use our bundled ffmpeg instead of looking for a system install
whisper.audio.load_audio = load_audio


# Detect GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")


# Load the fine-tuned RoBERTa mental health classifier
print("Loading mental health model...")
mh_tokenizer = AutoTokenizer.from_pretrained("roberta_model")
mh_model = AutoModelForSequenceClassification.from_pretrained("roberta_model").to(device)
le = joblib.load("label_encoder.pkl")
mh_model.eval()
print("Mental health model ready.")

# Load the wav2vec2 speech emotion recognition model
print("Loading speech emotion model...")
emotion_pipe = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=0 if device == "cuda" else -1
)
print("Speech emotion model ready.")

# Load Whisper for audio transcription
print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("All models loaded and ready.\n")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = clean_text(data["text"])

    if not text.strip():
        return jsonify({"error": "Text is empty after cleaning"}), 400

    inputs = mh_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    ).to(device)

    with torch.no_grad():
        logits = mh_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_idx = int(torch.argmax(logits))
    prediction = le.inverse_transform([pred_idx])[0]
    confidence = {label: round(prob, 4) for label, prob in zip(le.classes_, probs)}

    return jsonify({"prediction": prediction, "confidence": confidence})


@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Decode the uploaded audio into a numpy array
        audio_array = load_audio(tmp_path, sr=16000)

        # Transcribe speech to text using Whisper
        transcript = whisper_model.transcribe(audio_array, language="en")["text"].strip()

        # Run the transcript through the mental health classifier
        cleaned = clean_text(transcript)
        if cleaned.strip():
            inputs = mh_tokenizer(
                cleaned, return_tensors="pt", truncation=True, max_length=128
            ).to(device)
            with torch.no_grad():
                logits = mh_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            pred_idx = int(torch.argmax(logits))
            mh_pred = le.inverse_transform([pred_idx])[0]
            mh_conf = {label: round(prob, 4) for label, prob in zip(le.classes_, probs)}
        else:
            mh_pred = "Unknown"
            mh_conf = {}

        # Detect the speaker's emotion from the raw audio
        emotion_results = emotion_pipe({"array": audio_array, "sampling_rate": 16000})
        label_map = {"neu": "Neutral", "hap": "Happy", "sad": "Sad", "ang": "Angry"}
        top_emotion = label_map.get(emotion_results[0]["label"], emotion_results[0]["label"])
        emotion_conf = {
            label_map.get(r["label"], r["label"]): round(r["score"], 4)
            for r in emotion_results
        }

        return jsonify({
            "transcript": transcript,
            "mental_health": {"prediction": mh_pred, "confidence": mh_conf},
            "voice_emotion": {"prediction": top_emotion, "confidence": emotion_conf}
        })

    finally:
        os.unlink(tmp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": device.upper()})


if __name__ == "__main__":
    app.run(debug=True)
