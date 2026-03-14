# Mental Health Classifier 🧠

AI-powered mental health detection from text and voice using RoBERTa, Whisper, and wav2vec2. Built with Flask.

## Features
-  **Text Analysis** — Classifies mental health conditions from typed input using a fine-tuned RoBERTa model
-  **Voice Analysis** — Records/uploads audio, transcribes it with Whisper, and detects speech emotion using wav2vec2
-  **Real-time** — REST API with Flask, supports GPU acceleration

## Tech Stack
- **Model**: Fine-tuned RoBERTa (`roberta-base`)
- **Speech-to-Text**: OpenAI Whisper
- **Emotion Detection**: wav2vec2 (superb/wav2vec2-base-superb-er)
- **Backend**: Flask (Python)
- **Hardware**: CUDA GPU supported

## Setup
```bash
pip install -r requirements.txt
python app.py
