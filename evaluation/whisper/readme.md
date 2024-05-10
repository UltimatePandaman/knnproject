### What is this?
This subdirectory is used for ASR (automatic speech recognition) using Whisper v3 large from OpenAI.
SRC: https://huggingface.co/openai/whisper-large-v3

### Installation
```bash
pip install git+https://github.com/openai/whisper.git 
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate tqdm 
sudo apt install ffmpeg
```

### Usage
Place your waveforms into `data` and change the paths settings in ``transcribe.py``. In particular you want to change the `speakers`.
Run ``python transcribe.py``
