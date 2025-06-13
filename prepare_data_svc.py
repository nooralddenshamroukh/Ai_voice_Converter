import os
import shutil
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from rmvpe import RMVPE
from config_svc import config

SAMPLE_RATE = config["sampling_rate"]
RAW_DIR = config["raw_audio_path"]
PROC_DIR = config["processed_audio_path"]
FEAT_DIR = config["feature_path"]
F0_DIR = config["f0_path"]

extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
model_wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").eval().to(config["device"])
f0_model = RMVPE(config["rmvpe_model_path"], is_half=False, onnx=False, device=config["device"])

for folder in [PROC_DIR, FEAT_DIR, F0_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

for fname in sorted(os.listdir(RAW_DIR)):
    if not fname.endswith(".wav"):
        continue
    name = fname.replace(".wav", "")
    wav_path = os.path.join(RAW_DIR, fname)

    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    wav = wav / max(wav.abs().max(), 1e-9)
    torchaudio.save(os.path.join(PROC_DIR, fname), wav, SAMPLE_RATE)

    inputs = extractor(wav.squeeze().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        feat = model_wavlm(**inputs.to(config["device"])).last_hidden_state.squeeze(0).cpu()

    f0 = f0_model.infer_from_audio(wav.squeeze().numpy())
    f0_tensor = torch.tensor(f0[:feat.shape[0]], dtype=torch.float32)

    torch.save(feat, os.path.join(FEAT_DIR, f"{name}.pt"))
    torch.save(f0_tensor, os.path.join(F0_DIR, f"{name}.pt"))

    print(f"✅ Done: {name}")
