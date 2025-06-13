import os
import torch
import torchaudio
import numpy as np
import faiss
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from rmvpe import RMVPE
from voice_converter_model import HiFiGANGenerator
from config_svc import config

INPUT_PATH = "source/input.wav"
OUTPUT_DIR = "converted_wavs"
SPEAKER_DB = "speaker_faiss.index"
NAMES_PATH = "speaker_names.npy"
TRANSPOSE = 0
MAX_CHUNK_SEC = 6

def split_audio(wav, sr, max_len_sec):
    max_len = sr * max_len_sec
    return [wav[:, i:i + max_len] for i in range(0, wav.shape[1], max_len)]

def normalize_and_limit(signal, peak=0.95):
    signal = signal / max(signal.abs().max(), 1e-8)
    return signal.clamp(-peak, peak)

def noise_gate(signal, threshold=1e-4):
    signal[signal.abs() < threshold] = 0
    return signal

device = config["device"]

extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base").to(device).eval()

f0_model = RMVPE(config["rmvpe_model_path"], is_half=False, onnx=False, device=device)

model = HiFiGANGenerator(input_channels=769).to(device)
model.load_state_dict(torch.load(config["model_save_path"], map_location=device))
model.eval()

wav, sr = torchaudio.load(INPUT_PATH)
wav = wav.mean(dim=0, keepdim=True)
if sr != config["sampling_rate"]:
    wav = torchaudio.functional.resample(wav, sr, config["sampling_rate"])
sr = config["sampling_rate"]

chunks = split_audio(wav, sr, MAX_CHUNK_SEC)
print(f"🔹 Split audio into {len(chunks)} chunks")

if os.path.exists(SPEAKER_DB) and os.path.exists(NAMES_PATH):
    print("🔹 Loading FAISS database...")
    index = faiss.read_index(SPEAKER_DB)
    speaker_names = np.load(NAMES_PATH)
else:
    raise FileNotFoundError("FAISS index or speaker names not found. Run faiss_build.py first.")

outputs = []
out_lengths = []

for i, chunk in enumerate(chunks):
    print(f"🔸 Processing chunk {i + 1}/{len(chunks)}...")

    inputs = extractor(chunk.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        feat = wavlm(**inputs.to(device)).last_hidden_state.squeeze(0).cpu()

    f0 = f0_model.infer_from_audio(chunk.squeeze().numpy())
    factor = 2 ** (TRANSPOSE / 12)
    f0_shifted = torch.tensor(f0[:feat.shape[0]] * factor, dtype=torch.float32).unsqueeze(1)

    speaker_embed = feat.mean(dim=0).numpy().reshape(1, -1)
    _, indices = index.search(speaker_embed, 1)
    best_speaker = speaker_names[indices[0][0]]
    print(f"    🎯 Closest speaker: {best_speaker}")

    min_len = min(feat.shape[0], f0_shifted.shape[0])
    x = torch.cat([feat[:min_len], f0_shifted[:min_len]], dim=1).T.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x).squeeze(0).cpu()

    out = out.unsqueeze(0) if out.ndim == 1 else out
    out = normalize_and_limit(out)
    out = noise_gate(out)
    outputs.append(out)
    out_lengths.append(out.shape[-1])
    print(f"    ✅ Output shape: {out.shape}")

min_len = min(out_lengths)
outputs = [o[:, :min_len] for o in outputs]
print(f"🔁 All chunks cut to length: {min_len}")
final = torch.cat(outputs, dim=1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
filename = os.path.join(OUTPUT_DIR, "converted_final.wav")
torchaudio.save(filename, final, sr)
print(f"✅ Saved converted file at: {filename}")
