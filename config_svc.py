import torch

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sampling_rate": 16000,
    "hop_size": 256,
    "win_size": 1024,
    "segment_size": 16000,

    "raw_audio_path": "data/raw",
    "processed_audio_path": "data/processed",
    "feature_path": "data/features",
    "f0_path": "data/f0",

    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 2e-4,
    "model_save_path": "hifigan_model_best.pth",

    "rmvpe_model_path": "rmvpe.pt"
}
