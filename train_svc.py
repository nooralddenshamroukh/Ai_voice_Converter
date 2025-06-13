import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from config_svc import config
from voice_converter_model import HiFiGANGenerator
from discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator

SEGMENT_SIZE = config["segment_size"]
HOP = config["hop_size"]

class VoiceDataset(Dataset):
    def __init__(self):
        self.names = [f.replace(".pt", "") for f in os.listdir(config["feature_path"]) if f.endswith(".pt")]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        feat = torch.load(os.path.join(config["feature_path"], name + ".pt"))
        f0 = torch.load(os.path.join(config["f0_path"], name + ".pt")).unsqueeze(1)
        x = torch.cat([feat[:f0.size(0)], f0], dim=1).T

        n_frames = x.size(1)
        seg_frames = SEGMENT_SIZE // HOP
        if n_frames > seg_frames:
            start = torch.randint(0, n_frames - seg_frames + 1, (1,)).item()
            x = x[:, start:start + seg_frames]
        else:
            x = F.pad(x, (0, seg_frames - n_frames))

        wav, _ = torchaudio.load(os.path.join(config["processed_audio_path"], name + ".wav"))
        wav = wav.mean(0, keepdim=True)
        if wav.size(1) > SEGMENT_SIZE:
            start = torch.randint(0, wav.size(1) - SEGMENT_SIZE + 1, (1,)).item()
            wav = wav[:, start:start + SEGMENT_SIZE]
        else:
            wav = F.pad(wav, (0, SEGMENT_SIZE - wav.size(1)))

        return x, wav

def multi_stft_loss(y_hat, y):
    losses = []
    for n_fft, hop in [(256, 64), (512, 128), (1024, 256)]:
        if y_hat.size(-1) < n_fft:
            continue
        s1 = torch.stft(y_hat.squeeze(1), n_fft=n_fft, hop_length=hop, return_complex=True)
        s2 = torch.stft(y.squeeze(1), n_fft=n_fft, hop_length=hop, return_complex=True)
        losses.append(F.l1_loss(torch.abs(s1), torch.abs(s2)))
    return sum(losses)

def train():
    device = config["device"]
    G = HiFiGANGenerator(input_channels=769).to(device)
    MPD = MultiPeriodDiscriminator().to(device)
    MSD = MultiScaleDiscriminator().to(device)

    g_opt = torch.optim.AdamW(G.parameters(), lr=config["learning_rate"])
    d_opt = torch.optim.AdamW(list(MPD.parameters()) + list(MSD.parameters()), lr=config["learning_rate"] * 0.5)

    dataset = VoiceDataset()
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    scaler = GradScaler()

    best_loss = float("inf")
    for epoch in range(config["epochs"]):
        total_g, total_d = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with autocast():
                y_hat = G(x)
                L = min(y_hat.size(-1), y.size(-1))
                y_hat, y = y_hat[..., :L], y[..., :L]

                loss_l1 = F.l1_loss(y_hat, y)
                loss_stft = multi_stft_loss(y_hat, y)

                fm_loss, adv_loss = 0, 0
                for D in [MPD, MSD]:
                    real_feats = D(y)
                    fake_feats = D(y_hat)
                    for rf, ff in zip(real_feats, fake_feats):
                        for r_fmap, f_fmap in zip(rf[1], ff[1]):
                            fm_loss += F.l1_loss(f_fmap, r_fmap.detach())
                        adv_loss += F.mse_loss(ff[0], torch.ones_like(ff[0]))

                g_loss = 0.6 * loss_l1 + 0.4 * loss_stft + 2.0 * fm_loss + 0.1 * adv_loss

            g_opt.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(g_opt)
            scaler.update()
            total_g += g_loss.item()

            with autocast():
                d_loss = 0
                for D in [MPD, MSD]:
                    real_feats = D(y.detach())
                    fake_feats = D(y_hat.detach())
                    for rf, ff in zip(real_feats, fake_feats):
                        d_loss += F.mse_loss(ff[0], torch.zeros_like(ff[0])) + F.mse_loss(rf[0], torch.ones_like(rf[0]))
                d_loss /= (2 * len(MPD.discriminators) + 2 * len(MSD.discriminators))

            d_opt.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.step(d_opt)
            scaler.update()
            total_d += d_loss.item()

        print(f"Epoch {epoch} | G_loss: {total_g/len(loader):.4f} | D_loss: {total_d/len(loader):.4f}")

        if total_g/len(loader) < best_loss:
            best_loss = total_g/len(loader)
            torch.save(G.state_dict(), config["model_save_path"])
            print("✅ Saved best G checkpoint")

if __name__ == "__main__":
    train()
