# Voice Converter: Human-to-Human Voice Transformation

## 🔊 Overview
This project demonstrates a **voice conversion system** that transforms speech from one human voice to another. The system is capable of modifying the speaker identity while preserving the content and naturalness of the speech.

The model was trained on a target voice and takes input audio from a different speaker, producing output audio that sounds as if spoken by the target individual.

⚠️ **This project is for academic and non-commercial purposes only.**

## 📌 Disclaimer
This project **does not claim ownership** of any third-party libraries or models used. In particular:

- The **RMVPE** (Robust Mel-spectrogram-based Voice Pitch Estimator) model is developed and maintained by its original authors.
- The **HiFi-GAN** (High-Fidelity Generative Adversarial Network for Speech Synthesis) model is developed and maintained by its original creators.

We use these libraries under their respective licenses solely for educational and experimental purposes in our graduation project. No part of this repository intends to infringe on the intellectual property or licensing terms of these projects. We highly encourage users to refer to the original repositories for production or commercial usage.

## ⚙️ How It Works

1. **Input Audio**: A clean voice sample (source) is provided, which contains the speech content to be converted.

2. **Pitch Extraction**:
   - RMVPE is used to extract the pitch (F0 contour) from the input audio.
   - This pitch information is essential for capturing the intonation and prosody of the original speaker.

3. **Feature Encoding**:
   - Mel-spectrograms are generated from the input speech.
   - These are processed to extract linguistic and speaker-independent features.

4. **Voice Conversion**:
   - A trained model (based on the target voice) uses these features and pitch to generate new mel-spectrograms that match the **target voice characteristics**.

5. **Waveform Generation**:
   - HiFi-GAN is used as a vocoder to convert the generated mel-spectrograms into high-fidelity waveform audio.

6. **Output**:
   - The final output is a natural-sounding audio sample that maintains the content of the input but mimics the **target speaker's** voice.


## 📄 License
This repository is released under the **MIT License**, excluding external libraries (RMVPE, HiFi-GAN, etc.), which retain their respective original licenses. Please consult their documentation for terms of use.

---

### 🔗 References

- [HiFi-GAN (Official)](https://github.com/jik876/hifi-gan)
- [RMVPE (Official)](https://github.com/yxlllc/RMVPE)

---

## 👨‍🎓 Academic Use Only

This project was developed as part of a **graduation project** and is not intended for commercial distribution or deployment. If you are a developer or researcher interested in voice synthesis, please support the original authors of the RMVPE and HiFi-GAN projects.



