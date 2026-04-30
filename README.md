# Live Audio Analyzer & Spectrogram

A real-time Python audio digital signal processing (DSP) tool featuring a live Oscilloscope (Time Domain) and Spectrum Analyzer (Frequency Domain). 

This project serves as the real-time audio ingestion and visualization Proof of Concept (PoC) for my Computer Science Bachelor's completion project (TCC) at FC-UNESP.

By transitioning core DSP concepts (typically handled in C++ frameworks like JUCE) into a lightweight Python environment, this tool acts as the foundational data pipeline for training and deploying machine learning models to classify complex guitar chords and musical harmony.

## 🚀 Features

* **Real-Time DSP:** Captures live audio streams via `sounddevice` with extremely low latency.
* **Rolling Buffer Architecture:** Efficient memory management using NumPy ring buffers to continuously process overlapping audio frames.
* **Live Oscilloscope (Graph A):** Displays raw amplitude over time to visualize waveform structures and clipping/distortion.
* **Live Spectrum Analyzer (Graph B):** Computes the Fast Fourier Transform (FFT) on the fly, applying Hann windowing to prevent spectral leakage and converting raw magnitudes to human-readable Decibels (dB).
* **Rapid ML Prototyping:** Optimized specifically for extracting the fundamental frequencies and harmonic overtones of a guitar.

## 🛠️ Prerequisites & Installation

You will need Python 3.x installed on your machine. 

Clone the repository and install the required scientific and audio libraries:
```bash
pip install numpy matplotlib scipy sounddevice
