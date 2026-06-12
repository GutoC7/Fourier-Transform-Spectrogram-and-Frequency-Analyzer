import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP PARAMETERS & BUFFERS
# ==========================================
SAMPLE_RATE = 22050
UPDATE_RATE_MS = 100       
BUFFER_DURATION = 0.5      

BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
HOP_SIZE = int(SAMPLE_RATE * (UPDATE_RATE_MS / 1000.0))

audio_buffer = np.zeros(BUFFER_SIZE)

# --- HMM PARAMETERS ---
# How strongly the system resists changing chords. 
# 0.0 = Frame-by-frame (lots of flickering)
# 0.99 = Massive smoothing (might feel laggy on fast chord changes)
STAY_PROBABILITY = 0.95 
CHANGE_PROBABILITY = (1.0 - STAY_PROBABILITY) / 23.0

# Initialize our prior belief (start by assuming all 24 chords are equally likely)
current_beliefs = np.ones(24) / 24.0

# Build the 24x24 Transition Matrix
# Diagonal is STAY_PROBABILITY, everything else is CHANGE_PROBABILITY
transition_matrix = np.full((24, 24), CHANGE_PROBABILITY)
np.fill_diagonal(transition_matrix, STAY_PROBABILITY)

# ==========================================
# 2. BUILD HARMONIC CHORD TEMPLATES (24 States)
# ==========================================
pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
templates = []
chord_labels = []

def get_harmonic_profile(root_idx, alpha=0.6):
    profile = np.zeros(12)
    harmonic_shifts = [0, 0, 7, 0, 4] 
    for n, shift in enumerate(harmonic_shifts):
        weight = alpha ** n
        profile[(root_idx + shift) % 12] += weight
    return profile

for i in range(12):
    # Major Triads
    template_maj = get_harmonic_profile(i) + get_harmonic_profile((i + 4) % 12) + get_harmonic_profile((i + 7) % 12)
    templates.append(template_maj / np.max(template_maj))
    chord_labels.append(f"{pitch_classes[i]} Major")
    
for i in range(12):
    # Minor Triads (Appended after Majors, so indices 12-23 are minor)
    template_min = get_harmonic_profile(i) + get_harmonic_profile((i + 3) % 12) + get_harmonic_profile((i + 7) % 12)
    templates.append(template_min / np.max(template_min))
    chord_labels.append(f"{pitch_classes[i]} Minor")

template_matrix = np.array(templates)

# ==========================================
# 3. AUDIO CALLBACK FUNCTION
# ==========================================
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status: print(status)
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# ==========================================
# 4. GUI SETUP
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title("Live HMM Chord Recognizer")

# Top Graph: Raw Chroma
bars = ax1.bar(pitch_classes, np.zeros(12), color='darkblue', alpha=0.7)
ax1.set_ylim(0, 1.0)
ax1.set_title("Live Acoustic Evidence (Harmonic Chroma)")
ax1.grid(axis='y', alpha=0.3)

# Bottom Graph: HMM Probabilities
hmm_bars = ax2.bar(range(24), np.zeros(24), color='darkred', alpha=0.7)
ax2.set_xticks(range(24))
ax2.set_xticklabels(chord_labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylim(0, 1.0)
ax2.set_title("HMM Hidden State Probabilities")
ax2.grid(axis='y', alpha=0.3)

chord_text = ax1.text(0.5, 0.85, "Listening...", transform=ax1.transAxes, 
                     fontsize=24, fontweight='bold', ha='center', color='black',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkred', linewidth=2))

plt.tight_layout()

# ==========================================
# 5. LIVE PROCESSING LOOP (FORWARD ALGORITHM)
# ==========================================
def update_gui(frame):
    global current_beliefs
    
    # 1. Extract Acoustic Evidence
    chromagram = librosa.feature.chroma_stft(y=audio_buffer, sr=SAMPLE_RATE, n_fft=2048, hop_length=512)
    chroma_vector = np.mean(chromagram, axis=1)
    if np.max(chroma_vector) > 0: chroma_vector /= np.max(chroma_vector)
    
    # Dot product gives us the raw acoustic similarity scores
    emission_scores = np.dot(template_matrix, chroma_vector)
    
    # Softmax normalization to turn raw scores into proper emission probabilities
    emission_probs = np.exp(emission_scores * 5.0) # Multiply by 5 to sharpen the peaks
    emission_probs /= np.sum(emission_probs)
    
    # 2. Apply the HMM Forward Step!
    if np.mean(np.abs(audio_buffer)) < 0.001:
        # If silent, reset beliefs to uniform
        current_beliefs = np.ones(24) / 24.0
        best_chord = "Silence"
    else:
        # MATH: P(Current) = Emission * (Transition_Matrix * P(Previous))
        prior_prediction = np.dot(transition_matrix, current_beliefs)
        new_beliefs = emission_probs * prior_prediction
        
        # Normalize the new beliefs so they sum to 1.0
        current_beliefs = new_beliefs / np.sum(new_beliefs)
        
        best_idx = np.argmax(current_beliefs)
        best_chord = chord_labels[best_idx]
    
    # 3. Update Visuals
    for bar, height in zip(bars, chroma_vector): bar.set_height(height)
    for bar, height in zip(hmm_bars, current_beliefs): bar.set_height(height)
    chord_text.set_text(best_chord)
    
    return list(bars) + list(hmm_bars) + [chord_text]

print("Starting live HMM recognition... Play a chord!")
try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=HOP_SIZE, callback=audio_callback)
    with stream:
        ani = FuncAnimation(fig, update_gui, interval=UPDATE_RATE_MS, blit=False)
        plt.show()
except KeyboardInterrupt: pass