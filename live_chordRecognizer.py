import sounddevice as sd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings

# Suppress librosa warnings about short audio buffers
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP PARAMETERS & BUFFERS
# ==========================================
SAMPLE_RATE = 22050
UPDATE_RATE_MS = 100       # How often the GUI updates
BUFFER_DURATION = 0.5      # Half a second of "memory" for the chord

BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
HOP_SIZE = int(SAMPLE_RATE * (UPDATE_RATE_MS / 1000.0))

audio_buffer = np.zeros(BUFFER_SIZE)

# ==========================================
# 2. BUILD HARMONIC CHORD TEMPLATES
# ==========================================
pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
templates = []
chord_labels = []

def get_harmonic_profile(root_idx, alpha=0.6):
    """
    Calculates the energy distribution of a single note and its first 5 harmonics.
    alpha controls how quickly the energy of higher overtones decays.
    """
    profile = np.zeros(12)
    # The chroma shifts for the 1st, 2nd, 3rd, 4th, and 5th harmonics
    harmonic_shifts = [0, 0, 7, 0, 4] 
    
    for n, shift in enumerate(harmonic_shifts):
        weight = alpha ** n
        profile[(root_idx + shift) % 12] += weight
        
    return profile

for i in range(12):
    # --- Major Triad (Root, Major 3rd, Perfect 5th) ---
    # We sum the harmonic profiles of all three constituent notes
    template_maj = get_harmonic_profile(i) + \
                   get_harmonic_profile((i + 4) % 12) + \
                   get_harmonic_profile((i + 7) % 12)
                   
    # Normalize the template so the highest value is 1.0
    template_maj = template_maj / np.max(template_maj)
    templates.append(template_maj)
    chord_labels.append(f"{pitch_classes[i]} Major")
    
    # --- Minor Triad (Root, Minor 3rd, Perfect 5th) ---
    template_min = get_harmonic_profile(i) + \
                   get_harmonic_profile((i + 3) % 12) + \
                   get_harmonic_profile((i + 7) % 12)
                   
    template_min = template_min / np.max(template_min)
    templates.append(template_min)
    chord_labels.append(f"{pitch_classes[i]} Minor")

template_matrix = np.array(templates)

# ==========================================
# 3. AUDIO CALLBACK FUNCTION
# ==========================================
def audio_callback(indata, frames, time, status):
    """Continuously fills the rolling buffer with new audio."""
    global audio_buffer
    if status:
        print(status)
    
    new_audio = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = new_audio

# ==========================================
# 4. GUI SETUP
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title("Live Chord Recognizer")

# Create a bar chart for the 12 chroma bins
bars = ax.bar(pitch_classes, np.zeros(12), color='darkblue', alpha=0.7)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Chroma Energy (Normalized)")
ax.set_title("Live Chroma Feature Extraction")
ax.grid(axis='y', alpha=0.3)

# Add a large text element to display the detected chord
chord_text = ax.text(0.5, 0.85, "Listening...", transform=ax.transAxes, 
                     fontsize=30, fontweight='bold', ha='center', color='darkred',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# ==========================================
# 5. LIVE PROCESSING LOOP
# ==========================================
def update_gui(frame):
    """Processes the buffer and updates the screen."""
    
    # 1. Extract Chroma
    # We use a smaller n_fft here for speed in the live loop
    chromagram = librosa.feature.chroma_stft(y=audio_buffer, sr=SAMPLE_RATE, n_fft=2048, hop_length=512)
    
    # 2. Average the frames to get one stable vector for this half-second
    chroma_vector = np.mean(chromagram, axis=1)
    
    # Normalize the vector so the highest peak is always 1.0 (helps visual scaling)
    if np.max(chroma_vector) > 0:
        chroma_vector = chroma_vector / np.max(chroma_vector)
    
    # 3. Template Matching
    # Dot product compares our live vector against all 24 templates
    scores = np.dot(template_matrix, chroma_vector)
    
    # Avoid predicting a chord if the room is perfectly silent (background noise threshold)
    if np.mean(np.abs(audio_buffer)) < 0.001:
        best_chord = "Silence"
    else:
        best_idx = np.argmax(scores)
        best_chord = chord_labels[best_idx]
    
    # 4. Update the visual elements
    for bar, height in zip(bars, chroma_vector):
        bar.set_height(height)
    chord_text.set_text(best_chord)
    
    return list(bars) + [chord_text]

print("Looking for audio devices...")
print(sd.query_devices())
print("\nStarting live chord recognition... Play a chord!")

try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                            blocksize=HOP_SIZE, callback=audio_callback)
    with stream:
        ani = FuncAnimation(fig, update_gui, interval=UPDATE_RATE_MS, blit=False)
        plt.show()
except KeyboardInterrupt:
    print("Stream stopped.")
except Exception as e:
    print(f"Error: {e}")