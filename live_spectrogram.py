import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal.windows import hann

SAMPLE_RATE = 44100
WINDOW_SIZE = 4096       # resolution
TIME_HISTORY = 3.0       
UPDATE_RATE_MS = 30      # GUI update rate in ms

# calculate derived parameters
HOP_SIZE = int(SAMPLE_RATE * (UPDATE_RATE_MS / 1000.0))
COLUMNS = int((TIME_HISTORY * SAMPLE_RATE) / HOP_SIZE)

# pre-calculate the hann window to prevent spectral leakage
window = hann(WINDOW_SIZE)

# 2D array to hold our scrolling spectrogram data
# rows = frequency bins, columns = time slices
spectrogram_data = np.zeros((WINDOW_SIZE // 2 + 1, COLUMNS))

# ring buffer
audio_buffer = np.zeros(WINDOW_SIZE)

def audio_callback(indata, frames, time, status):
    """
    This function is called by the OS every time a new chunk of audio arrives 
    from your guitar interface.
    """
    global audio_buffer, spectrogram_data
    
    if status:
        print(status)
    
    # grab the new audio chunk 
    new_audio = indata[:, 0]
    
    # shift the old audio out
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = new_audio
    
    # Hann window and calculate FFT
    # rfft bc the audio signal is real 
    windowed_data = audio_buffer * window
    fft_result = np.abs(np.fft.rfft(windowed_data))
    
    # convert to Db 
    fft_db = 20 * np.log10(fft_result + 1e-10) # 1e-10 prevents log(0) errors
    
    # shift the 2D spectrogram matrix left by one column
    spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
    
    # insert the new FFT slice into the right-most column
    spectrogram_data[:, -1] = fft_db

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Live Guitar Spectrogram")
ax.set_xlabel("Time (Rolling)")
ax.set_ylabel("Frequency (Hz)")

# calculate the actual frequency bins for the Y-axis
freqs = np.fft.rfftfreq(WINDOW_SIZE, 1 / SAMPLE_RATE)
MAX_FREQ_DISPLAY = 1500 # cap it at 1.5kHz to focus on guitar

img = ax.imshow(spectrogram_data, aspect='auto', cmap='magma', 
                origin='lower', extent=[0, TIME_HISTORY, 0, SAMPLE_RATE/2],
                vmin=-80, vmax=0) # Adjust vmin/vmax if it's too dark or bright

ax.set_ylim(0, MAX_FREQ_DISPLAY)
fig.colorbar(img, ax=ax, label='Magnitude (dB)')

def update_gui(frame):
    """Called periodically by FuncAnimation to redraw the screen."""
    img.set_array(spectrogram_data)
    return [img]

print("Looking for audio devices...")
print(sd.query_devices())

print("\nStarting live audio stream... Play your guitar!")
try:
    # Open the audio stream
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, # change the input if the wrong device is selected automatically: device=ID
                            blocksize=HOP_SIZE, callback=audio_callback)
    with stream:
        # Start the GUI loop
        ani = FuncAnimation(fig, update_gui, interval=UPDATE_RATE_MS, blit=True)
        plt.show()
except KeyboardInterrupt:
    print("Stream stopped.")
except Exception as e:
    print(f"Error: {e}")
