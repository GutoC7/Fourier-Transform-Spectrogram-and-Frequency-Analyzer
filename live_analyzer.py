import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal.windows import hann

SAMPLE_RATE = 44100
WINDOW_SIZE = 4096       # size of the FFT window
UPDATE_RATE_MS = 30      # GUI update rate in ms

# how many samples arrive per GUI update
HOP_SIZE = int(SAMPLE_RATE * (UPDATE_RATE_MS / 1000.0))

# ring buffer again
audio_buffer = np.zeros(WINDOW_SIZE)
window = hann(WINDOW_SIZE)

def audio_callback(indata, frames, time, status):
    """Grabs new audio from the interface and pushes it into the ring buffer."""
    global audio_buffer
    if status:
        print(status)
    
    # shift old audio out
    new_audio = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = new_audio

# set up the two graphs vertically stacked
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title("Real-Time Audio Analyzer")

# graph A 
t_axis = np.linspace(0, WINDOW_SIZE / SAMPLE_RATE, WINDOW_SIZE, endpoint=False)
line_t, = ax1.plot(t_axis, audio_buffer, color='darkblue', lw=1)

ax1.set_title("(A) Amplitude over time (Oscilloscope)")
ax1.set_xlabel("Time (Seconds)")
ax1.set_ylabel("Amplitude")
ax1.set_xlim(0, WINDOW_SIZE / SAMPLE_RATE)
ax1.set_ylim(-0.5, 0.5) # adjust these limits if your audio is too molested 
ax1.grid(True, alpha=0.3)

# graph B 
f_axis = np.fft.rfftfreq(WINDOW_SIZE, 1 / SAMPLE_RATE)
line_f, = ax2.plot(f_axis, np.zeros(len(f_axis)), color='darkred', lw=1.5)

ax2.set_title("(B) Magnitude over frequency (Spectrum Analyzer)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_xlim(0, 1500)   # focus on the guitar frequency range
ax2.set_ylim(0, 0.05)   # adjust this if the peaks shoot off the top of your ass
ax2.grid(True, alpha=0.3)

plt.tight_layout()

def update_gui(frame):
    """Calculates the FFT and updates the lines on the screen."""
    
    # update the Time Domain line
    line_t.set_ydata(audio_buffer)
    
    # calculate the FFT for the Frequency Domain
    # apply the Hann window to prevent spectral leakage, then calculate magnitude
    windowed_data = audio_buffer * window
    fft_result = np.abs(np.fft.rfft(windowed_data)) 
    
    # normalize the magnitude so it stays consistent regardless of window size
    fft_mag = (fft_result / WINDOW_SIZE) * 2 
    
    # update the Frequency Domain line
    line_f.set_ydata(fft_mag)
    
    return line_t, line_f

print("Starting live audio stream... Play your guitar!")
try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                            blocksize=HOP_SIZE, callback=audio_callback)
    with stream:
        # start the GUI loop using blit=true for faster rendering
        ani = FuncAnimation(fig, update_gui, interval=UPDATE_RATE_MS, blit=True)
        plt.show()
except KeyboardInterrupt:
    print("Stream stopped.")
except Exception as e:
    print(f"Error: {e}")