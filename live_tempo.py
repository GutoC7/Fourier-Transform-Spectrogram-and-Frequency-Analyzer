import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal.windows import hann

# 1. Setup Parameters
SAMPLE_RATE = 44100
WINDOW_SIZE = 2048
UPDATE_RATE_MS = 20
HOP_SIZE = int(SAMPLE_RATE * (UPDATE_RATE_MS / 1000.0))
TIME_HISTORY = 3.0
COLUMNS = int((TIME_HISTORY * SAMPLE_RATE) / HOP_SIZE)

window = hann(WINDOW_SIZE)
audio_buffer = np.zeros(WINDOW_SIZE)

# GUI Plotting Buffers
time_axis = np.linspace(0, TIME_HISTORY, COLUMNS)
energy_plot = np.zeros(COLUMNS)
spectral_plot = np.zeros(COLUMNS)
phase_plot = np.zeros(COLUMNS)

# State memory for derivatives (crucial for live processing!)
prev_energy = 0
prev_mag = np.zeros(WINDOW_SIZE // 2 + 1)
prev_phase1 = np.zeros(WINDOW_SIZE // 2 + 1)
prev_phase2 = np.zeros(WINDOW_SIZE // 2 + 1)

def wrap_phase(p):
    return (p + np.pi) % (2 * np.pi) - np.pi

def audio_callback(indata, frames, time, status):
    global audio_buffer, energy_plot, spectral_plot, phase_plot
    global prev_energy, prev_mag, prev_phase1, prev_phase2
    
    new_audio = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = new_audio
    
    # Compute FFT for the current frame
    windowed_data = audio_buffer * window
    fft_result = np.fft.rfft(windowed_data)
    mag = np.abs(fft_result)
    phase = np.angle(fft_result)
    
    # 1. Energy Novelty Live
    current_energy = np.sum(mag**2)
    e_nov = max(0, current_energy - prev_energy)
    prev_energy = current_energy
    
    # 2. Spectral Novelty Live (Log-magnitude difference)
    log_mag = np.log(1 + 10 * mag)
    prev_log_mag = np.log(1 + 10 * prev_mag)
    spec_diff = log_mag - prev_log_mag
    s_nov = np.sum(np.maximum(0, spec_diff)) # Half-wave rectify
    prev_mag = mag
    
    # 3. Phase Novelty Live (Second derivative)
    # phase_diff2 = current_phase - 2*prev_phase1 + prev_phase2
    p_diff1_current = wrap_phase(phase - prev_phase1)
    p_diff1_prev = wrap_phase(prev_phase1 - prev_phase2)
    p_diff2 = wrap_phase(p_diff1_current - p_diff1_prev)
    
    p_nov = np.sum(mag * np.abs(p_diff2))
    
    # Shift memory
    prev_phase2 = prev_phase1
    prev_phase1 = phase
    
    # Update Plotting Buffers
    energy_plot = np.roll(energy_plot, -1); energy_plot[-1] = e_nov
    spectral_plot = np.roll(spectral_plot, -1); spectral_plot[-1] = s_nov
    phase_plot = np.roll(phase_plot, -1); phase_plot[-1] = p_nov

# Set up UI
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
fig.canvas.manager.set_window_title("Live Novelty Analyzer")

line_e, = ax1.plot(time_axis, energy_plot, color='darkred')
ax1.set_title("Energy-Based Novelty"); ax1.set_ylim(0, 20000) # Adjust based on input gain

line_s, = ax2.plot(time_axis, spectral_plot, color='darkblue')
ax2.set_title("Spectral-Based Novelty"); ax2.set_ylim(0, 1200)   

line_p, = ax3.plot(time_axis, phase_plot, color='darkgreen')
ax3.set_title("Phase-Based Novelty (Note Onset Sensitivity)"); ax3.set_ylim(0, 2500)

for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, TIME_HISTORY)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

def update_gui(frame):
    line_e.set_ydata(energy_plot)
    line_s.set_ydata(spectral_plot)
    line_p.set_ydata(phase_plot)
    return line_e, line_s, line_p

print("Starting live novelty analysis...")
try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=HOP_SIZE, callback=audio_callback)
    with stream:
        ani = FuncAnimation(fig, update_gui, interval=UPDATE_RATE_MS, blit=True)
        plt.show()
except KeyboardInterrupt:
    pass