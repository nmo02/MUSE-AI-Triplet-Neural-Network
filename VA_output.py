import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf

# File paths
valence_file = "1820_valence.csv"
arousal_file = "1820_arousal.csv"
audio_file = "1820.wav"

# Load and clean the valence CSV
df_valence = pd.read_csv(valence_file)

# Select only numeric columns
df_valence = df_valence.select_dtypes(include=['number'])
valence_values = df_valence.mean(axis=1).to_numpy()  # Average across raters

# Load and clean the arousal CSV
df_arousal = pd.read_csv(arousal_file)

# Select only numeric columns
df_arousal = df_arousal.select_dtypes(include=['number'])
arousal_values = df_arousal.mean(axis=1).to_numpy()  # Average across raters

# Load the audio file to get duration
audio_data, sr = sf.read(audio_file)
duration = len(audio_data) / sr  # Total duration in seconds
timestamps = np.linspace(0, duration, len(valence_values))  # Adjust timestamps to match data points

# Ensure that the frame rate matches the actual time spacing between valence/arousal points
frame_interval = (duration / len(valence_values)) * 1000  # Convert to milliseconds

# Set up the valence-arousal animation
fig, ax = plt.subplots()
ax.set_xlim(-1, 1)  # Valence range
ax.set_ylim(-1, 1)  # Arousal range
ax.set_xlabel("Valence")
ax.set_ylabel("Arousal")
ax.set_title("Live Valence-Arousal Animation")

# Initialize the moving point
point, = ax.plot([], [], 'bo', markersize=10)

# Line to show the movement over time
line, = ax.plot([], [], 'r-', alpha=0.5)  # Red line with slight transparency

x_data, y_data = [], []  # To store past points for trajectory

def update(frame):
    valence = valence_values[frame]
    arousal = arousal_values[frame]
    
    x_data.append(valence)
    y_data.append(arousal)

    point.set_data([valence], [arousal])  # Ensure it is a sequence
    line.set_data(x_data, y_data)  # Update trajectory

    return point, line

ani = animation.FuncAnimation(fig, update, frames=len(valence_values), interval=frame_interval, blit=True)

# Save animation to file
ani.save("valence_arousal_animation.mp4", fps=10, extra_args=['-vcodec', 'libx264'])

print(f"Animation saved as 'valence_arousal_animationn.mp4'. Frame interval: {frame_interval:.2f} ms")
