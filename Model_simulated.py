import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Simulate random predictions
def generate_random_predictions(num_frames=20):
    predictions = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(num_frames)]
    return np.array(predictions)  # Convert to numpy array for consistency

# Animation Function for Valence-Arousal Space
def animate_valence_arousal(predictions):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Valence-Arousal Animation")

    # Ensure sequences for animation
    valence_sequence = predictions[:, 0]
    arousal_sequence = predictions[:, 1]

    # Initial point setup
    point, = ax.plot([], [], 'ro', markersize=8)

    def init():
        point.set_data([], [])
        return point,

    def update(frame):
        valence = valence_sequence[frame]
        arousal = arousal_sequence[frame]
        point.set_data([valence], [arousal])  # Set as sequences
        return point,

    ani = animation.FuncAnimation(fig, update, frames=len(predictions), init_func=init, blit=True, interval=500)
    plt.show()

# Save the random predictions to a text file
def save_predictions_to_file(predictions, output_file="predictions_simulated.txt"):
    with open(output_file, 'w') as f:
        f.write("Time Window\tValence\tArousal\n")
        for i, (valence, arousal) in enumerate(predictions):
            f.write(f"{i}\t{valence:.4f}\t{arousal:.4f}\n")

if __name__ == "__main__":
    # Simulate 20 random time window predictions
    num_time_windows = 20
    predictions = generate_random_predictions(num_time_windows)

    # Save predictions to a text file
    output_file = "predictions_simulated.txt"
    save_predictions_to_file(predictions, output_file)
    print(f"Simulated predictions saved to {output_file}")

    # Display animated valence-arousal mapping
    animate_valence_arousal(predictions)
