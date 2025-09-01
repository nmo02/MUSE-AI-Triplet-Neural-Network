import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import librosa

# Neural Network Definition (should match the architecture used during training)
class ValenceArousalPredictor(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValenceArousalPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, 2)  # [valence, arousal]

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load the trained model
def load_trained_model(input_size, hidden_size, model_path="deam_valence_arousal_model.pth"):
    model = ValenceArousalPredictor(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

# Extract features (tempo, tonality, spectral centroid) in windows of the audio file
def extract_features_in_windows(audio_file, window_size=5.0, overlap=2.5):
    y, sr = librosa.load(audio_file, sr=None)

    # Calculate number of frames
    hop_length = int((window_size - overlap) * sr)
    window_length = int(window_size * sr)

    features = []
    for start in range(0, len(y) - window_length + 1, hop_length):
        segment = y[start:start + window_length]

        # Tonality (Chroma-based key estimate)
        chroma = librosa.feature.chroma_cens(y=segment, sr=sr)
        key = np.argmax(np.mean(chroma, axis=1))

        # Spectral Centroid (Timbre)
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))

        # Tempo (Beat Tracking)
        tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
        tempo = float(tempo)

        # Append the feature vector
        feature_vector = np.array([key, spectral_centroid, tempo], dtype=np.float32)
        features.append(torch.tensor(feature_vector))

    return torch.stack(features)

# Predict valence and arousal for each time window
def predict_valence_arousal(model, features):
    with torch.no_grad():
        predictions = model(features).numpy()  # (num_windows, 2)
    return predictions

# Save predictions to a text file
def save_predictions_to_file(predictions, output_file="predictions_inference.txt"):
    with open(output_file, 'w') as f:
        f.write("Time Window\tValence\tArousal\n")
        for i, (valence, arousal) in enumerate(predictions):
            f.write(f"{i}\t{valence:.4f}\t{arousal:.4f}\n")
    print(f"Predictions saved to {output_file}")

# Animate the valence-arousal space
def animate_valence_arousal(predictions):
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Valence-Arousal Animation")

    # Extract valence and arousal sequences
    valence_sequence = predictions[:, 0]
    arousal_sequence = predictions[:, 1]

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

if __name__ == "__main__":
    # Paths to input audio file and trained model
    audio_file = "path_to_your_wav_file.wav"
    model_path = "deam_valence_arousal_model.pth"

    # Load the trained model
    input_size = 3  # Based on the 3 extracted features: [key, spectral_centroid, tempo]
    hidden_size = 32  # Must match what you used during training
    model = load_trained_model(input_size, hidden_size, model_path)

    # Extract features from the audio file in time windows
    features = extract_features_in_windows(audio_file)

    # Predict valence and arousal
    predictions = predict_valence_arousal(model, features)

    # Save predictions to a text file
    save_predictions_to_file(predictions)

    # Display animated valence-arousal mapping
    animate_valence_arousal(predictions)
