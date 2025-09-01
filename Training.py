import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import pandas as pd
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

def process_single_file(file_info):
    """Process a single audio file with minimal memory usage"""
    file_path, idx = file_info
    try:
        # Memory-efficient audio loading
        with sf.SoundFile(file_path) as sf_desc:
            # Determine frame count and channels
            frames = sf_desc.frames
            channels = sf_desc.channels
            
            # Calculate chunk size based on memory constraints
            chunk_size = min(10000, frames)
            
            # Initialize arrays for feature calculation
            rms_sum = 0
            zcr_sum = 0
            centroid_sum = 0
            count = 0
            
            # Process audio in small chunks
            for start in range(0, frames, chunk_size):
                end = min(start + chunk_size, frames)
                chunk = sf_desc.read(chunk_size)
                
                # Convert to mono if needed
                if channels > 1:
                    chunk = chunk.mean(axis=1)
                
                # Compute features for this chunk
                rms = np.sqrt(np.mean(chunk ** 2))
                zcr = np.sum(np.abs(np.diff(np.signbit(chunk)))) / len(chunk)
                if len(chunk) > 1:
                    magnitudes = np.abs(np.fft.rfft(chunk))
                    freqs = np.fft.rfftfreq(len(chunk), 1/sf_desc.samplerate)
                    centroid = np.sum(magnitudes * freqs) / (np.sum(magnitudes) + 1e-8)
                else:
                    centroid = 0
                
                # Accumulate
                rms_sum += rms
                zcr_sum += zcr
                centroid_sum += centroid
                count += 1
            
            # Compute final features
            features = np.array([
                rms_sum / count,
                zcr_sum / count,
                centroid_sum / count
            ], dtype=np.float32)

        # Get annotations
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        valence_path = os.path.join(valence_dir, f"{base_name}.csv")
        arousal_path = os.path.join(arousal_dir, f"{base_name}.csv")
        
        if os.path.exists(valence_path) and os.path.exists(arousal_path):
            valence = pd.read_csv(valence_path).iloc[:, 1].values.mean()
            arousal = pd.read_csv(arousal_path).iloc[:, 1].values.mean()
            
            if not (np.isnan(valence) or np.isnan(arousal)):
                return idx, features, np.array([valence, arousal], dtype=np.float32)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
    
    return None

def init_worker(v_dir, a_dir):
    """Initialize worker process with global variables"""
    global valence_dir, arousal_dir
    valence_dir = v_dir
    arousal_dir = a_dir

class DEAMDataset(Dataset):
    def __init__(self, audio_dir, valence_dir, arousal_dir, min_valid_samples=100):
        start_time = time.time()
        
        # Get list of audio files
        print("\nScanning for audio files...")
        self.audio_files = [
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if f.endswith('.wav')
        ]
        total_files = len(self.audio_files)
        print(f"Found {total_files} audio files")

        # Pre-allocate tensors
        self.features = torch.zeros((total_files, 3), dtype=torch.float32)
        self.annotations = torch.zeros((total_files, 2), dtype=torch.float32)
        self.valid_mask = torch.zeros(total_files, dtype=torch.bool)

        # Process files in parallel
        num_workers = min(8, mp.cpu_count())
        print(f"\nProcessing files using {num_workers} workers...")

        # Create work items
        work_items = [(f, i) for i, f in enumerate(self.audio_files)]

        # Process in smaller batches
        batch_size = 100
        valid_count = 0
        
        for batch_start in range(0, len(work_items), batch_size):
            batch_end = min(batch_start + batch_size, len(work_items))
            current_batch = work_items[batch_start:batch_end]
            
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker,
                initargs=(valence_dir, arousal_dir)
            ) as executor:
                futures = list(executor.map(process_single_file, current_batch))
                
                for result in futures:
                    if result is not None:
                        idx, feat, annot = result
                        self.features[idx] = torch.from_numpy(feat)
                        self.annotations[idx] = torch.from_numpy(annot)
                        self.valid_mask[idx] = True
                        valid_count += 1
                
                print(f"Processed {batch_end}/{len(work_items)} files. "
                      f"Valid samples so far: {valid_count}")

        # Create indices for valid samples
        self.valid_indices = torch.nonzero(self.valid_mask).squeeze()
        
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Dataset Summary:")
        print(f"Total files found: {total_files}")
        print(f"Valid samples: {valid_count}")
        print(f"Processing speed: {total_files / processing_time:.2f} files per second")

        if valid_count < min_valid_samples:
            raise ValueError(
                f"Insufficient valid samples. Required: {min_valid_samples}\n"
                f"Found only {valid_count} complete pairs"
            )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        valid_idx = self.valid_indices[idx]
        return self.features[valid_idx], self.annotations[valid_idx]

class ValenceArousalPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2)
        )
        
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

def train_model(audio_dir, valence_dir, arousal_dir, input_size=3, hidden_size=64):
    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"\nUsing device: {device}")
    
    # Load dataset
    dataset = DEAMDataset(audio_dir, valence_dir, arousal_dir)
    
    # Calculate optimal batch size
    batch_size = 1024  # Large batch size for GPU utilization
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Initialize model
    model = ValenceArousalPredictor(input_size, hidden_size).to(device)
    
    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    num_epochs = 20
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for features, labels in pbar:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with autocast():
                    predictions = model(features)
                    loss = criterion(predictions, labels)
                
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)
    
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'final_loss': avg_loss,
        'epoch': num_epochs
    }, "deam_valence_arousal_model.pth")
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    # Define paths for your DEAM dataset
    audio_directory = Path1
    valence_directory = Path2
    arousal_directory = Path3
    
    # Verify directories exist
    print("\nChecking directories...")
    for directory, name in [(audio_directory, "Audio"), 
                          (valence_directory, "Valence"), 
                          (arousal_directory, "Arousal")]:
        if not os.path.exists(directory):
            print(f"Error: {name} directory not found at: {directory}")
            print("Please ensure the DEAM dataset is properly organized in the expected directory structure.")
            sys.exit(1)
        else:
            print(f"✓ {name} directory found")
            print(f"Sample files in {name} directory:")
            print(os.listdir(directory)[:5])

    model = train_model(audio_directory, valence_directory, arousal_directory)