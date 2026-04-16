import numpy as np
import pandas as pd
import os

def generate_dummy_data(output_path="dummy_emg.csv", duration_seconds=60, fs=250):
    n_samples = duration_seconds * fs
    time = np.arange(n_samples) / fs
    
    # 8 Channels
    n_channels = 8
    data = np.random.normal(0, 1, size=(n_samples, n_channels))
    
    # Labels
    labels = [""] * n_samples
    
    # Generate segments
    current_sample = 0
    segment_types = ["rest", "jaws", "rest", "left", "rest", "right", "rest"]
    
    for seg_type in segment_types:
        duration = np.random.randint(2, 5) # 2-5 seconds
        n_seg_samples = duration * fs
        
        end_sample = min(current_sample + n_seg_samples, n_samples)
        
        # Modify signal based on label
        if seg_type == "jaws":
            data[current_sample:end_sample] *= 5 # High amplitude
            for i in range(current_sample, end_sample):
                labels[i] = "jaws"
        elif seg_type == "left":
            data[current_sample:end_sample, 0:4] *= 3 # Left channels active
            for i in range(current_sample, end_sample):
                labels[i] = "left"
        elif seg_type == "right":
            data[current_sample:end_sample, 4:8] *= 3 # Right channels active
            for i in range(current_sample, end_sample):
                labels[i] = "right"
        else:
             for i in range(current_sample, end_sample):
                labels[i] = "rest"
                
        current_sample = end_sample
        if current_sample >= n_samples:
            break
            
    # Create DataFrame
    df = pd.DataFrame(data, columns=[f"Channel {i+1}" for i in range(n_channels)])
    df["Timestamp"] = time
    df["label"] = labels
    
    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with shape {df.shape}")

if __name__ == "__main__":
    generate_dummy_data()
