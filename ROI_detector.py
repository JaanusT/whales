import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import csv
import math

def process_whale_audio(audio_path, sr=4000, n_fft=4000, hop_length=400):
    
    #Processes the entire audio file at once to avoid boundary cutoffs, generates a CSV of timestamps, and exports 60-second visual chunks.
    
    filename = os.path.basename(audio_path)
    base_name = os.path.splitext(filename)[0]
    print(f"\n--- Analyzing: {filename} ---")
    
    # Load entire audio and generate Linear Spectrogram
    y, sr = librosa.load(audio_path, sr=sr)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    
    # Crop to 1500 Hz (Since sr=4000 and n_fft=4000, 1 bin = exactly 1 Hz)
    max_bin = 1500 
    magnitude_cropped = magnitude[:max_bin, :]
    
    # PCEN & Formatting
    pcen_S = librosa.pcen(magnitude_cropped, sr=sr, hop_length=hop_length)
    pcen_norm = cv2.normalize(pcen_S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    pcen_norm = np.ascontiguousarray(np.flipud(pcen_norm)) 

    #A 5x5 median blur completely erases 1x1 and 2x2 random noise spikes, but preserves true 3x3+ solid shapes (like your dashes and streaks).
    blurred_img = cv2.medianBlur(pcen_norm, 5)

    # Statistical Thresholding
    # Calculates the average background volume (mean) and only keeps sounds that are significantly louder (3.5 standard deviations above the mean).
    mean_val, std_val = cv2.meanStdDev(blurred_img)
    threshold_value = mean_val[0][0] + (3.5 * std_val[0][0])
    
    # We set a hard floor (e.g., 40 out of 255) so perfectly quiet files don't highlight dark noise
    threshold_value = max(threshold_value, 40) 
    
    _, thresh_img = cv2.threshold(blurred_img, threshold_value, 255, cv2.THRESH_BINARY)

    # A small vertical rectangle just to stitch broken streaks together, without drawing giant ghost boxes.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)) 
    closed_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)


    # Contour Detection & Extraction

    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    time_per_pixel = hop_length / sr # 0.1 seconds per pixel
    
    for contour in contours:
        x, y_box, w, h = cv2.boundingRect(contour)
        
        # Gentle Size Filter
        # Must be at least 2 pixels wide (0.2s) OR 4 pixels tall (4 Hz).
        # This guarantees Zone 4 dashes survive, while dropping the remaining tiny specks.
        if w > 1 or h > 3: 
            start_time = x * time_per_pixel
            duration = w * time_per_pixel
            end_time = start_time + duration
            
            top_hz = 1500 - y_box
            bottom_hz = 1500 - (y_box + h)

            rois.append([round(start_time, 2), round(end_time, 2), round(duration, 2), bottom_hz, top_hz, x, y_box, w, h])

    print(f"Found {len(rois)} distinct, clean acoustic elements.")

    # EXPORT: CSV & IMAGES
    
    # Create an output folder for this specific audio file
    output_dir = f"Output_{base_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Export CSV
    csv_path = os.path.join(output_dir, f"{base_name}_timestamps.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Start_Time_sec", "End_Time_sec", "Duration_sec", "Bottom_Freq_Hz", "Top_Freq_Hz"])
        for roi in rois:
            writer.writerow(roi[:5]) # Write only the human-readable data
            
            # Draw boxes on the global image array for visual export
            cv2.rectangle(pcen_norm, (roi[5], roi[6]), (roi[5] + roi[7], roi[6] + roi[8]), 255, 2)

    print(f"CSV saved to: {csv_path}")

    # Export 60-Second Image Chunks
    pixels_per_second = sr / hop_length
    chunk_width_pixels = int(60 * pixels_per_second) 
    total_width = pcen_norm.shape[1]
    
    num_chunks = math.ceil(total_width / chunk_width_pixels)
    
    for i in range(num_chunks):
        start_px = i * chunk_width_pixels
        end_px = min(start_px + chunk_width_pixels, total_width)
        
        chunk_img = pcen_norm[:, start_px:end_px]
        
        # Plotting to fit a laptop screen (16:9 aspect ratio)
        plt.figure(figsize=(16, 9), dpi=120) 
        plt.imshow(chunk_img, cmap='magma', aspect='auto', extent=[start_px * time_per_pixel, end_px * time_per_pixel, 0, 1500])
        plt.title(f"{base_name} - Minute {i+1}")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.tight_layout()
        
        img_path = os.path.join(output_dir, f"{base_name}_chunk_{i+1:03d}.png")
        plt.savefig(img_path)
        plt.close() # Close memory to prevent RAM overload
        
    print(f"Saved {num_chunks} image chunks to folder: {output_dir}\n")

def run_gui():
    root = tk.Tk()
    root.withdraw() 
    root.call('wm', 'attributes', '.', '-topmost', True)

    file_paths = filedialog.askopenfilenames(
        title="Select Whale Audio Files",
        filetypes=[("WAV Audio", "*.wav"), ("All Files", "*.*")]
    )

    if not file_paths:
        print("No files selected. Exiting.")
        return

    for audio_file in file_paths:
        process_whale_audio(audio_file)

if __name__ == "__main__":
    run_gui()