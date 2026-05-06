import os
import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter

def process_wav_files():
    # Hide the root tkinter window so only the file dialog appears
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select multiple .wav files
    print("Opening file dialog...")
    file_paths = filedialog.askopenfilenames(
        title="Select .wav files to process",
        filetypes=[("WAV files", "*.wav")]
    )

    if not file_paths:
        print("No files selected. Exiting.")
        return

    # Configuration Variables
    trim_ms = 5000         # 5 seconds = 5000 milliseconds
    freq_min = 1          # Lower bound for high-pass filter (in Hz)
    freq_max = 1500        # Upper bound for low-pass filter (in Hz)
    
    # Calculate the new sample rate (Nyquist theorem: must be at least 2x the max frequency)
    new_sample_rate = freq_max * 2 

    for file_path in file_paths:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        
        try:
            # Load the audio file
            audio = AudioSegment.from_wav(file_path)

            # Check if the audio is actually longer than 5 seconds
            if len(audio) <= trim_ms:
                print(" -> Skipped: File is 5 seconds or shorter.")
                continue

            # 1. Slice the audio to remove the first 5000 ms
            processed_audio = audio[trim_ms:]
            print(" -> Audio trimmed.")

            # 2. Apply High-Pass Filter (removes rumble below freq_min)
            processed_audio = high_pass_filter(processed_audio, freq_min)
            
            # 3. Apply Low-Pass Filter (removes hiss/noise above freq_max)
            processed_audio = low_pass_filter(processed_audio, freq_max)
            print(f" -> Filtered frequencies outside {freq_min}Hz - {freq_max}Hz.")

            # 4. Downsample the audio to shrink the frequency container
            processed_audio = processed_audio.set_frame_rate(new_sample_rate)
            print(f" -> Downsampled to {new_sample_rate} Hz to remove empty upper frequencies.")

            # Generate the new output file path
            directory, original_filename = os.path.split(file_path)
            name, extension = os.path.splitext(original_filename)
            
            # Create "transformed" subfolder if it doesn't exist
            transformed_dir = os.path.join(directory, "transformed")
            os.makedirs(transformed_dir, exist_ok=True)
            
            output_filepath = os.path.join(transformed_dir, f"_proc_{name}_processed{extension}")

            # Export the processed audio as a new .wav file
            processed_audio.export(output_filepath, format="wav")
            print(f" -> Success! Saved as: {os.path.basename(output_filepath)}")

        except Exception as e:
            print(f" -> Error processing file: {e}")

    print("\nAll done!")

if __name__ == "__main__":
    process_wav_files()