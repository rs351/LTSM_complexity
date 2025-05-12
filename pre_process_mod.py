import os
import numpy as np
import pandas as pd
from scipy.fft import fft
from pydub import AudioSegment
import wfdb
import mne
import sys

# Function to perform FFT
def perform_fft(signal, Fs):
    N = len(signal)
    fft_result = fft(signal)
    magnitude = np.abs(fft_result / N)
    magnitude = magnitude[:N // 2 + 1]
    magnitude[1:-1] *= 2
    frequencies = np.linspace(0, Fs / 2, N // 2 + 1)
    return frequencies, magnitude

# Function to load audio file
def load_audio(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp3', '.wav', '.flac', '.m4a']:
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        samples = samples.astype(np.float32) / 2**15
        return samples, audio.frame_rate
    else:
        raise ValueError("Unsupported audio format")
        
# Function to load .dat files
def load_wfdb_record(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    record = wfdb.rdsamp(file_name)
    signal = record[0][:, 1]  # Assuming we use the second signal channel to use the filtered data
    Fs = record[1]['fs'] # Collecting the sampling frequency from the data
    
    return signal, Fs

# Function to load .edf files 
def load_edf_record(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Get the data as a numpy array and channel names
    data = raw.get_data()
    channels = raw.info['ch_names']

    signal = data[0, :] # Taking the signals from the first channel
    Fs = 256 # Sampling Frequency collected from Metadata

    return signal, Fs

# Main script
if __name__ == "__main__":
    disc_n = 40
    tgt_N_pts = 1.4e4
    tgt_N_oscs = 1.4e3

    # Ensure a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python pre-process.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    print(f"Processing file: {file_path}")

    if ext in ['.xlsx', '.csv']:
        if ext == '.xlsx':
            data = pd.read_excel(file_path)
        else:
            data = pd.read_csv(file_path)
        time_series = pd.to_numeric(data.iloc[:, 4], errors='coerce').values
        T = 60 * 60  # Sampling period in seconds
        Fs = 1 / T  # Sampling frequency in Hz
        is_audio = False
    elif ext in ['.mp3', '.wav', '.flac', '.m4a']:
        try:
            time_series, Fs = load_audio(file_path)
            T = 1 / Fs  # Sampling period in seconds
            is_audio = True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            sys.exit(1)
    elif ext == '.dat':
        try:
            time_series, Fs = load_wfdb_record(file_path)
            T = 1 / Fs  # Sampling period in seconds
            is_audio = False
        except Exception as e:
            print(f"Error loading wfdb record: {e}")
            sys.exit(1)

    elif ext == '.edf':
        try:
            time_series, Fs = load_edf_record(file_path)
            T = 1 / Fs  # Sampling period in seconds
            is_audio = False
        except Exception as e:
            print(f"Error loading edf record: {e}")
            sys.exit(1)
    else:
        print("Unsupported file format")
        sys.exit(1)

    # Remove NaNs
    time_series = time_series[~np.isnan(time_series)]
    if len(time_series) == 0:
        print("Time series is empty after removing NaNs")
        sys.exit(1)
    
    time_series_n = time_series - np.mean(time_series)
    time_series_n = time_series_n / np.max(np.abs(time_series_n))

    # Perform Fourier Transform
    freq, mag_s = perform_fft(time_series_n, Fs)

    # Find maximum amplitude frequency
    freq_max_hz = freq[np.argmax(mag_s)]
    print(f"Maximum amplitude frequency: {freq_max_hz} Hz")

    # Display maximum amplitude frequency as a time period in secs, mins, hrs
    T_max_seconds = 1 / freq_max_hz
    T_max_mins = T_max_seconds / 60
    T_max_hrs = T_max_seconds / 3600
    print(f"Period: {T_max_seconds} seconds, {T_max_mins} minutes, {T_max_hrs} hours")

    # Display number of max amp oscillations in time series
    tot_oscs = len(time_series_n) * T * freq_max_hz
    print(f"Total oscillations in the time series: {tot_oscs}")

    # Truncate and resample the time series
    start_pt = 0.5
    window = round(len(time_series_n) * tgt_N_oscs / tot_oscs)
    gap = len(time_series_n) - window

    time_series_trunc = time_series_n[int(start_pt * gap):int(start_pt * gap) + window]
    downsample_factor = round(len(time_series_trunc) / tgt_N_pts)
    time_series_trunc_resampled = time_series_trunc[::downsample_factor]
    time_series_trunc_resampled = (time_series_trunc_resampled - np.min(time_series_trunc_resampled)) / (np.max(time_series_trunc_resampled) - np.min(time_series_trunc_resampled))
    time_series_trunc_resampled_disc = np.round(time_series_trunc_resampled * disc_n) / disc_n

    # Save the array to a CSV file in the same directory as the input file
    output_file_name = os.path.splitext(os.path.basename(file_path))[0] + '_out_for_analysis.csv'
    output_file_path = os.path.join(os.path.dirname(file_path), output_file_name)
    np.savetxt(output_file_path, time_series_trunc_resampled_disc, delimiter=',', fmt='%.6f')
    print(f"Output file saved as: {output_file_path}")
