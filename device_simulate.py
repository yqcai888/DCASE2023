import os
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


audio_path = 'original_audio_by_device_A.wav'
impulse_path = 'impulse_response_by_device_B.wav'


def plot_waveform(signal, sr):
    # Calculate the duration of the audio file
    duration = len(signal) / sr

    # Create a time axis for the waveform plot
    time = librosa.times_like(signal, sr=sr)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(time, signal, linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform ({duration:.2f} seconds)')
    plt.show()


def device_simulate():
    # Load the WAV file recorded by device A
    org_signal, fs = librosa.load(audio_path, sr=sr)
    # plot_waveform(org_signal, fs)

    # Load the impulse response recorded with device B
    ir_signal, ir_fs = librosa.load(impulse_path, sr=sr)
    # plot_waveform(ir_signal, ir_fs)

    # Resample the impulse response if necessary to match the sampling rate of the recording
    if fs != ir_fs:
        ir_signal = librosa.resample(y=ir_signal, orig_sr=ir_fs, target_sr=fs)

    # Convolve the two signals
    convolved_signal = np.convolve(org_signal, ir_signal, mode='full')

    # Keep the audio length consistent
    if len(convolved_signal) > len(org_signal):
        convolved_signal = convolved_signal[:len(org_signal)]
    # plot_waveform(convolved_signal, fs)

    # Normalization
    convolved_signal = convolved_signal / np.max(np.abs(convolved_signal))

    # Export the resulting signal as a new WAV file
    sf.write('simulated_audio.wav', audio_simulated, samplerate=fs)


if __name__ == '__main__':
    device_simulate()
