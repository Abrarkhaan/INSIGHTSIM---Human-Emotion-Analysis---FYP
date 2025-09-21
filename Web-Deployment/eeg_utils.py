from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import mne
from itertools import chain
from scipy.stats import pearsonr


def load_EEG_models():
    # Load pre-trained models
    valence_model = load("/content/drive/MyDrive/flask deployment/MLPClassifier_valence(alpha=1e-05, hidden_layer_sizes=(92, 46), learning_rate_init=0.09,              max_iter=1000, n_iter_no_change=80).pkl")
    arousal_model = load("/content/drive/MyDrive/flask deployment/MLPClassifier_arousal(alpha=1e-05, hidden_layer_sizes=(53, 50), learning_rate_init=0.09,              max_iter=1000, n_iter_no_change=80).pkl")
    dominance_model = load("/content/drive/MyDrive/flask deployment/MLPClassifier_dominance(activation='tanh', alpha=1e-05, hidden_layer_sizes=(58, 33),              learning_rate_init=0.09, max_iter=1000, n_iter_no_change=80,              solver='sgd').pkl")
    liking_model = load("/content/drive/MyDrive/flask deployment/MLPClassifier_Liking(alpha=1e-05, hidden_layer_sizes=(53, 25), learning_rate_init=0.09,              max_iter=1000, n_iter_no_change=80).pkl")
    return valence_model, arousal_model, dominance_model, liking_model

def get_feature(data):
    channel_no = [0, 2, 16, 19]  # Specify channels to use
    feature_matrix = []

    # Process each channel specified in channel_no
    for ith_channel in channel_no:
        signal = data[ith_channel, :]  # Extract the signal for the channel

        # Compute PSD
        psd, freqs = plt.psd(signal, Fs=128, NFFT=256)

        # Extract frequency band powers
        theta_mean = np.mean(psd[np.logical_and(freqs >= 4, freqs <= 7)])
        alpha_mean = np.mean(psd[np.logical_and(freqs >= 8, freqs <= 13)])
        beta_mean = np.mean(psd[np.logical_and(freqs >= 13, freqs <= 30)])
        gamma_mean = np.mean(psd[np.logical_and(freqs >= 30, freqs <= 40)])

        feature_matrix.append([theta_mean, alpha_mean, beta_mean, gamma_mean])

    # Flatten the feature matrix for all selected channels
    return np.array(list(chain.from_iterable(feature_matrix)))

# Signal processing and feature extraction functions
def SignalPreProcess(eeg_rawdata):
    print("in signal preprocess")
    N_C = 20  # Number of ICA components
    droping_components = 'one'  # Drop one or two components
    ch_names = [
        "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz",
        "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
        "PO4", "O2"
    ]
    info = mne.create_info(ch_names=ch_names, ch_types=['eeg'] * 32, sfreq=128, verbose=False)
    raw_data = mne.io.RawArray(eeg_rawdata, info, verbose=False)
    raw_data.load_data().filter(l_freq=4, h_freq=48, method='fir', verbose=False)
    print("before ica")
    ica = ICA(n_components=N_C, random_state=97, verbose=False)
    ica.fit(raw_data)
    ica_sources = ica.get_sources(raw_data).get_data()
    eog_signal = raw_data.copy().pick_channels(['Fp1']).get_data()[0]
    print("before correlations")
    correlations = np.array([pearsonr(component, eog_signal)[0] for component in ica_sources])
    if droping_components == 'one':
        ica.exclude = [np.argmax(np.abs(correlations))]
    else:
        top_two_indices = np.argsort(np.abs(correlations))[-2:]
        ica.exclude = list(top_two_indices)

    ica.apply(raw_data, verbose=False)
    raw_data.set_eeg_reference('average', ch_type='eeg')
    return np.array(raw_data.get_data())

def signal_pro(input_data):
    print("in signal pro")
    input_data = SignalPreProcess(input_data.copy())
    return input_data