import pickle
import pandas as pd
import numpy as np
from video_utils import facial_preprocess
from eeg_utils import load_EEG_models, get_feature, signal_pro
from get_emotion import get_group_name, get_emotions_from_group

valence_model, arousal_model, dominance_model, liking_model = load_EEG_models()

def process_files(dat_file, csv_file):
    try:
        # Process EEG data
        deap_dataset = pickle.load(dat_file.stream, encoding='latin1')
        print("EEG file read")
        # Extract EEG data
        data = np.array(deap_dataset['data'])
        data = data[0:40, 0:32, 384:8064]
        data_32_channels = data[0][:32]
        print("EEG data extracted")
        # Process EEG data
        processed_eeg_data = signal_pro(data_32_channels)
        print("EEG data processed")
        eeg_feature = get_feature(processed_eeg_data)
        print("EEG features extracted")
        eeg_features = eeg_feature.reshape(1, -1)
        print("EEG features reshaped")

        # EEG Predictions
        eeg_valence_prediction = valence_model.predict(eeg_features)
        eeg_arousal_prediction = arousal_model.predict(eeg_features)
        eeg_dominance_prediction = dominance_model.predict(eeg_features)
        eeg_liking_prediction = liking_model.predict(eeg_features)
        print("EEG predictions obtained")
        # EEG Group and Emotions
        eeg_group_name = get_group_name(eeg_valence_prediction, eeg_arousal_prediction, eeg_dominance_prediction, eeg_liking_prediction)
        eeg_emotions = get_emotions_from_group(eeg_group_name)
        print("EEG group and emotions processed")

        # Process facial data
        facial_data = pd.read_csv(csv_file.stream)
        print("Facial CSV file read")
        facial_valence_prediction,facial_arousal_prediction,facial_dominance_prediction,facial_liking_prediction = facial_preprocess(facial_data)
        print("Facial predictions obtained")
        print(facial_valence_prediction,facial_arousal_prediction,facial_dominance_prediction,facial_liking_prediction)
        # Facial Group and Emotions
        facial_group_name = get_group_name(facial_valence_prediction, facial_arousal_prediction, facial_dominance_prediction, facial_liking_prediction)
        facial_emotions = get_emotions_from_group(facial_group_name)
        print("Facial group and emotions processed")

        # Prepare structured output
        return {
            "predictions": (
                f"EEG Predictions:\n"
                f"  Valence: {eeg_valence_prediction[0]}\n"
                f"  Arousal: {eeg_arousal_prediction[0]}\n"
                f"  Dominance: {eeg_dominance_prediction[0]}\n"
                f"  Liking: {eeg_liking_prediction[0]}\n"
                f"\nFacial Predictions:\n"
                f"  Valence: {facial_valence_prediction[0]}\n"
                f"  Arousal: {facial_arousal_prediction[0]}\n"
                f"  Dominance: {facial_dominance_prediction[0]}\n"
                f"  Liking: {facial_liking_prediction[0]}"
            ),
            "groups": (
                f"EEG Group: {eeg_group_name}\n"
                f"Facial Group: {facial_group_name}"
            ),
            "emotions": (
                f"EEG Emotions: {eeg_emotions}\n"
                f"Facial Emotions: {facial_emotions}"
            )
        }

    except Exception as e:
        return {"error": f"Error during file processing: {str(e)}"}