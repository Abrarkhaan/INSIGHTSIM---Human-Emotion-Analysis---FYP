# InsightSim: Multimodal Human Emotion Analysis

**InsightSim** is a multimodal human emotion analysis and recognition system that leverages video-based facial features (OpenFace) and EEG signals from the DEAP dataset to predict emotional states (valence, arousal, dominance, liking) using single and multi-label classification. The project includes training pipelines for video and EEG features, and a Flask-based web application for real-time emotion prediction and visualization.

This repository contains:
1. `DEAP Dataset video processing, SIFT features extraction, models building notebooks`
2. `AMAGOS Dataset video processing, HOG features extraction, models building notebooks`
3. `AMIGOS Dataset video features (extracted with OpenFace) processing, models building notebooks`
4. `DEAP Dataset video features (extracted with OpenFace) processing, models building notebooks`
5. `DEAP Dataset EEG signals processing, models building notebook`
6. `FLASK Web Deployment with NGROK`

---

## Project Overview

The DEAP (Database for Emotion Analysis using Physiological signals) and AMIGOS (A Dataset for Affect, Personality and Mood Research on Individuals and Groups) datasets are used to train models for classifying four emotional dimensions:
- **Valence**: Pleasantness (positive/negative).
- **Arousal**: Intensity (high/low).
- **Dominance**: Control (dominant/submissive).
- **Liking**: Preference (like/dislike).
<img width="982" height="371" alt="image" src="https://github.com/user-attachments/assets/7640cdd4-38c5-4fe1-bd81-7fe1b7202777" />


The system extracts:
- **Facial Features**: OpenFace features (e.g., action units like AU01_c to AU45_c), reduced via PCA, processed with a neural network. (This step is done with OpenFace through CLI or OpenFace Application)
<img width="794" height="394" alt="image" src="https://github.com/user-attachments/assets/54792046-432d-4aba-920d-ab69846401ab" />


- **EEG Features**: Power spectral density (PSD) from four channels (Fp1, F3, Fp2, F4) across theta, alpha, beta, and gamma bands, classified using MLPClassifier.
<img width="976" height="220" alt="image" src="https://github.com/user-attachments/assets/3d3deaac-d985-479b-b52c-506c2918fb91" />


Predictions are mapped to 16 emotional states (e.g., "Joyful, Enthusiastic, Empowered" for HvHaHdHl) via group names (e.g., HvLaLdHl).

### Key Features
- Multi-label classification using neural networks (Keras) for facial features and MLPClassifier (scikit-learn) for EEG.
- Flask web app with an intuitive UI for uploading data and viewing predictions, groups, and emotions.
- Real-time processing of EEG (.dat) and facial (.csv) files.
- Visualizations of prediction results (planned enhancement).

---

## Repository Structure

```
..
```

---

## Prerequisites

- **Python**: 3.10 or higher
- **Environment**: Google Colab (recommended) or local Jupyter environment
- **Dependencies**: Listed in `requirements.txt`
- **Dataset**: DEAP dataset, AMIGOS dataset (not included; requires access)
- **ngrok**: For public URL in Flask deployment (auth token required)

### Dependencies
Install required libraries:
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy==1.26.4
pandas
tensorflow
scikit-learn==1.5.2
mne==0.22.0
scipy==1.13.0
fooof
pickle
flask
flask-ngrok
pyngrok
matplotlib
seaborn
joblib
```

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abrarkhaan/INSIGHTSIM---Human-Emotion-Analysis---FYP.git
   cd INSIGHTSIM---Human-Emotion-Analysis---FYP
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare DEAP Dataset**:
   - Obtain the DEAP dataset (.dat files for EEG, Videos for OpenFace features).
   - Place data in a directory (e.g., `/data/deap/`).
   - Update file paths.

4. **Set Up ngrok**:
   - Sign up for an ngrok account and obtain an auth token.
   - Replace the hardcoded token in `DEAP_FLASK_Deployment.ipynb` with your token or use an environment variable:
     ```python
     import os
     NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN', 'your-token-here')
     ```

5. **Google Drive (Optional)**:
   - If using Google Colab, mount Google Drive for model and data access:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Place models in `/content/drive/MyDrive/for_abrar_frontend_testing/` and `/content/drive/MyDrive/flask deployment/`.

---

## Usage

---
..
---

## Project Workflow

1. **Data Preparation**:
   - Facial: Load OpenFace features, select action units, apply PCA, binarize labels.
   - EEG: Load .dat files, apply ICA and band-pass filtering, extract PSD features.

2. **Model Training**:
   - Facial: Train a Keras neural network with PCA components.
   - EEG: Train MLPClassifier models on 16 PSD features (4 bands × 4 channels).

3. **Deployment**:
   - Upload EEG and facial data via the Flask app.
   - Process data, predict binary labels, map to group names and emotions.
   - Display results in a tabbed interface.

---

## Results

- **Facial Model**: Achieves multi-label classification with a neural network, handling 44 features post-PCA.
- **EEG Model**: Uses MLPClassifier for robust PSD-based classification.
- **Deployment**: Successfully processes sample data, e.g.:
  - EEG: [1, 0, 0, 1] → HvLaLdHl → "Relaxed and Affectionate"
  - Facial: [0, 1, 1, 1] → LvHaHdHl → "Determined but Critical"

**User Interface**
<img width="975" height="523" alt="image" src="https://github.com/user-attachments/assets/36886881-db3a-4b20-a073-84622227b792" />

**Results on App**
<img width="975" height="502" alt="image" src="https://github.com/user-attachments/assets/e111f6e7-9215-4f1d-b721-0cef1ebda47e" />
<img width="975" height="467" alt="image" src="https://github.com/user-attachments/assets/8c197e56-3ba7-4f5f-8b43-e086c6e95725" />
<img width="975" height="489" alt="image" src="https://github.com/user-attachments/assets/0a110e2c-4fa3-4cab-b044-581c88c2a412" />
<img width="975" height="499" alt="image" src="https://github.com/user-attachments/assets/3520ff33-0a79-46cf-ab9b-dcd29ac7eda4" />


---

## Limitations and Future Work

- **Limitations**:
  - Google Drive dependency limits portability.
  - Facial feature averaging may lose temporal dynamics.
  - EEG processing uses only one trial; full dataset utilization is needed.
  - Version mismatches (e.g., scikit-learn 1.5.2 vs. 1.6.1) risk incompatibility.

- **Future Work**:
  - Integrate PCA in the deployment pipeline for facial features.
  - Process all EEG trials for robust predictions.
  - Add visualizations (e.g., bar charts for predictions).
  - Deploy on a production server (e.g., Heroku) instead of ngrok.
  - Fuse EEG and facial predictions for improved accuracy.

---

## Contributing???

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please include tests and update documentation for new features.

---

## License

--.

---

## Acknowledgments

- **DEAP Dataset**: Koelstra, S., et al. (2012). "DEAP: A Database for Emotion Analysis using Physiological Signals."
- **AMIGOS Dataset**: "AMIGOS: A Dataset for Affect, Personality and Mood Research on Individuals and Groups."
- **OpenFace**: For facial feature extraction.
- **MNE-Python**: For EEG signal processing.
- **NGROK**
- **Pytorch**
- **Scikit-learn**
- **OpenCV Haar Cascades**

---

## Contact

For questions or feedback, contact:
- **LinkedIn**: [AbrarKhaan](https://linkedin.com/in/abrarkhaan)
- **GitHub**: [AbrarKhaan](https://github.com/abrarkhaan)
