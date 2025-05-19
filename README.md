# InsightsIM: Multimodal Emotion Recognition with DEAP Dataset

**InsightsIM** is a multimodal emotion recognition system that leverages video-based facial features (OpenFace) and EEG signals from the DEAP dataset to predict emotional states (valence, arousal, dominance, liking) using multi-label classification. The project includes training pipelines for video and EEG features, and a Flask-based web application for real-time emotion prediction and visualization.

This repository contains three Jupyter notebooks:
1. `VIDEO_DEAP_FYP.ipynb`: Processes OpenFace facial features for training and prediction.
2. `EEG_DEAP_FYP.ipynb`: Sets up EEG signal processing and preprocessing.
3. `DEAP_FLASK_Deployment.ipynb`: Deploys a web interface for uploading EEG (.dat) and facial (.csv) data, predicting emotions, and displaying results.

---

## Project Overview

The DEAP (Database for Emotion Analysis using Physiological signals) dataset is used to train models for classifying four emotional dimensions:
- **Valence**: Pleasantness (positive/negative).
- **Arousal**: Intensity (high/low).
- **Dominance**: Control (dominant/submissive).
- **Liking**: Preference (like/dislike).

The system extracts:
- **Facial Features**: 44 OpenFace features (e.g., action units like AU01_c to AU45_c), reduced via PCA, processed with a neural network.
- **EEG Features**: Power spectral density (PSD) from four channels (Fp1, F3, Fp2, F4) across theta, alpha, beta, and gamma bands, classified using MLPClassifier.

Predictions are mapped to 16 emotional states (e.g., "Joyful, Enthusiastic, Empowered" for HvHaHdHl) via group names (e.g., HvLaLdHl).

### Key Features
- Multi-label classification using neural networks (Keras) for facial features and MLPClassifier (scikit-learn) for EEG.
- Flask web app with an intuitive UI for uploading data and viewing predictions, groups, and emotions.
- Real-time processing of EEG (.dat) and facial (.csv) files.
- Visualizations of prediction results (planned enhancement).

---

## Repository Structure

```
InsightsIM/
├── VIDEO_DEAP_FYP.ipynb        # Facial feature processing and model training
├── EEG_DEAP_FYP.ipynb          # EEG signal preprocessing and setup
├── DEAP_FLASK_Deployment.ipynb # Flask web app for deployment
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── images/                     # Folder for screenshots and diagrams
│   ├── ui_screenshot.png       # [Placeholder] Web app UI
│   ├── pipeline_diagram.png    # [Placeholder] System architecture
│   └── predictions_chart.png   # [Placeholder] Prediction visualization
```

---

## Prerequisites

- **Python**: 3.10 or higher
- **Environment**: Google Colab (recommended) or local Jupyter environment
- **Dependencies**: Listed in `requirements.txt`
- **Dataset**: DEAP dataset (not included; requires access)
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
   git clone https://github.com/your-username/InsightsIM.git
   cd InsightsIM
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare DEAP Dataset**:
   - Obtain the DEAP dataset (.dat files for EEG, .csv files for OpenFace features).
   - Place data in a directory (e.g., `/data/deap/`) or use Kaggle dataset (`syedaliessazaidi/final-deap-dataset`).
   - Update file paths in `VIDEO_DEAP_FYP.ipynb` and `DEAP_FLASK_Deployment.ipynb` if not using Google Drive.

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

### 1. Training Facial Models (`VIDEO_DEAP_FYP.ipynb`)
- **Purpose**: Loads OpenFace features, trains a Keras neural network, and saves models.
- **Steps**:
  1. Run the notebook in Google Colab or Kaggle.
  2. Update the dataset path to point to `new_kaggle_all_combined_features_and_labels.csv`.
  3. Select 44 action unit features (e.g., AU01_c to AU45_c), apply PCA, and binarize labels (threshold at 5).
  4. Train the neural network (512→256→128→64→4 units) for multi-label classification.
  5. Save models as `fyp3_valence.pkl`, `fyp3_arousal.pkl`, etc.
- **Output**: Trained models for valence, arousal, dominance, and liking.

**Upload Image: Training Pipeline**
- Take a screenshot of the neural network architecture or training output (e.g., loss/accuracy plots).
- Save as `images/pipeline_diagram.png`.
- Embed in README:
  ```markdown
  ![Training Pipeline](images/pipeline_diagram.png)
  ```

### 2. EEG Preprocessing (`EEG_DEAP_FYP.ipynb`)
- **Purpose**: Sets up EEG signal processing with MNE and explores feature extraction.
- **Steps**:
  1. Run the notebook to install dependencies (MNE, fooof, etc.).
  2. Load DEAP .dat files and preprocess using ICA and band-pass filtering.
  3. (Note: Incomplete; feature extraction and training are in `DEAP_FLASK_Deployment.ipynb`.)
- **Output**: Preprocessed EEG data ready for feature extraction.

### 3. Web Deployment (`DEAP_FLASK_Deployment.ipynb`)
- **Purpose**: Deploys a Flask app to process EEG and facial data, predict emotions, and display results.
- **Steps**:
  1. Run the notebook in Google Colab.
  2. Ensure models are in the specified Google Drive paths.
  3. Start the Flask app and access the ngrok URL (e.g., `https://b4e4-34-75-145-246.ngrok-free.app`).
  4. Upload an EEG .dat file and a facial .csv file via the web interface.
  5. View predictions, group names (e.g., HvLaLdHl), and emotions (e.g., "Relaxed and Affectionate") in the tabs.
- **Output**: JSON response with predictions, groups, and emotions; displayed in the UI.

**Upload Image: Web UI**
- Take a screenshot of the Flask app interface showing the upload form and tabs.
- Save as `images/ui_screenshot.png`.
- Embed in README:
  ```markdown
  ![Web UI](images/ui_screenshot.png)
  ```

**Upload Image: Prediction Results**
- Capture the output of a sample prediction (e.g., Model Results tab with EEG and facial predictions).
- Save as `images/predictions_chart.png`.
- Embed in README:
  ```markdown
  ![Prediction Results](images/predictions_chart.png)
  ```

---

## Project Workflow

1. **Data Preparation**:
   - Facial: Load OpenFace features, select 44 action units, apply PCA, binarize labels.
   - EEG: Load .dat files, apply ICA and band-pass filtering, extract PSD features.

2. **Model Training**:
   - Facial: Train a Keras neural network with 44 PCA components.
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

**Upload Image: Sample Output**
- Embed the sample prediction output screenshot (`images/predictions_chart.png`) here:
  ```markdown
  ![Sample Prediction Output](images/predictions_chart.png)
  ```

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

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please include tests and update documentation for new features.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **DEAP Dataset**: Koelstra, S., et al. (2012). "DEAP: A Database for Emotion Analysis using Physiological Signals."
- **OpenFace**: For facial feature extraction.
- **MNE-Python**: For EEG signal processing.
- **xAI**: For providing Grok to assist in project analysis and documentation.

---

## Contact

For questions or feedback, contact:
- **Your Name**: your.email@example.com
- **GitHub**: [your-username](https://github.com/your-username)

**Upload Image: System Architecture**
- Create a diagram of the project workflow (data → preprocessing → models → Flask app).
- Save as `images/pipeline_diagram.png`.
- Embed above in the "Training Facial Models" section or here:
  ```markdown
  ![System Architecture](images/pipeline_diagram.png)
  ```
