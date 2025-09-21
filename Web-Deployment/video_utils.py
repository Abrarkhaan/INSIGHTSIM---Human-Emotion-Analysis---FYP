import joblib

def facial_prediction(X):

  path_valence = "/content/drive/MyDrive/for_abrar_frontend_testing/fyp3_valence.pkl"
  path_arousal = "/content/drive/MyDrive/for_abrar_frontend_testing/fyp3_arousal.pkl"
  path_dominance = "/content/drive/MyDrive/for_abrar_frontend_testing/fyp3_dominance.pkl"
  path_liking = "/content/drive/MyDrive/for_abrar_frontend_testing/fyp3_liking.pkl"

  # Load the models
  model_valence = joblib.load(path_valence)
  model_arousal = joblib.load(path_arousal)
  model_dominance = joblib.load(path_dominance)
  model_liking = joblib.load(path_liking)

  # Make predictions on the test data
  y_pred_valence = model_valence.predict(X)
  y_pred_arousal = model_arousal.predict(X)
  y_pred_dominance = model_dominance.predict(X)
  y_pred_liking = model_liking.predict(X)

  binary_valence = (y_pred_valence > 0.5).astype(int)
  binary_arousal = (y_pred_arousal > 0.5).astype(int)
  binary_dominance = (y_pred_dominance > 0.5).astype(int)
  binary_liking = (y_pred_liking > 0.5).astype(int)

  return binary_valence,binary_arousal,binary_dominance,binary_liking

def facial_preprocess(main_df):
    print(main_df.info())
    print("Missing values before cleanup:\n", main_df.isnull().sum())

    main_df = main_df.dropna()

    for col in main_df.columns:
        if main_df[col].dtype == 'float64':
            main_df[col] = main_df[col].fillna(main_df[col].mean())
        elif main_df[col].dtype == 'int64':
            main_df[col] = main_df[col].fillna(main_df[col].median())

    print("Missing values after cleanup:\n", main_df.isnull().sum())

    if 'frame' in main_df.columns:
        main_df['frame'] = main_df['frame'].astype(int)

    # Extract features
    X = main_df.iloc[:, 5:-4].values  # shape: (n_frames, n_features)

    # Use average or first frame
    X_avg = X.mean(axis=0).reshape(1, -1)  # shape: (1, n_features)
    print("Input shape to facial_prediction:", X_avg.shape)

    valence, arousal, dominance, liking = facial_prediction(X_avg)

    return valence, arousal, dominance, liking

