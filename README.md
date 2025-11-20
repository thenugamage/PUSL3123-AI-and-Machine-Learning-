# PUSL3123-AI-and-Machine-Learning-
PUSL3123 AI and Machine Learning Assignment

**Acceleration-Based User Authentication**  
**Smartwatch IMU Gait Biometrics – Group 64**

**Module:** PUSL3123 – AI and Machine Learning  
**Lecturer:** Dr Neamah Al-Naffakh  
**Programme:** BSc (Hons) Software Engineering  
**Group:** 64

---

### 1. Overview

This repository contains all MATLAB code and experiments for a smartwatch-based behavioural authentication system using accelerometer and gyroscope data.

The main idea is to recognise the legitimate smartwatch wearer from their **gait** (walking pattern), using short windows of IMU data and supervised machine-learning models.

We work with:

- 10 users
    
- 2 days of data per user (FD = First Day, MD = Another Day)
    
- Multiple models:
    
    - FFMLP (PatternNet) – main coursework model
        
    - KNN baseline
        
    - PCA + Neural Network (experimental)
        
    - Fisher + KNN (experimental, over-fitted upper bound)
        

---

### 2. Repository Structure

Root folder:

- **FFMLP** – main coursework implementation (three-script pipeline)
    
- **Test_model_1** – PCA + Neural Network and Fisher + KNN experiments
    
- **Test_model_2** – alternative pipeline with strong KNN baseline
    
- **.gitignore** – Git ignore rules (not important for running the code)
    
- **README** – this description
    

Inside each project:

1. **FFMLP**
    
    - `data/raw_data/`  
        20 smartwatch CSV files: U1–U10 × {FD, MD}.
        
    - `data/interim/`  
        Automatically generated filtered and normalised signals.
        
    - `data/processed/`  
        Automatically generated features, templates, trained models, metrics and plots.
        
    - `script1_preprocessing_features.m`  
        Pre-processing and feature extraction.
        
    - `script2_training_and_testing.m`  
        FFMLP (PatternNet) training and testing (Method A and Method B).
        
    - `script3_evaluation_and_optimisation.m`  
        FAR/FRR/EER evaluation, DET curves and optimisation experiments.
        
2. **Test_model_1**
    
    - `data/…` – similar structure to FFMLP.
        
    - `src/script1_preprocess_and_features.m` – preprocessing and features.
        
    - `src/model_training.m` – PCA + Neural Network, Fisher + KNN.
        
    - `src/testing_optimization.m` – extra tests/optimisation.
        
3. **Test_model_2**
    
    - `config/config.m` – central configuration (paths, parameters).
        
    - `data/raw/` – optional copy of original IMU CSVs.
        
    - `data/interim/` – preprocessed data (generated).
        
    - `data/processed/` – extracted features (generated).
        
    - `results/` – metrics CSVs and plots (generated).
        
    - `src/data_preprocessing.m` – preprocessing pipeline.
        
    - `src/feature_extraction.m` – feature extraction.
        
    - `src/model_training.m` – FFMLP-type models.
        
    - `src/model_training_knn.m` – KNN baseline experiments.
        

All folders `data/interim`, `data/processed` and `results` are generated automatically and can be deleted; they will be recreated when the scripts are run.

---

### 3. Main Pipeline – FFMLP Project

This is the **official coursework implementation** and matches the written report.

#### 3.1 Script 1 – Pre-processing and Feature Extraction

`FFMLP/script1_preprocessing_features.m`

1. Reads 20 CSV files from `data/raw_data/`  
    (U1NW_FD.csv … U10NW_MD.csv).
    
2. Pre-processing:
    
    - Fixes timestamps and merges duplicates.
        
    - Resamples all channels to 31 Hz.
        
    - Applies a 4th-order Butterworth low-pass filter at 10 Hz to:  
        AccX, AccY, AccZ, GyroX, GyroY, GyroZ.
        
3. Normalisation:
    
    - Computes global mean and standard deviation for each channel.
        
    - Applies global z-score normalisation across all users and days.
        
4. Segmentation:
    
    - Splits each signal into 2-second windows with 50% overlap.
        
5. Feature extraction:
    
    - For each window and each of the 6 channels, calculates:  
        mean, standard deviation, minimum, maximum, range and RMS.
        
    - Total of 36 features per window.
        
6. Outputs:
    
    - `features_all.(mat/csv)` – all windows, with user ID and day type.
        
    - `reference_templates.(mat/csv)` – FD windows only.
        
    - `testing_templates.(mat/csv)` – MD windows only.
        
    - Variability and PCA plots in `data/processed/analysis/`.
        

Script 1 is **safe to run multiple times**; it clears and rebuilds the interim and processed data.

---

#### 3.2 Script 2 – FFMLP Training and Testing

`FFMLP/script2_training_and_testing.m`

Implements a feed-forward multilayer perceptron (MATLAB `patternnet`) with:

- 1 hidden layer, 20 neurons
    
- Hyperbolic tangent (tansig) activation in hidden layer
    
- Softmax output layer
    
- Cross-entropy loss
    
- Scaled Conjugate Gradient training (`trainscg`)
    

Two evaluation methods are used:

**Method A – 70/30 random split (same-session)**

- Pooled 7160 windows (FD + MD).
    
- Stratified 70/30 train/test split by user.
    
- Achieved approximately:
    
    - 96.9% training accuracy
        
    - 95.0% test accuracy
        

**Method B – FD → MD (cross-day)**

- Train only on FD windows (3580) and test on MD windows (3580).
    
- Features standardised using FD statistics only.
    
- Network trained 5 times with different random seeds; the best MD model is kept.
    
- Best run achieved:
    
    - 97.74% FD training accuracy
        
    - 88.32% MD test accuracy
        

Script 2 saves:

- `trained_patternnet_A.mat` – Method A model and parameters.
    
- `trained_patternnet_B_fd_md.mat` – best FD→MD model.
    
- `patternnet_outputs.mat` – data for Script 3 (scores, labels, etc.).
    

---

#### 3.3 Script 3 – Biometric Evaluation and Optimisation

`FFMLP/script3_evaluation_and_optimisation.m`

This script treats the FFMLP outputs as scores in a biometric system and evaluates:

1. **FD-only DET and EER**
    
    - Uses FD (training) scores only.
        
    - Computes False Acceptance Rate (FAR) and False Rejection Rate (FRR) for thresholds from 0 to 1.
        
    - Plots a DET-style curve (FRR vs FAR).
        
    - Equal Error Rate (EER) ≈ 0.73% at threshold τ ≈ 0.18.
        
2. **FD → MD FAR/FRR/EER**
    
    - Uses MD scores with the FD-trained model.
        
    - For each user:
        
        - Genuine scores: that user’s own MD windows.
            
        - Impostor scores: MD windows from all other users.
            
    - Computes per-user accuracy and EER.
        
    - Computes global EER by pooling all genuine and impostor scores.
        
    - Global EER ≈ 4.26% at threshold τ ≈ 0.03.
        
    - Per-user metrics are saved in `eer_results_patternnet_fd_md.csv`.
        
3. **Optimisation experiments**
    
    - Hidden neuron sweep (10, 20, 40 neurons) for FD→MD.
        
    - KNN baseline (k = 1, 3, 5) using the same FD→MD split.
        
    - PatternNet with top 10 Fisher-ranked features (using `feature_variability_top.csv`).
        

The script also outputs DET curves for FD-only and FD→MD and stores detailed FAR/FRR curves for further analysis.

---

### 4. Experimental Projects

Although FFMLP is the main coursework model, two extra projects explore alternative methods.

#### 4.1 Test_model_2 – KNN Baseline and Alternative FFMLP

- Implements a similar preprocessing and feature-extraction pipeline.
    
- `model_training_knn.m` trains KNN classifiers with various k values.
    
- For example, k = 1 achieves around 88.16% cross-day (FD→MD) accuracy, similar to the FFMLP.
    
- This confirms the quality of the features but shows that KNN is less scalable because it must store all training windows.
    

#### 4.2 Test_model_1 – PCA + Neural Network and Fisher + KNN

- PCA + Neural Network:
    
    - Applies PCA to reduce dimensionality, then trains an FFMLP on the principal components.
        
    - Achieves high average accuracy (~95.66%) but highly unstable EER values (0–28%), making it unreliable.
        
- Fisher + KNN:
    
    - Uses Fisher-style discriminant projections combined with KNN.
        
    - Achieves very high accuracy (~99.74%) with extremely low FAR/FRR in same-session tests.
        
    - However, this is clearly over-fitted and unsuitable for realistic deployment; it mainly serves as an upper bound.
        

---

### 5. Summary of Key Results

- **Method A (70/30 mixed):**  
    About 95% test accuracy.
    
- **Method B (FD→MD, FFMLP):**  
    88.32% MD test accuracy, global EER ≈ 4.26%.
    
- **FD-only upper bound:**  
    EER ≈ 0.73%.
    
- **KNN baseline (FD→MD):**  
    About 88.16% MD accuracy for k = 1.
    
- **PCA + NN and Fisher + KNN:**  
    Useful for comparison and discussion but not used as the final practical model.
    

Overall, the FFMLP (PatternNet) with 36 time-domain features provides a good balance between accuracy, stability and practicality for smartwatch-based continuous authentication.