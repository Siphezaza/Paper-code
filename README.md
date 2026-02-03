>📋  A template README.md for code accompanying a Machine Learning paper

# Unsupervised anomaly detection in large-scale estuarine acoustic telemetry data

This repository is the official implementation of [Unsupervised-explainable anomaly
detection in large-scale estuarine acoustic telemetry data](https://arxiv.org/pdf/2502.01543?). 

The purpose of this repository is not only to release code, but to guide practitioners step-by-step through the complete workflow of detecting anomalous detections in acoustic telemetry datasets using unsupervised machine learning and deep learning, with a particular focus on neural network autoencoders (NN-AE).

This README is intentionally written as an extended tutorial so that users can reproduce, understand, and adapt the methodology to their own telemetry systems.

## 1. What Problem Does This Code Solve?

Acoustic telemetry datasets often contain millions of detections collected irregularly over long time periods. These data frequently include anomalous detections arising from:

- Code collisions between transmitters

- Receiver malfunction

- Environmental interference

- Biological events such as mortality or tag loss

Traditional rule-based filtering approaches rely on fixed, manually defined thresholds, which are difficult to generalise to large, heterogeneous datasets.

This repository implements an unsupervised anomaly detection framework that:

- Learns only normal movement patterns

- Automatically identifies deviations as anomalies

- Avoids manual threshold selection through a data-driven optimisation procedure

 ## 2. Overview of the Workflow

The AcousNomaly pipeline consists of the following stages:

- Import raw acoustic telemetry detections

- Clean and pre-process the data

- Engineer telemetry-specific movement features

- Resample irregular time-series detections

- Train unsupervised anomaly detection models

- Automatically determine an optimal anomaly threshold

- Detect and interpret anomalous detections

Each stage is implemented explicitly and documented in the notebooks provided.

Quick Start
-----------
For users who want to reproduce the main results with minimal setup, this repository provides a fully executable workflow in Google Colab.

1. **Open the repository in Google Colab**  
   Clone or upload the GitHub repository into a Google Colab environment.

2. **Run the main analysis notebook** (Notebook with plots for the paper (2).ipynb)  
   Execute the notebook:
This notebook implements the complete end-to-end pipeline used in the paper, including:
- Data loading and preprocessing  
- Telemetry-specific feature engineering  
- Resampling of irregular detection time series  
- Training and evaluation of unsupervised anomaly detection models  
- Generation of figures and performance metrics reported in the manuscript  

3. **Use pre-trained models and optimised parameters (optional)**  
To reproduce results without retraining, the repository includes saved models and optimised parameters:
- `autoencoder_model(original_batch_size_128_50epchs3).keras`
- `best_isolation_forest_model_original.pkl`
- `best_lof_params_original.pkl`
- `best_dbscan_params_90s.pkl`
- `dbscan_results_90s.pkl`

These files allow users to directly apply trained models to new telemetry datasets or to reproduce the reported results efficiently.

4. **Inspect outputs and figures**  
All figures used in the paper (including time-series visualisations, confusion matrices, and model comparisons) are generated within the notebook.

Each step of the notebook is clearly commented and explicitly linked to the corresponding sections of the paper, enabling both reproducibility and methodological transparency.

## Repository Structure

The repository is organised to support both full reproducibility of the paper results and easy adaptation to new acoustic telemetry datasets.
─ Notebook with plots for the paper (2).ipynb
(Main end-to-end analysis notebook used to generate the results and figures reported in the paper

─ autoencoder_model(original_batch_size_128_50epchs3).keras (Trained neural network autoencoder (NN-AE)).

─ best_isolation_forest_model_original.pkl (Pre-trained Isolation Forest model).

─ best_lof_params_original.pkl (Optimised Local Outlier Factor parameters).

─ best_dbscan_params_90s.pkl (Optimised DBSCAN parameters for the 90 s augmented dataset).

─ dbscan_results_90s.pkl (Stored DBSCAN anomaly detection results).

─ README.md
Extended tutorial-style documentation for the repository.
Saved models and parameter files allow users to reproduce reported results efficiently, while the main notebook provides a transparent implementation of the full methodology.

## 3. Requirements

Software Requirements
This code was developed in a Google Colab environment with Python 3.12.7. Key libraries include:

- TensorFlow 2.18.0
- scikit-learn 1.5.1 (note: the notebook pins to 1.2.2 or 1.3.2 in some cells for compatibility—see troubleshooting below)
- tensorflow / keras
- pandas, matplotlib, seaborn, dill
- Additional: numpy, scipy (for data processing); sympy (for math if needed)

To install, create a requirements.txt file with the following and run pip install -r requirements.txt:
text

```
- tensorflow==2.18.0
- scikit-learn==1.5.1
- keras
- pandas
- matplotlib
- seaborn
- dill
- numpy==1.24.4
- scipy==1.10.1
```
Tested Environment:

- Windows 11 Enterprise (Version 23H2)
- 12th Gen Intel® Core™ i7-12700 CPU @ 2.10 GHz
- 16 GB RAM
- Google Colab (recommended for quick start, as it handles GPU/TPU if needed for NN-AE training)

## 4. Step-by-Step Tutorial
# Step 1: Data Pre-processing

This step ensures data integrity by:

- Removing missing values

- Removing duplicate detections

- Removing biologically implausible false tag IDs

- Standardising timestamps and station labels

Only biologically credible detections are retained for modelling.

## Example Code Snippet (full details in notebook section #loading data and models):
```
import pandas as pd

# Load raw detections (replace with your file path)
data_path = 'path/to/your_telemetry_data.txt'  # e.g., '/content/drive/MyDrive/Notebook_with_plots/data_without_augmentation.txt'
data = pd.read_csv(data_path, sep='\t')

# Remove duplicates and missing values (standardize timestamps)
data['datetime'] = pd.to_datetime(data['datetime'])  # Ensure datetime format
data = data.drop_duplicates().dropna(subset=['datetime', 'fishid', 'station'])
```

# Step 2: Feature Engineering

Telemetry-specific features are derived from raw detections, including:

- Distance travelled between consecutive detections

- Duration spent at the same station

- Number of detections per individual

- Number of days detected

- Number of unique stations visited

- Number of consecutive missing stations

These features encode movement behaviour rather than raw detections, improving anomaly detection performance.

## Example Code Snippet (full details in notebook section #global functions):
```
# Assuming 'data' from Step 1; add features like distance and duration
data['distance'] = data.groupby('fishid')['lat'].diff().abs()  # Example: distance between detections
data['duration_in_same_station'] = data.groupby(['fishid', 'station'])['datetime'].diff().dt.total_seconds()
data.fillna(0, inplace=True)  # Handle initial NaNs
```
# Step 3: Resampling Irregular Telemetry Data

Acoustic telemetry data are inherently irregularly sampled due to tag transmission schedules and animal movement.

To address this, a resampling strategy is applied that:

- Derives a consistent sampling rate from the data

- Approximates Shannon/Nyquist sampling requirements

- Reconstructs missing samples for normal detections

This produces time series suitable for machine learning and deep learning models.

## Example Code Snippet (full details in notebook section #temporal analysis of movements and detections functions):
```
from sklearn.utils import resample

# Resample data (e.g., to median rate; assumes 'data' from previous steps)
median_rate_seconds = 8  # Or compute from your data
resampled_data = resample(data[data['Anomaly'] == 0], n_samples=len(data), random_state=42)
resampled_data['datetime'] = pd.date_range(start=data['datetime'].min(), periods=len(resampled_data), freq=f'{median_rate_seconds}s')
```
# Step 4: Unsupervised Model Training

The following unsupervised anomaly detection models are implemented and compared:

- Isolation Forest (IF)

- Local Outlier Factor (LOF)

- DBSCAN

- Neural Network Autoencoder (NN-AE)

The NN-AE is trained exclusively on normal detections, learning to reconstruct normal movement patterns. Anomalies are identified through elevated reconstruction errors.

## Example Code Snippet (full details in notebook section #building and fitting the-model; here for NN-AE):
```
from tensorflow.keras.models import load_model
import numpy as np

# Load pre-trained NN-AE model
model_path = 'autoencoder_model(original_batch_size_128_50epchs3).keras'
model = load_model(model_path, compile=False)

# Predict on scaled features (assumes X_scaled from preprocessing)
predictions = model.predict(X_scaled)
reconstruction_errors = np.mean(np.square(X_scaled - predictions), axis=1)
```
# Step 5: Automatic Threshold Optimisation

A key contribution of this work is a data-driven threshold selection algorithm for autoencoders.

Instead of selecting arbitrary percentiles, the algorithm:

- Identifies thresholds that achieve maximum recall (zero false normals)

- Among these, selects the threshold that minimises false anomalies

- Produces a robust, reproducible decision boundary

This removes the need for manual threshold tuning.

## Example Code Snippet (full details in notebook section #global functions):
```
import numpy as np
from sklearn.metrics import recall_score

# Compute errors and find optimal threshold (assumes reconstruction_errors from Step 4)
percentile = 65  # Example from paper
threshold = np.percentile(reconstruction_errors, percentile)
y_pred = (reconstruction_errors > threshold).astype(int)
recall = recall_score(y_true, y_pred)  # Optimize for max recall
```
# Step 6: Results and Interpretation

Outputs include:

- Anomaly labels per detection

- Confusion matrices and performance metrics

- Time-series visualisations highlighting anomalous behaviour

- Fish-level summaries for ecological interpretation

Detected anomalies correspond to biologically implausible movement patterns such as prolonged stationarity, skipped receivers, or single-station detections.

## Results (65s Augmented Data)

The table below reports the performance of all unsupervised anomaly detection models evaluated on the
90 s augmented dataset. Precision, Recall, and F1-score are computed from the confusion matrices
reported in the paper.

| Model name              | Precision (%) | Recall (%) | F1-score (%) |Accuracy (%)  |
|-------------------------|---------------|------------|--------------|--------------|
| Isolation Forest (IF)   |      100         |  0.16      |   0.33         |   67.62      |
| Local Outlier Factor    |  13.16          |     0.40   |    0.78          |   66.83      |
| DBSCAN                  |  4.69          |     0.24   |     0.47      |     66.00    |
| Autoencoder (NN-AE)     | 99.55           |     100.0  | 99.77         |      99.85   |


## Adapting the Code to Other Systems

Although demonstrated using dusky kob in the Breede Estuary, this framework is transferable to:

- Other fish species

- Other estuaries or receiver arrays

- Large multi-year telemetry networks

Users may need to adjust:

- Feature definitions

- Anomaly criteria

- Resampling intervals

The overall pipeline remains unchanged.

## Troubleshooting / FAQ
Here are some common issues and solutions based on testing the workflow:

- Dependency Conflicts: The notebook pins scikit-learn to versions like 1.2.2 or 1.3.2 in some cells (e.g., !pip install scikit-learn==1.2.2). If you encounter version errors (e.g., with numpy or scipy), run the uninstall/reinstall commands from the notebook first. 

- Handling Large Datasets: If memory errors occur during loading or training (e.g., millions of detections), process in chunks with pandas (e.g., pd.read_csv(..., chunksize=100000)). For NN-AE training, reduce batch size (e.g., from 128 to 64) or use Colab's GPU runtime.

- Irregular Timestamps or Resampling Failures: Ensure data is sorted by 'datetime' before resampling. Check for timezone issues with data['datetime'] = pd.to_datetime(data['datetime'], utc=True). If gaps are too large, adjust median_sampling_rate_seconds in the augmentation function.

- Adapting to Non-Estuarine Data: For ocean or river systems, modify features like 'distance' to account for currents (e.g., add velocity). Test on a small subset first and retrain models.

- Model Loading Errors: If .keras or .pkl files fail to load, ensure paths match (e.g., mount Google Drive with drive.mount('/content/drive')). Use import dill for pickle files as in the notebook.


## Relation to the Paper

This repository implements the full methodology described in the paper, including:

- Telemetry-specific feature engineering

- Resampling of irregular detections

- Unsupervised anomaly detection

- Automated threshold optimisation

The notebooks correspond directly to the Methods and Results sections.

## Data Availability and Ethics
----------------------------
Due to ethical and permitting restrictions, the raw acoustic telemetry data used in this study are not publicly available. Data will be made available on request.
The repository therefore focuses on providing:
- fully reproducible code,
- clear data format specifications, and
- example workflows that can be applied to other telemetry datasets.

Users can adapt the pipeline to their own data by matching the input format described in the preprocessing notebooks.


## Citation

If you use this code, please cite:
```setup
@article{zaza_acousnomaly,
  title={Unsupervised anomaly detection in large-scale estuarine acoustic telemetry data},
  author={Zaza, Siphendulwe and Atemkeng, Marcellin and Murray, Taryn S. and Filmalter, John D. and Cowley, Paul D.},
  year={2025}
}
```
