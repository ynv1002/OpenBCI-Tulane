#!/usr/bin/env python
# coding: utf-8

# # BCI Project: Data Cleaning and Analysis
# 
# In this notebook, we'll clean and analyze the two datasets you provided. Both are EEG recordings from an OpenBCI Cyton using standard 10-20 placement (C3: Ch1, C4: Ch3, Cz: Ch2, P3: Ch4, P4: Ch6, Pz: Ch5, O1: Ch7, O2: Ch8):
# 1. `LR-2-27-26-(01).csv` (Repeated clench Left vs Right hand)
# 2. `HR-2-27-26-(01).csv` (Sustained hold vs Repeated clench)
# 
# ### Step 1: Loading Data and Inspections
# Let's import `pandas`, load the datasets, assign the correct column names, and check for any flatlined or completely railed channels.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File Paths
eeg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-2-27-26-(01).csv'
hr_eeg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN/HR-2-27-26-(01).csv'

# Read data (treating it as tabular data)
eeg_df = pd.read_csv(eeg_file, header=None, sep='\t')
hr_eeg_df = pd.read_csv(hr_eeg_file, header=None, sep='\t')

assert eeg_df.shape[1] >= 9, "Expected at least 9 columns (index + 8 channels)"

print(f"LR EEG Data Shape: {eeg_df.shape}")
print(f"HR EEG Data Shape: {hr_eeg_df.shape}")


# In[ ]:


# OpenBCI data channels: 
# 0: Sample Index
# 1: C3, 2: Cz, 3: C4, 4: P3, 5: Pz, 6: P4, 7: O1, 8: O2
# Last column: Marker

def format_openbci_df(df):
    num_cols = df.shape[1]
    col_names = [f'col_{i}' for i in range(num_cols)]
    col_names[0] = 'Sample_Index'
    for i in range(1, 9):
        col_names[i] = f'Channel_{i}'

    # The last column is the marker
    col_names[-1] = 'Marker'

    df.columns = col_names

    # Keep only the relevant columns for our analysis (Channels and Marker)
    relevant_cols = ['Sample_Index'] + [f'Channel_{i}' for i in range(1, 9)] + ['Marker']
    return df[relevant_cols]

eeg_df = format_openbci_df(eeg_df)
hr_eeg_df = format_openbci_df(hr_eeg_df)

print("LR EEG Markers found:\n", eeg_df['Marker'].value_counts())
print("\nHR EEG Markers found:\n", hr_eeg_df['Marker'].value_counts())
print(eeg_df.head())


# In[ ]:


def check_flat_or_railed(df, dataset_name):
    print(f"\n--- {dataset_name} Channel Checks ---")
    channels = [f'Channel_{i}' for i in range(1, 9)]
    for ch in channels:
        ch_data = df[ch]
        std_dev = ch_data.std()
        min_val = ch_data.min()
        max_val = ch_data.max()
        if std_dev == 0:
            print(f"⚠️ WARNING: {ch} is completely FLATLINED (constant value: {min_val})")
        elif max_val >= 187000 or min_val <= -187000:  # Common max/min rail for OpenBCI Cyton (24-bit)
            print(f"⚠️ WARNING: {ch} is RAILED (hits ADC limits: min {min_val}, max {max_val})")
        else:
            print(f"✅ {ch} seems okay (min: {min_val:.2f}, max: {max_val:.2f}, std: {std_dev:.2f})")

check_flat_or_railed(eeg_df, "LR EEG Data")
check_flat_or_railed(hr_eeg_df, "HR EEG Data")


# ### Step 2: Data Cleaning and Interval Extraction
# 
# As we've seen, **Channel_2 (Cz)** and **Channel_5 (Pz)** are railed on both datasets. We will drop them from the datasets.
# 
# **Movement Classes**:
# 1. **Left Clench** (LR Dataset, Markers 1->2)
# 2. **Right Clench** (LR Dataset, Markers 3->4)
# 3. **Repeated Clench** (HR Dataset, Markers 3->4)
# 
# *(Note: The Sustained Hold [HR Dataset 1->2] was used for protocol structure, but is not a target classification class).* 
# 
# **Norm (Baseline) Extraction**:
# We need to explicitly extract the *complement* of the active trials. Valid baseline data exists:
# - Before the very first trial starts.
# - Between the end of the first trial and start of the second.
# - After the final trial ends.
# 
# We will find the explicit start/end indices of the trials, and then slice the data *outside* those ranges for our "Norm" samples.

# In[ ]:


# Drop railed channels
channels_to_drop = ['Channel_2', 'Channel_5']
eeg_clean = eeg_df.drop(columns=channels_to_drop, errors='ignore').reset_index(drop=True)
hr_eeg_clean = hr_eeg_df.drop(columns=channels_to_drop, errors='ignore').reset_index(drop=True)

def get_trial_intervals(df, start_marker, end_marker):
    intervals = []
    in_trial = False
    start_idx = None
    markers = df['Marker'].values
    for i, m in enumerate(markers):
        if m == start_marker and not in_trial:
            start_idx = i
            in_trial = True
        elif m == end_marker and in_trial:
            intervals.append((start_idx, i))
            in_trial = False

    return intervals

def extract_trials_from_intervals(df, intervals):
    return [df.iloc[start:end+1] for start, end in intervals]

def get_norm_intervals(df, all_trial_intervals, buffer=500):
    sorted_invs = sorted(all_trial_intervals, key=lambda x: x[0])
    norm_intervals = []
    current_idx = 0
    df_length = len(df)
    for start, end in sorted_invs:
        if start - buffer > current_idx:
            norm_intervals.append((current_idx, start - buffer))
        current_idx = end + buffer
    if df_length > current_idx:
        norm_intervals.append((current_idx, df_length - 1))
    return norm_intervals

def sample_norm_windows(df, norm_intervals, window_size=2000):
    norm_trials = []
    group_ids = []  # Track which distinct interval block a sample came from
    for group_idx, (start, end) in enumerate(norm_intervals):
        duration = end - start
        if duration >= window_size:
            for chunk_start in range(start, end - window_size + 1, window_size):
                norm_trials.append(df.iloc[chunk_start:chunk_start+window_size])
                group_ids.append(group_idx)
    return norm_trials, group_ids

lr_left_invs = get_trial_intervals(eeg_clean, 1.0, 2.0)
lr_right_invs = get_trial_intervals(eeg_clean, 3.0, 4.0)
lr_norm_invs = get_norm_intervals(eeg_clean, lr_left_invs + lr_right_invs)
trials_left = extract_trials_from_intervals(eeg_clean, lr_left_invs)
trials_right = extract_trials_from_intervals(eeg_clean, lr_right_invs)
trials_norm_lr, g_lr = sample_norm_windows(eeg_clean, lr_norm_invs)

hr_hold_invs = get_trial_intervals(hr_eeg_clean, 1.0, 2.0) 
hr_repeat_invs = get_trial_intervals(hr_eeg_clean, 3.0, 4.0)
hr_norm_invs = get_norm_intervals(hr_eeg_clean, hr_hold_invs + hr_repeat_invs)
trials_repeated = extract_trials_from_intervals(hr_eeg_clean, hr_repeat_invs)
trials_norm_hr, g_hr = sample_norm_windows(hr_eeg_clean, hr_norm_invs)

print(f"Extracted Left Clench: {len(trials_left)} trials")
print(f"Extracted Right Clench: {len(trials_right)} trials")
print(f"Extracted Repeated Clench: {len(trials_repeated)} trials")
print(f"Extracted Norm (LR Dataset): {len(trials_norm_lr)} windows")
print(f"Extracted Norm (HR Dataset): {len(trials_norm_hr)} windows")


# ### Step 3: Exploratory Data Visualization
# 
# Let's look at one trial from each of the 4 independent categories to visualize what the raw active signals look like.

# In[ ]:


def plot_trial_channels(trial_df, title):
    if trial_df is None or len(trial_df) == 0:
        return

    channel_cols = [c for c in trial_df.columns if c.startswith('Channel_')]
    fig, axes = plt.subplots(len(channel_cols), 1, figsize=(12, 2 * len(channel_cols)), sharex=True)
    fig.suptitle(title, fontsize=16)
    time_axis = np.arange(len(trial_df))
    for i, col in enumerate(channel_cols):
        axes[i].plot(time_axis, trial_df[col], linewidth=0.5)
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Samples from start of trial')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

if len(trials_left) > 0:
    plot_trial_channels(trials_left[0], "Left Clench - Trial 1 Raw Data")
if len(trials_norm_lr) > 0:
    plot_trial_channels(trials_norm_lr[0], "Norm (Baseline) - Sample Window Raw Data")


# ### Step 4: Digital Signal Processing (DSP) Filters
# 
# To build a model that separates the movements, we first need to clean the environmental noise from the biological signal.
# Since ALL of our data is EEG recorded from the scalp, we will apply the exact same filters to both datasets:
# 1. **Notch Filter (60Hz)**: Removes the standard alternating current (power-line) humming noise that the environment generates.
# 2. **Bandpass Filter**: Keeps only the typical EEG frequencies (0.5Hz to 50Hz).
# 
# Let's write a function using `scipy.signal` to apply these filters, and then plot the cleaned signals.

# In[ ]:


from scipy.signal import iirnotch, butter, filtfilt

fs = 250.0
nyq = 0.5 * fs

def apply_filters(df):
    filtered_df = df.copy()
    channel_cols = [c for c in df.columns if c.startswith('Channel_')]
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    b_band, a_band = butter(N=2, Wn=[0.5/nyq, 50.0/nyq], btype='band')

    for col in channel_cols:
        sig = filtered_df[col].values
        sig_notch = filtfilt(b_notch, a_notch, sig)
        sig_final = filtfilt(b_band, a_band, sig_notch)
        filtered_df[col] = sig_final
    return filtered_df

filtered_left = apply_filters(trials_left[0]) if trials_left else None
filtered_norm = apply_filters(trials_norm_lr[0]) if trials_norm_lr else None

plot_trial_channels(filtered_left, "FILTERED: Left Clench")
plot_trial_channels(filtered_norm, "FILTERED: Norm (Baseline)")


# ### Step 5: Windowed Feature Extraction (Motor Imagery Focus)
# 
# Since our data consists of 8-second continuous intervals of rhythmic clamping (bursts) and sustained clamping (holds), measured purely via a Cyton EEG Cap (C3, C4, O1, O2, etc.), we need temporal statistics across the whole window.
# 
# We will compute:
# 1. **Variance & RMS**: Captures the overall amplitude energy during the movement.
# 2. **Waveform Length (Wamp)**: Captures the 'complexity' or 'burstiness' of the rhythm.
# 3. **Bandpower**: Captures specific spectral energy (e.g. Mu rhythm [8-12Hz] desynchronization in C3/C4 during motor tasks).
# 
# *Note: Channels 2 and 5 (Cz and Pz) were dropped due to railing, so our primary Motor Cortex indicators will be C3 (Channel 1) and C4 (Channel 3).*

# In[ ]:


from scipy.signal import welch

def extract_features(trial_df):
    channels = [c for c in trial_df.columns if c.startswith('Channel_')]
    features = {}
    for ch in channels:
        data = trial_df[ch].values
        features[f'{ch}_Var'] = np.var(data)
        features[f'{ch}_RMS'] = np.sqrt(np.mean(data**2))
        features[f'{ch}_WL'] = np.sum(np.abs(np.diff(data)))
        freqs, psd = welch(data, fs=250.0, nperseg=250)
        mu_band = np.logical_and(freqs >= 8, freqs <= 12)
        beta_band = np.logical_and(freqs >= 13, freqs <= 30)
        features[f'{ch}_Mu_Power'] = np.sum(psd[mu_band])
        features[f'{ch}_Beta_Power'] = np.sum(psd[beta_band])
    return features

master_features = []
master_labels = []
group_ids = []
current_group = 0

def process_and_append(trials_list, label, is_norm=False, group_mapping=None):
    global current_group
    for i, trial in enumerate(trials_list):
        filt_trial = apply_filters(trial)
        feats = extract_features(filt_trial)
        master_features.append(feats)
        master_labels.append(label)
        if is_norm and group_mapping:
            # Shift group index globally to prevent overlap across HR/LR datasets
            group_ids.append(current_group + group_mapping[i])
        else:
            # Active trials are completely independent events, they get unique groups
            offset = max(group_mapping) + 1 if group_mapping else 0
            group_ids.append(current_group + offset + i)

# Must be built sequentially to increment current_group correctly
process_and_append(trials_norm_lr, 'Norm', True, g_lr)
current_group = max(group_ids) + 1
process_and_append(trials_norm_hr, 'Norm', True, g_hr)
current_group = max(group_ids) + 1
process_and_append(trials_left, 'Left')
current_group = max(group_ids) + 1
process_and_append(trials_right, 'Right')
current_group = max(group_ids) + 1
process_and_append(trials_repeated, 'Repeated')

df_ml = pd.DataFrame(master_features)
df_ml['Label'] = master_labels
groups_array = np.array(group_ids)

print(f"Constructed Feature Dataset Shape: {df_ml.shape}")
print("Distinct Stratified Groups for Validation:", len(np.unique(groups_array)))


# In[ ]:


# Let's visualize the differences across the 4 classes.
# This confirms that both our Motor Cortex expectations and Norm baseline expectations are visible.
print("--- Average Feature Values per Class ---")
print(df_ml.groupby('Label')[['Channel_1_Mu_Power', 'Channel_3_Mu_Power', 'Channel_1_WL', 'Channel_7_WL']].mean())


# ### Step 5: Data Preparation for Machine Learning
# 
# We will split the DataFrame into our feature matrix `X` and target arrays `y`.
# - `y_multi`: The 4-way targets (Norm, Left, Right, Repeated).
# - `y_bin`: The binary target ('Norm' vs 'Active' [Left+Right+Repeated]).
# 
# We also add an index (GroupID) to use in `GroupKFold`. This prevents baseline windows extracted sequentially from the same rest interval from leaking across the train/test barrier (which would artificially inflate the baseline accuracy). 

# In[ ]:


from sklearn.model_selection import StratifiedGroupKFold

X = df_ml.drop(columns=['Label']).values
y_multi = df_ml['Label'].values

# Create binary target (Norm vs Active)
y_bin = np.array(['Norm' if label == 'Norm' else 'Active' for label in y_multi])

groups = groups_array # Pulled from previous cell

print(f"X Shape: {X.shape}")
print("y_bin Distribution:", np.unique(y_bin, return_counts=True))
print("Number of distinct groups for CV:", len(np.unique(groups)))


# ### Step 6: Binary Intent Detector (Norm vs Active)
# 
# We will evaluate 4 algorithms using a 5-Fold Stratified Group CV:
# 1. **LDA (Linear Discriminant Analysis)**: A standard BCI baseline.
# 2. **Logistic Regression**
# 3. **SVM (RBF Kernel)**: Handles non-linear separation well.
# 4. **Random Forest**: An ensemble method that provides native feature importances.
# 
# We will wrap each model in an `sklearn.pipeline.Pipeline` with a `StandardScaler` to prevent information leakage during cross-validation.

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define models
models = {
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

# Results storage
results = {}
cv = StratifiedGroupKFold(n_splits=5)

for name, clf in models.items():
    # Create a pipeline to ensure scaling happens purely on the training folds
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])

    # Run CV to get scoring metrics
    cv_results = cross_validate(pipeline, X, y_bin, groups=groups, cv=cv, 
                                scoring=['accuracy', 'f1_macro'], return_train_score=False)

    # Get predictions out-of-fold for Confusion Matrix
    y_pred = cross_val_predict(pipeline, X, y_bin, groups=groups, cv=cv)

    results[name] = {
        'acc_mean': cv_results['test_accuracy'].mean(),
        'acc_std': cv_results['test_accuracy'].std(),
        'f1_mean': cv_results['test_f1_macro'].mean(),
        'predictions': y_pred
    }

# Print Report
print("--- Binary Classifier CV Results (Norm vs Active) ---")
for name, res in results.items():
    print(f"{name}:")
    print(f"  Accuracy: {res['acc_mean']:.3f} (+/- {res['acc_std']:.3f})")
    print(f"  Macro F1: {res['f1_mean']:.3f}\n")

# Plot Confusion Matrices side-by-side
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_bin, res['predictions'], labels=['Active', 'Norm'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, print_labels=['Active', 'Norm'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(name)
plt.tight_layout()
plt.show()


# ### Step 7: Feature Importances
# 
# Now we will run **Permutation Importance** on our best performing models to see which of our 30 statistical features was most responsible for driving the model's accuracy. 
# 
# For the **Random Forest** algorithm, we can safely pull the native `feature_importances_` attribute.

# In[ ]:


import pandas as pd
from sklearn.inspection import permutation_importance

# Re-train the RF model on the full data simply to extract the total feature importances for reporting
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])
rf_pipeline.fit(X, y_bin)

# Extract RF Native importances
importances = rf_pipeline.named_steps['classifier'].feature_importances_
rf_feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
rf_feat_df = rf_feat_df.sort_values(by='Importance', ascending=False).head(10)

print("--- Top 10 Features (Random Forest Native) ---")
print(rf_feat_df.to_string(index=False))

# Let's ALSO run Permutation Importance on the Logistic Regression model, 
# simulating dropping features to measure "damage" done to the accuracy score.
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
lr_pipeline.fit(X, y_bin)

perm_import = permutation_importance(lr_pipeline, X, y_bin, n_repeats=10, random_state=42)
perm_feat_df = pd.DataFrame({'Feature': X.columns, 'Importance_Mean': perm_import.importances_mean})
perm_feat_df = perm_feat_df.sort_values(by='Importance_Mean', ascending=False).head(10)

print("\n--- Top 10 Features (Logistic Regression Permutation) ---")
print(perm_feat_df.to_string(index=False))


# ### Step 8: Next Steps - Left vs Right Classification (Optional Focus Model)
# 
# As a quick test, let's filter out the Norm and Repeated chunks, and test if the Random Forest can strictly distinguish between a Left Clench and a Right Clench with the available motor cortex channels.

# In[ ]:


# Filter the dataframe down for Left/Right separation only
df_lr_only = df_ml[df_ml['Label'].isin(['Left', 'Right'])]
X_lr = df_lr_only.drop(columns=['Label'])
y_lr = df_lr_only['Label'].values

# Let's use a standard StratifiedKFold since we aren't pulling from contiguous baseline chunks for this
from sklearn.model_selection import StratifiedKFold
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

cv_res_lr = cross_validate(rf_clf, X_lr, y_lr, cv=cv_lr, scoring=['accuracy'])

print("--- Left vs Right Random Forest Classifier ---")
print(f"Accuracy: {cv_res_lr['test_accuracy'].mean():.3f} (+/- {cv_res_lr['test_accuracy'].std():.3f})")

