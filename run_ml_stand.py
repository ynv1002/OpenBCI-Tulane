import pandas as pd
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, welch
import warnings
warnings.filterwarnings('ignore')

eeg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-2-27-26-(01).csv'
hr_eeg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN/HR-2-27-26-(01).csv'

eeg_df = pd.read_csv(eeg_file, header=None, sep='\t')
hr_eeg_df = pd.read_csv(hr_eeg_file, header=None, sep='\t')

def format_openbci_df(df):
    col_names = [f'col_{i}' for i in range(df.shape[1])]
    col_names[0], col_names[-1] = 'Sample_Index', 'Marker'
    for i in range(1, 9): col_names[i] = f'Channel_{i}'
    return df[['Sample_Index'] + [f'Channel_{i}' for i in range(1, 9)] + ['Marker']]

eeg_df, hr_eeg_df = format_openbci_df(eeg_df), format_openbci_df(hr_eeg_df)
cols_drop = ['Channel_2', 'Channel_5']
eeg_clean = eeg_df.drop(columns=cols_drop, errors='ignore').reset_index(drop=True)
hr_eeg_clean = hr_eeg_df.drop(columns=cols_drop, errors='ignore').reset_index(drop=True)

def get_trial_intervals(df, start_marker, end_marker):
    intervals, in_trial = [], False
    start_idx = None
    markers = df['Marker'].values
    for i, m in enumerate(markers):
        if m == start_marker and not in_trial:
            start_idx, in_trial = i, True
        elif m == end_marker and in_trial:
            intervals.append((start_idx, i))
            in_trial = False
    return intervals

def get_norm_intervals(df, intervals, buffer=500):
    sorted_invs = sorted(intervals, key=lambda x: x[0])
    norm_intervals, current_idx = [], 0
    for start, end in sorted_invs:
        if start - buffer > current_idx:
            norm_intervals.append((current_idx, start - buffer))
        current_idx = end + buffer
    if len(df) > current_idx:
        norm_intervals.append((current_idx, len(df) - 1))
    return norm_intervals

def sample_norm_windows(df, norm_intervals, window_size=2000):
    norm_trials = []
    # To easily assign groups later, let's keep track of which interval block this came from
    group_ids = []
    
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
trials_norm_lr, g_lr = sample_norm_windows(eeg_clean, lr_norm_invs)
trials_left = [eeg_clean.iloc[s:e+1] for s, e in lr_left_invs]
trials_right = [eeg_clean.iloc[s:e+1] for s, e in lr_right_invs]

hr_hold_invs = get_trial_intervals(hr_eeg_clean, 1.0, 2.0)
hr_repeat_invs = get_trial_intervals(hr_eeg_clean, 3.0, 4.0)
hr_norm_invs = get_norm_intervals(hr_eeg_clean, hr_hold_invs + hr_repeat_invs)
trials_norm_hr, g_hr = sample_norm_windows(hr_eeg_clean, hr_norm_invs)
trials_repeated = [hr_eeg_clean.iloc[s:e+1] for s, e in hr_repeat_invs]

fs = 250.0; nyq = 0.5 * fs
b_notch, a_notch = iirnotch(60.0, 30.0, fs)
b_band, a_band = butter(N=2, Wn=[0.5/nyq, 50.0/nyq], btype='band')

def extract_features(trial_df):
    channels = [c for c in trial_df.columns if c.startswith('Channel_')]
    features = {}
    for ch in channels:
        data = trial_df[ch].values
        sig = filtfilt(b_notch, a_notch, data)
        sig = filtfilt(b_band, a_band, sig)
        
        features[f'{ch}_Var'] = np.var(sig)
        features[f'{ch}_RMS'] = np.sqrt(np.mean(sig**2))
        features[f'{ch}_WL'] = np.sum(np.abs(np.diff(sig)))
        freqs, psd = welch(sig, fs=250.0, nperseg=250)
        
        features[f'{ch}_Mu_Power'] = np.sum(psd[np.logical_and(freqs >= 8, freqs <= 12)])
        features[f'{ch}_Beta_Power'] = np.sum(psd[np.logical_and(freqs >= 13, freqs <= 30)])
    return features

master_features, master_labels, groups = [], [], []
current_group = 0

def add_batch(trials, label, is_norm=False, group_mapping=None):
    global current_group
    for i, trial in enumerate(trials):
        master_features.append(extract_features(trial))
        master_labels.append(label)
        if is_norm and group_mapping:
            # Shift group index so it doesn't overlap across datasets
            groups.append(current_group + group_mapping[i])
        else:
            # Active trials are fully independent, give them unique groups
            groups.append(current_group + i + max(group_mapping) if group_mapping else current_group + i)

add_batch(trials_norm_lr, 'Norm', True, g_lr)
current_group = max(groups) + 1
add_batch(trials_norm_hr, 'Norm', True, g_hr)
current_group = max(groups) + 1
add_batch(trials_left, 'Left')
current_group = max(groups) + 1
add_batch(trials_right, 'Right')
current_group = max(groups) + 1
add_batch(trials_repeated, 'Repeated')

df_ml = pd.DataFrame(master_features)
X = df_ml.values
y_multi = np.array(master_labels)
y_bin = np.array(['Norm' if l == 'Norm' else 'Active' for l in y_multi])
groups = np.array(groups)

print(f"X Shape: {X.shape}")
print(f"y_bin Distribution: {np.unique(y_bin, return_counts=True)}")
print(f"Distinct groups for CV (Norm blocks grouped, Actives independent): {len(np.unique(groups))}")

from sklearn.model_selection import StratifiedGroupKFold, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

models = {
    'LDA': LinearDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
}

cv = StratifiedGroupKFold(n_splits=5)
print("\n--- Binary Classifier CV Results (Norm vs Active) ---")
for name, clf in models.items():
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', clf)])
    cv_res = cross_validate(pipeline, X, y_bin, groups=groups, cv=cv, scoring=['accuracy', 'f1_macro'])
    print(f"{name}:")
    print(f"  Accuracy: {cv_res['test_accuracy'].mean():.3f} (+/- {cv_res['test_accuracy'].std():.3f})")
    print(f"  Macro F1: {cv_res['test_f1_macro'].mean():.3f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
pipe = Pipeline([('scaler', StandardScaler()), ('clf', rf)])
pipe.fit(X, y_bin)
rf_feat_df = pd.DataFrame({'Feature': df_ml.columns, 'Importance': pipe.named_steps['clf'].feature_importances_})
print("\n--- Top 5 Features (RF) ---")
print(rf_feat_df.sort_values(by='Importance', ascending=False).head(5).to_string(index=False))

df_lr_only = df_ml[np.isin(y_multi, ['Left', 'Right'])]
cv_lr = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_res_lr = cross_validate(RandomForestClassifier(n_estimators=100, random_state=42), df_lr_only.values, y_multi[np.isin(y_multi, ['Left', 'Right'])], cv=cv_lr, scoring=['accuracy'])
print("\n--- Left vs Right Classifier ---")
print(f"Accuracy: {cv_res_lr['test_accuracy'].mean():.3f} (+/- {cv_res_lr['test_accuracy'].std():.3f})")
