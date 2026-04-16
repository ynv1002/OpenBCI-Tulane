import pandas as pd
import numpy as np
from scipy.signal import iirnotch, butter, filtfilt, welch

eeg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-2-27-26-(01).csv'
hr_eeg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN/HR-2-27-26-(01).csv'

eeg_df = pd.read_csv(eeg_file, header=None, sep='\t')
hr_eeg_df = pd.read_csv(hr_eeg_file, header=None, sep='\t')

def format_openbci_df(df):
    num_cols = df.shape[1]
    col_names = [f'col_{i}' for i in range(num_cols)]
    col_names[0] = 'Sample_Index'
    for i in range(1, 9): col_names[i] = f'Channel_{i}'
    col_names[-1] = 'Marker'
    df.columns = col_names
    relevant_cols = ['Sample_Index'] + [f'Channel_{i}' for i in range(1, 9)] + ['Marker']
    return df[relevant_cols]

eeg_df = format_openbci_df(eeg_df)
hr_eeg_df = format_openbci_df(hr_eeg_df)

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
    for start, end in norm_intervals:
        duration = end - start
        if duration >= window_size:
            for chunk_start in range(start, end - window_size + 1, window_size):
                norm_trials.append(df.iloc[chunk_start:chunk_start+window_size])
    return norm_trials

lr_left_invs = get_trial_intervals(eeg_clean, 1.0, 2.0)
lr_right_invs = get_trial_intervals(eeg_clean, 3.0, 4.0)
lr_norm_invs = get_norm_intervals(eeg_clean, lr_left_invs + lr_right_invs)

trials_left = extract_trials_from_intervals(eeg_clean, lr_left_invs)
trials_right = extract_trials_from_intervals(eeg_clean, lr_right_invs)
trials_norm_lr = sample_norm_windows(eeg_clean, lr_norm_invs)

hr_hold_invs = get_trial_intervals(hr_eeg_clean, 1.0, 2.0)
hr_repeat_invs = get_trial_intervals(hr_eeg_clean, 3.0, 4.0)
hr_norm_invs = get_norm_intervals(hr_eeg_clean, hr_hold_invs + hr_repeat_invs)

trials_repeated = extract_trials_from_intervals(hr_eeg_clean, hr_repeat_invs)
trials_norm_hr = sample_norm_windows(hr_eeg_clean, hr_norm_invs)

print(f"Left Clench: {len(trials_left)} trials")
print(f"Right Clench: {len(trials_right)} trials")
print(f"Repeated Clench: {len(trials_repeated)} trials")
print(f"Norm (LR Dataset): {len(trials_norm_lr)} windows (~8s each)")
print(f"Norm (HR Dataset): {len(trials_norm_hr)} windows (~8s each)")

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
        filtered_df[col] = filtfilt(b_band, a_band, sig_notch)
    return filtered_df

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

master_features, master_labels = [], []
def process_and_append(trials_list, label):
    for trial in trials_list:
        filt_trial = apply_filters(trial)
        master_features.append(extract_features(filt_trial))
        master_labels.append(label)

process_and_append(trials_norm_lr, 'Norm')
process_and_append(trials_norm_hr, 'Norm')
process_and_append(trials_left, 'Left')
process_and_append(trials_right, 'Right')
process_and_append(trials_repeated, 'Repeated')

df_ml = pd.DataFrame(master_features)
df_ml['Label'] = master_labels

print("\n--- Class Distribution ---")
print(df_ml['Label'].value_counts())

print("\n--- Average Feature Values per Class ---")
print(df_ml.groupby('Label')[['Channel_1_Mu_Power', 'Channel_3_Mu_Power', 'Channel_1_WL', 'Channel_7_WL']].mean())

