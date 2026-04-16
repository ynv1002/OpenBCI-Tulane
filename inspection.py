import pandas as pd

eeg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-2-27-26-(01).csv'
emg_file = '/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN/HR-2-27-26-(01).csv'

eeg_df = pd.read_csv(eeg_file, header=None, sep='\t')
emg_df = pd.read_csv(emg_file, header=None, sep='\t')

def format_openbci_df(df):
    num_cols = df.shape[1]
    col_names = [f'col_{i}' for i in range(num_cols)]
    col_names[0] = 'Sample_Index'
    for i in range(1, 9):
        col_names[i] = f'Channel_{i}'
    col_names[-1] = 'Marker'
    df.columns = col_names
    relevant_cols = ['Sample_Index'] + [f'Channel_{i}' for i in range(1, 9)] + ['Marker']
    return df[relevant_cols]

eeg_df = format_openbci_df(eeg_df)
emg_df = format_openbci_df(emg_df)

print("Marker value counts for EEG:")
print(eeg_df['Marker'].value_counts())

print("\nMarker value counts for EMG:")
print(emg_df['Marker'].value_counts())

def check_flat_or_railed(df, dataset_name):
    print(f"\n--- {dataset_name} Channel Checks ---")
    channels = [f'Channel_{i}' for i in range(1, 9)]
    for ch in channels:
        ch_data = df[ch]
        std_dev = ch_data.std()
        min_val = ch_data.min()
        max_val = ch_data.max()
        if std_dev == 0:
            print(f"WARNING: {ch} is completely FLATLINED (val: {min_val})")
        elif max_val >= 187000 or min_val <= -187000:
            print(f"WARNING: {ch} is RAILED (min {min_val}, max {max_val})")
        else:
            print(f"OK: {ch} (min: {min_val:.2f}, max: {max_val:.2f}, std: {std_dev:.2f})")

check_flat_or_railed(eeg_df, "EEG Data")
check_flat_or_railed(emg_df, "EMG Data")
