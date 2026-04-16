import json

notebook_path = '/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/data_cleaning_analysis.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if "### Step 9:" in "".join(cell.get('source', [])):
        print("Sliding window code already appended. Skipping.")
        exit(0)

markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Step 9: Advanced Feature Extraction (Sliding Windows)\n",
        "\n",
        "Instead of extracting one set of features per 8s trial, we will extract features using a 1-second sliding window with 75% overlap (hop = 0.25s) for active trials. For the Norm (baseline) data, we will extract 1-second windows with less overlap to prevent redundant data.\n",
        "\n",
        "We'll also create an 'onset-focused' dataset using only windows within the first 2 seconds of the trial."
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import StratifiedKFold, cross_validate\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "fs = 250.0\n",
        "window_size = int(1.0 * fs)  # 250 samples\n",
        "hop_active = int(0.25 * fs)  # 62 samples (approx 75% overlap)\n",
        "hop_norm = int(1.0 * fs)     # 250 samples (0% overlap to avoid redundancy)\n",
        "\n",
        "def sliding_window_extraction(trials_list, label, is_norm=False, active_hop=hop_active, norm_hop=hop_norm, onset_only=False):\n",
        "    features_list = []\n",
        "    labels_list = []\n",
        "    hop = norm_hop if is_norm else active_hop\n",
        "    \n",
        "    for trial in trials_list:\n",
        "        filtered_trial = apply_filters(trial)\n",
        "        trial_length = len(filtered_trial)\n",
        "        \n",
        "        # For onset_only, limit maximum start index.\n",
        "        # We want windows entirely within 0-2s (0-500 samples).\n",
        "        # The window is 250 samples, so start_idx can go up to 250.\n",
        "        max_start = int(1.0 * fs) if onset_only and not is_norm else trial_length - window_size\n",
        "        \n",
        "        start_idx = 0\n",
        "        while start_idx <= max_start:\n",
        "            window_df = filtered_trial.iloc[start_idx:start_idx + window_size]\n",
        "            feats = extract_features(window_df)\n",
        "            features_list.append(feats)\n",
        "            labels_list.append(label)\n",
        "            start_idx += hop\n",
        "            \n",
        "    return features_list, labels_list\n",
        "\n",
        "# Extract sliding windows for Left and Right\n",
        "feats_left, labels_left = sliding_window_extraction(trials_left, 'Left')\n",
        "feats_right, labels_right = sliding_window_extraction(trials_right, 'Right')\n",
        "\n",
        "# Build Full Left/Right Dataset\n",
        "df_sw_lr = pd.DataFrame(feats_left + feats_right)\n",
        "y_sw_lr = np.array(labels_left + labels_right)\n",
        "\n",
        "print(f\"Sliding Window (1s, 75% overlap) Left/Right Shape: {df_sw_lr.shape}\")\n",
        "print(f\"Class Distribution: {np.unique(y_sw_lr, return_counts=True)}\")\n",
        "\n",
        "# Train / Evaluate\n",
        "cv_sw = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
        "rf_pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])\n",
        "res_sw = cross_validate(rf_pipe, df_sw_lr.values, y_sw_lr, cv=cv_sw, scoring=['accuracy'])\n",
        "\n",
        "print(\"\\n--- Left vs Right (1s Sliding Window) ---\")\n",
        "print(f\"Accuracy: {res_sw['test_accuracy'].mean():.3f} (+/- {res_sw['test_accuracy'].std():.3f})\")\n",
        "\n",
        "# Extract Onset-focused windows for Left and Right (0-2s only)\n",
        "feats_left_onset, labels_left_onset = sliding_window_extraction(trials_left, 'Left', onset_only=True)\n",
        "feats_right_onset, labels_right_onset = sliding_window_extraction(trials_right, 'Right', onset_only=True)\n",
        "\n",
        "df_onset_lr = pd.DataFrame(feats_left_onset + feats_right_onset)\n",
        "y_onset_lr = np.array(labels_left_onset + labels_right_onset)\n",
        "\n",
        "print(f\"\\nOnset-Focused (0-2s) Shape: {df_onset_lr.shape}\")\n",
        "res_onset = cross_validate(rf_pipe, df_onset_lr.values, y_onset_lr, cv=cv_sw, scoring=['accuracy'])\n",
        "\n",
        "print(\"--- Left vs Right (1s Sliding Window, Onset 0-2s Only) ---\")\n",
        "print(f\"Accuracy: {res_onset['test_accuracy'].mean():.3f} (+/- {res_onset['test_accuracy'].std():.3f})\")"
    ]
}

nb['cells'].extend([markdown_cell, code_cell])

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
