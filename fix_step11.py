import json

notebook_path = '/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/data_cleaning_analysis.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Find step 11 code cell
# It's the last cell
last_cell = nb['cells'][-1]
source = last_cell['source']

new_source = []
for line in source:
    if "feats_norm_lr, labels_norm_lr" in line:
        new_source.extend([
            "def sliding_window_extraction_grouped_v2(trials_list, label, start_group_id, hop_size):\n",
            "    features_list = []\n",
            "    labels_list = []\n",
            "    group_ids = []\n",
            "    current_group = start_group_id\n",
            "    for trial in trials_list:\n",
            "        filtered_trial = apply_filters(trial)\n",
            "        trial_length = len(filtered_trial)\n",
            "        max_start = trial_length - window_size\n",
            "        start_idx = 0\n",
            "        while start_idx <= max_start:\n",
            "            window_df = filtered_trial.iloc[start_idx:start_idx + window_size]\n",
            "            feats = extract_features(window_df)\n",
            "            features_list.append(feats)\n",
            "            labels_list.append(label)\n",
            "            group_ids.append(current_group)\n",
            "            start_idx += hop_size\n",
            "        current_group += 1\n",
            "    return features_list, labels_list, group_ids, current_group\n",
            "\n",
            "hop_norm = int(1.0 * fs) # 0% overlap for baseline\n",
            "hop_active = int(0.25 * fs)\n",
            "feats_norm_lr, labels_norm_lr, groups_norm_lr, next_id = sliding_window_extraction_grouped_v2(trials_norm_lr, 'Norm', next_id, hop_norm)\n"
        ])
    elif "feats_norm_hr, labels_norm_hr" in line:
        new_source.append("feats_norm_hr, labels_norm_hr, groups_norm_hr, next_id = sliding_window_extraction_grouped_v2(trials_norm_hr, 'Norm', next_id, hop_norm)\n")
    elif "feats_repeated, labels_repeated" in line:
        new_source.append("feats_repeated, labels_repeated, groups_repeated, next_id = sliding_window_extraction_grouped_v2(trials_repeated, 'Repeated', next_id, hop_active)\n")
    else:
        new_source.append(line)

last_cell['source'] = new_source
last_cell['outputs'] = []
last_cell['execution_count'] = None

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)
