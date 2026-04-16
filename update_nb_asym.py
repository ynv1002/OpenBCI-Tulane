import json

notebook_path = '/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/data_cleaning_analysis.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Optional: skip if already added
has_step_10 = any("Step 10:" in "".join(c.get('source', [])) for c in nb['cells'])
if not has_step_10:
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Step 10: Contralateral Asymmetry Features (Left vs Right)\n",
            "\n",
            "We'll compute asymmetry features (differences and ratio) between the motor cortex channels (C3=Channel_1, C4=Channel_3) to give the model explicit directional cues."
        ]
    }

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Add derived features to df_sw\n",
            "df_asym = df_sw.copy()\n",
            "\n",
            "# Mu asymmetry\n",
            "df_asym['C3_C4_Mu_Diff'] = df_asym['Channel_1_Mu_Power'] - df_asym['Channel_3_Mu_Power']\n",
            "df_asym['C4_C3_Mu_Diff'] = df_asym['Channel_3_Mu_Power'] - df_asym['Channel_1_Mu_Power']\n",
            "df_asym['Mu_Asymmetry_Ratio'] = (df_asym['Channel_1_Mu_Power'] - df_asym['Channel_3_Mu_Power']) / (df_asym['Channel_1_Mu_Power'] + df_asym['Channel_3_Mu_Power'] + 1e-6)\n",
            "\n",
            "# Beta asymmetry\n",
            "df_asym['C3_C4_Beta_Diff'] = df_asym['Channel_1_Beta_Power'] - df_asym['Channel_3_Beta_Power']\n",
            "df_asym['C4_C3_Beta_Diff'] = df_asym['Channel_3_Beta_Power'] - df_asym['Channel_1_Beta_Power']\n",
            "\n",
            "X_asym = df_asym.values\n",
            "\n",
            "print(f\"Augmented Dataset Shape: {df_asym.shape}\")\n",
            "\n",
            "# Retrain models using StratifiedGroupKFold\n",
            "print(\"\\n--- Left vs Right (Augmented Features, Grouped CV) ---\")\n",
            "results_asym = {}\n",
            "for name, clf in models.items():\n",
            "    pipeline = Pipeline([\n",
            "        ('scaler', StandardScaler()),\n",
            "        ('classifier', clf)\n",
            "    ])\n",
            "    \n",
            "    cv_results = cross_validate(pipeline, X_asym, y_sw, groups=groups_sw, cv=cv_sw, scoring=['accuracy', 'f1_macro'])\n",
            "    y_pred = cross_val_predict(pipeline, X_asym, y_sw, groups=groups_sw, cv=cv_sw)\n",
            "    \n",
            "    results_asym[name] = {\n",
            "        'acc_mean': cv_results['test_accuracy'].mean(),\n",
            "        'acc_std': cv_results['test_accuracy'].std(),\n",
            "        'f1_mean': cv_results['test_f1_macro'].mean()\n",
            "    }\n",
            "    \n",
            "    print(f\"{name}:\")\n",
            "    print(f\"  Accuracy: {results_asym[name]['acc_mean']:.3f} (+/- {results_asym[name]['acc_std']:.3f})\")\n",
            "    print(f\"  Macro F1: {results_asym[name]['f1_mean']:.3f}\\n\")\n",
            "\n",
            "# Feature Importances\n",
            "print(\"\\n--- Augmented Feature Importances ---\")\n",
            "rf_pipe = Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])\n",
            "rf_pipe.fit(X_asym, y_sw)\n",
            "rf_importances = rf_pipe.named_steps['classifier'].feature_importances_\n",
            "rf_feat_df = pd.DataFrame({'Feature': df_asym.columns, 'Importance': rf_importances}).sort_values(by='Importance', ascending=False).head(10)\n",
            "print(\"\\nTop 10 Features (Random Forest Native):\")\n",
            "print(rf_feat_df.to_string(index=False))\n",
            "\n",
            "lr_pipe = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])\n",
            "lr_pipe.fit(X_asym, y_sw)\n",
            "lr_perm = permutation_importance(lr_pipe, X_asym, y_sw, n_repeats=10, random_state=42)\n",
            "lr_perm_df = pd.DataFrame({'Feature': df_asym.columns, 'Importance_Mean': lr_perm.importances_mean}).sort_values(by='Importance_Mean', ascending=False).head(10)\n",
            "print(\"\\nTop 10 Features (Logistic Regression Permutation):\")\n",
            "print(lr_perm_df.to_string(index=False))"
        ]
    }

    nb['cells'].extend([markdown_cell, code_cell])

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
