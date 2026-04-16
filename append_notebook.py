import json

notebook_path = '/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/data_cleaning_analysis.ipynb'

with open(notebook_path, 'r') as f:
    text_content = f.read()

# I messed up the JSON with the bash cat, so let's fix it by loading until the first '[' and finding the end manually or just removing the bad append
last_good_part = text_content.split('EOF')[0] # Usually the bad bash left an EOF or malformed string

# Let's just cleanly write out what we know the notebook should be at currently to guarantee we don't break their workspace
