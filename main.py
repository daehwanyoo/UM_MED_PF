import pandas as pd
import os
import shutil
import random

# Load the CSV files
train_df = pd.read_csv('osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('osic-pulmonary-fibrosis-progression/test.csv')
submission_df = pd.read_csv('osic-pulmonary-fibrosis-progression/sample_submission.csv')

# Get the patient identifiers that are already in test.csv
existing_test_patients = test_df['Patient'].unique()

# Select 5 unique patient identifiers from train.csv that are not in test.csv
new_test_patients = train_df.loc[~train_df['Patient'].isin(existing_test_patients), 'Patient'].unique()
selected_patients = pd.Series(new_test_patients).sample(5, random_state=1).tolist()

# Select one random row for each of the selected patients
selected_rows = []
for patient in selected_patients:
    patient_rows = train_df[train_df['Patient'] == patient]
    selected_rows.append(patient_rows.sample(n=1, random_state=1))

# Concatenate the selected rows into a new DataFrame and overwrite test.csv
new_test_df = pd.concat(selected_rows)
new_test_df.to_csv('osic-pulmonary-fibrosis-progression/test.csv', index=False)

# Path setup
train_path = 'osic-pulmonary-fibrosis-progression/train'
test_path = 'osic-pulmonary-fibrosis-progression/test'

# Delete all folders in the test directory
for folder in os.listdir(test_path):
    folder_path = os.path.join(test_path, folder)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

# Copy the new test patient folders from train to test
for patient_id in selected_patients:
    src_folder = os.path.join(train_path, patient_id)
    dest_folder = os.path.join(test_path, patient_id)
    if os.path.exists(src_folder):
        shutil.copytree(src_folder, dest_folder)

# Generate sample_submission.csv for the new test patients across the specified week range
new_submission_rows = []
week_range = range(-12, 134)  # From week -12 to 133
for week in week_range:
    for patient in selected_patients:
        new_submission_rows.append({
            'Patient_Week': f"{patient}_{week}",
            'FVC': 2000,  # Example value, adjust as necessary
            'Confidence': 100  # Example value, adjust as necessary
        })

# Create a DataFrame from the new rows and overwrite sample_submission.csv
new_submission_df = pd.DataFrame(new_submission_rows)
new_submission_df.to_csv('osic-pulmonary-fibrosis-progression/sample_submission.csv', index=False)

print("Test CSV updated, folders in test directory updated, and sample_submission.csv modified.")
