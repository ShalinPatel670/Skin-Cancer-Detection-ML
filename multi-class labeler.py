import csv

original_csv_path = r"C:\Users\16145\OneDrive\Desktop\skin_ml\540 metadata labels no header 1.csv"
new_csv_path = r"C:\Users\16145\OneDrive\Desktop\skin_ml\metadata_labels_multiclass_numeric.csv"

# Map class names to numbers
class_mapping = {
    'basal cell carcinoma': 0,
    'melanoma': 1,
    'squamous cell carcinoma': 2
}

# Open original CSV file and create new CSV file
with open(original_csv_path, 'r') as original_file, open(new_csv_path, 'w', newline='') as new_file:
    csv_reader = csv.reader(original_file)
    csv_writer = csv.writer(new_file)

    for row in csv_reader:
        # Assuming the diagnosis is in the first column
        diagnosis = row[0].strip()
        numeric_diagnosis = class_mapping.get(diagnosis, -1)  # -1 if diagnosis is not found
        new_row = [numeric_diagnosis] + row[1:]  # Keep other columns unchanged
        csv_writer.writerow(new_row)


print("New CSV file with numeric diagnosis labels created successfully.")
