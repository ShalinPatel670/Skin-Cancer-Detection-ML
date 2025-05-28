import pandas as pd

data = pd.read_csv(r"C:\Users\16145\OneDrive\Desktop\skin_ml\metadata540_onehot.csv")

categorical_columns = data.columns[1:-2]  # Exclude 'isic_id', 'sex', and 'diagnosis'

# One-hot vector encoding
for column in categorical_columns:
    unique_values = data[column].unique()
    binary_columns = [f'{column}_{value}' for value in unique_values]
    binary_data = pd.DataFrame(0, columns=binary_columns, index=data.index)
    
    for value in unique_values:
        binary_data[f'{column}_{value}'] = (data[column] == value).astype(int)
    
    data = pd.concat([data, binary_data], axis=1)

data['Sex'] = (data['sex'] == 'female').astype(int)
data.drop('sex', axis=1, inplace=True)

data['Diagnosis'] = (data['diagnosis'] == 'melanoma').astype(int)
data.drop('diagnosis', axis=1, inplace=True)

data.drop(categorical_columns, axis=1, inplace=True)

data.to_csv(r"C:\Users\16145\OneDrive\Desktop\skin_ml\preprocessed_onehotvectors_data.csv", index=False)

print(data)
