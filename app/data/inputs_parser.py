import pandas as pd
import json

df = pd.read_excel('raw_inputs.xls')

trials_per_subject = 648

df["participant"] = df.index // trials_per_subject

print(df.head())

participants = []

for i in range(0, len(df), 648):
    participants.append(df.iloc[i:i+648])

# Example: participant 1
p1 = participants[0]

participants = {
    i+1: df.iloc[i*648:(i+1)*648]
    for i in range(len(df)//648)
}

sequences = []

for participant, data in participants.items():
    for i in range(3):
        sequence_df = data.iloc[i*3:(i*3)+3]
        sequence = sequence_df['ResponseLabel'].to_list()
        sequences.append(sequence)

with open('extracted_sequences.json', 'w') as fp:
    json.dump({'extracted_sequences': sequences}, fp)
    