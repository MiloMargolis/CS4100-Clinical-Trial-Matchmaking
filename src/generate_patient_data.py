from faker import Faker
import pandas as pd
import random

fake = Faker()
conditions = ["lung cancer", "breast cancer", "prostate cancer"]

records = []
for _ in range(50):
    record = {
        "PatientID": fake.unique.uuid4(),
        "Age": random.randint(18, 85),
        "Sex": random.choice(["Male", "Female"]),
        "Condition": random.choice(conditions),
        "Keywords": fake.word()
    }
    records.append(record)

df = pd.DataFrame(records)
df.to_csv("data/patient_data.csv", index=False)
