import torch
import pandas as pd
from inference import load_model, predict

model = load_model()
df = pd.read_csv("data/test.csv")
df_submission = pd.DataFrame(columns=["ImageId", "Label"])
device = "cuda" if torch.cuda.is_available() else "cpu"

for i, sample in enumerate(df.values):
    sample = torch.tensor(sample).reshape(1, 28, 28).to(device=device, dtype=torch.float32)
    label = predict(model, sample)
    df_submission = df_submission.append({'ImageId': (i+1), 'Label': label}, ignore_index=True)


df_submission.to_csv('submission.csv', index=False)