import pandas as pd

def parquet_to_csv(parquet_file_path, csv_file_path):
    df = pd.read_parquet(parquet_file_path)
    df.to_csv(csv_file_path, index=False)


parquet_file_path = 'kaggle/kaggle_grealish/answer/submission.parquet'
csv_file_path = 'kaggle/kaggle_grealish/answer/submission.csv'

parquet_to_csv(parquet_file_path, csv_file_path)

print(f"Parquet file has been successfully converted to CSV and saved as {csv_file_path}")
