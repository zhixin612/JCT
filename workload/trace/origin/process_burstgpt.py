import pandas as pd

files = ['BurstGPT_without_fails_1.csv', 'BurstGPT_without_fails_2.csv']

for t, file in zip(range(2), files):
    data = pd.read_csv(file)
    # Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
    gaps = data['Timestamp'].diff().dropna()
    # keep 1~10000 values
    gaps = gaps[gaps < 10000]
    gaps = gaps[gaps > 1]
    print(1000/gaps.mean(), gaps.size, gaps.max(), gaps.min())
    gaps.to_csv('../burstgpt_{}.csv'.format(t), index=False, header=False)
