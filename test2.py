import pandas as pd
import re

file_path = "./data/train.csv"

with open(file_path, "r", encoding="utf-8") as f: 
    lines = f.readlines()

data = []

for i, line in enumerate(lines): 
    l = line.split(',')
    loc_dict = {}
    loc_dict['name'] = l[0]
    loc_dict['nav'] = l[1]
    fl = l[1::]
    full_items = "".join(l[2::])
    loc_dict['full_items'] = full_items
    data += [loc_dict]

df = pd.DataFrame(data)

print(df.head(-1))

## MEAN TIME PER ACTION ##
def compute_mt(col):
    times = re.findall(r"\bt\d+", col)
    times_int = [int(times[i][1::]) for i in range(len(times))]
    if times_int != []:
        return sum(times_int) / len(times_int)
    else: 
        return 0


## TIME SPENT ON WEBSITE ##
def compute_ts(col):
    times = re.findall(r"\bt\d+", col)
    times_int = [int(times[i][1::]) for i in range(len(times))]
    return sum(times_int)



df["mean_time"] = df["full_items"].apply(compute_mt)
df["time_spent"] = df["full_items"].apply(compute_ts)

print(df.head(-1))