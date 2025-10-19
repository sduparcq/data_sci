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
    full_items = ",".join(l[2::])
    loc_dict['full_items'] = full_items
    data += [loc_dict]

df = pd.DataFrame(data)


#region MEAN TIME PER ACTION 
def compute_mt(col):
    times = re.findall(r"\bt\d+", col)
    times_int = [int(times[i][1::]) for i in range(len(times))]
    if times_int != []:
        return sum(times_int) / len(times_int)
    else: 
        return 0
#endregion

#region TIME SPENT ON WEBSITE
def compute_ts(col):
    times = re.findall(r"\bt\d+", col)
    times_int = [int(times[i][1::]) for i in range(len(times))]
    return sum(times_int)
#endregion


#region MAPPING

mapping = {
    'Google Chrome': 0,
    'Microsoft Edge': 1,
    'Opera': 2,
    'Firefox': 3
}


df['nav'] = df['nav'].map(mapping)

#endregion

#region NORMALISATION
def normalize(df):
    for col_name in norm_columns:
        df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()

#endregion

#region COUNTING FEATURES

def pattern_ecran(item: str):
    pattern_ecran = re.compile(r"\((.*?)\)")
    return pattern_ecran.findall(item)

def conf_ecran(item: str):
    pattern_conf_ecran = re.compile(r"<(.*?)>")
    return pattern_conf_ecran.findall(item)

def pattern_chaine(item: str):
    pattern_chaine = re.compile(r"\$(.*?)\$")
    return pattern_chaine.findall(item)

def recup_col_chaine(df):
    col_set = set()
    fi = df['full_items']
    for el in fi:
        l = el.split(',')
        for ell in l:
            q = pattern_chaine(ell)
            if q != []:
                for a in q:
                    col_set.add(a)
    return col_set

col_chaine = recup_col_chaine(df)

def count_word(x, col_name):
    return x.count(col_name)

for col_name in col_chaine:
    df[col_name] = df['full_items'].apply(lambda x: count_word(x, col_name))

print(df[col_chaine.pop()])

df['conf_ecran'] = df['full_items'].apply(conf_ecran)
df['pattern_ecran'] = df['full_items'].apply(pattern_ecran)

print(df.head(-1))

#endregion

## variable vitesse

#region CREATE FEATURES / NORMALIZE
norm_columns = ['mean_time', 'time_spent']
df["mean_time"] = df["full_items"].apply(compute_mt)
df["time_spent"] = df["full_items"].apply(compute_ts)

normalize(df)
#endregion

print(df.head(-1))