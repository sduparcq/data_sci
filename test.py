import pandas as pd
import json

file_path = "./data/train.csv"

with open(file_path, "r", encoding="utf-8") as f: 
    lines = f.readlines()

## CHARGER + DEFINIR LES COLONNES

cols_set = set()

df_dict = {}

for l in lines:
    sep_line = l.split(",")
    for item in sep_line: 
        cols_set.add(item)

with open("columns.json", "w", encoding="utf-8") as f:
    json.dump(list(cols_set), f, ensure_ascii=False, indent=2)


## COLUMNS LOADING f:
#     columns = list(json.load(f))

# print(columns)


# with open("columns.json", "w", encoding="utf-8") as