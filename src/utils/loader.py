import pandas as pd
import os
from .memoizer import Memoizer


def load_data(path: str):
    cwd = os.getcwd()
    pathh = cwd + path
    with open(pathh, mode='r', encoding="utf-8") as f:
        lines =  f.readlines()
    
    dict_list = []
    
    for line in lines:
        line = line.split(",")
        user = line[0]
        nav = line[1]
        full_items = ",".join(line[2::])
        local_dict = {
            "user": user,
            "nav": nav,
            "full_items": full_items
        }
        dict_list.append(local_dict)

    return pd.DataFrame(dict_list)

