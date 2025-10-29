import functools
import hashlib
import pandas as pd

def hashable(obj):
    if isinstance(obj, pd.DataFrame):
        return hashlib.md5(pd.util.hash_pandas_object(obj, index=True).values.tobytes()).hexdigest()
    elif isinstance(obj, (list, tuple)):
        return tuple(hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, hashable(v)) for k, v in obj.items()))
    else:
        return obj

class Memoizer:
    def __init__(self, name=None):
        self.cache = {}
        self.name = name or "default"

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (hashable(args), hashable(kwargs))
            if key in self.cache:
                print("using the cache")
                return self.cache[key]
            result = func(*args, **kwargs)
            self.cache[key] = result
            return result
        return wrapper

    def clear(self):
        self.cache.clear()

    def size(self):
        return len(self.cache)

    def get_cache(self):
        return dict(self.cache)
