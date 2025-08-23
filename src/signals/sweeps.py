from itertools import product
import pandas as pd
import numpy as np

def param_grid(grid_dict: dict[str, list]) -> list[dict]:
    keys = list(grid_dict.keys())
    return [dict(zip(keys, vals)) for vals in product(*[grid_dict[k] for k in keys])]