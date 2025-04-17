from typing import Self
import checkpipe as pipe
from dataclasses import dataclass
import classes
import pandas as pd
from .typeclasses import *

@dataclass
class Df:
    json_schema: str
    df: pd.DataFrame

    @staticmethod
    def new(json_schema: str) -> 'Df':
        return Df(json_schema, pd.DataFrame())

@describe.instance(Df)
def impl_describe_for_Df(_instance: Df) -> str:
    return 'Dfs with more type safety'