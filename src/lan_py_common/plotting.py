from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .df import *

Scatter2Df = Df
ColRenameDict = Dict[str, str]

def to_scatter2_df(df: Df, col_x: str, col_y: str) -> Result[Tuple[Scatter2Df, ColRenameDict], CreateDfErr]:
    new_to_old = {"x": col_x, "y": col_y}
    old_to_new = {v: k for k, v in new_to_old.items()}
    return Ok(
        df
            .map_schema(
                DfJsonSchema.from_dict({
                    'ty': 'Scatter2Df',
                    'x': 'float',
                    'y': 'float',
                }).unwrap(),
                old_to_new
            )
            .and_then(lambda df: (df, new_to_old))
    )

LabeledScatter2Df = Df

def to_labeled_scatter2_df(df: Df, col_x: str, col_y: str, col_label: str) -> Result[Tuple[LabeledScatter2Df, ColRenameDict], CreateDfErr]:
    new_to_old = {"x": col_x, "y": col_y, "label": col_label}
    old_to_new = {v: k for k, v in new_to_old.items()}
    return Ok(
        df
            .map_schema(
                DfJsonSchema.from_dict({
                    'ty': 'Scatter2Df',
                    'x': 'float',
                    'y': 'float',
                    'label': 'str',
                }).unwrap(),
                old_to_new
            )
            .and_then(lambda df: (df, new_to_old)),
    )

def alt_scatter2(df: Scatter2Df, new_to_old_col: ColRenameDict, groupby_method: str = "mean"):
    import altair as alt

    supported_methods = {"mean", "sum", "min", "max", "median"}
    if groupby_method not in supported_methods:
        raise ValueError(f"Unsupported groupby method: {groupby_method}")

    grouped_df = (
        df.df
        .groupby("x", as_index=False)
        .agg({"y": groupby_method})
    )

    return alt.Chart(grouped_df).mark_point().encode(
        x=alt.X("x", title=new_to_old_col["x"]),
        y=alt.Y("y", title=new_to_old_col["y"])
    )


def alt_colored_scatter2(df: LabeledScatter2Df, new_to_old_col: ColRenameDict, groupby_method: str = "mean"):
    import altair as alt

    supported_methods = {"mean", "sum", "min", "max", "median"}
    if groupby_method not in supported_methods:
        raise ValueError(f"Unsupported groupby method: {groupby_method}")

    grouped_df = (
        df.df
        .groupby(["x", "label"], as_index=False)
        .agg({"y": groupby_method})
    )

    return alt.Chart(grouped_df).mark_point().encode(
        x=alt.X("x", title=new_to_old_col["x"]),
        y=alt.Y("y", title=new_to_old_col["y"]),
        color=alt.Color("label:N", title=new_to_old_col["label"])
    )