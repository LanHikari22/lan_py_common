from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib as mpl
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .df import *

ColRenameDict = Dict[str, str]

# Porting some of these from calc_dated_lib plotting


@dataclass
class Scatter2Df:
    df: Df
    new_to_old: ColRenameDict


def to_scatter2_df(df: Df, col_x: str, col_y: str) -> Result[Scatter2Df, CreateDfErr]:
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
            .and_then(lambda df: Scatter2Df(df, new_to_old))
    )


@dataclass
class LabeledScatter2Df:
    df: Df
    new_to_old: ColRenameDict


def to_labeled_scatter2_df(df: Df, col_x: str, col_y: str, col_label: str) -> Result[LabeledScatter2Df, CreateDfErr]:
    new_to_old = {"x": col_x, "y": col_y, "label": col_label}
    old_to_new = {v: k for k, v in new_to_old.items()}
    return Ok(
        df
            .map_schema(
                DfJsonSchema.from_dict({
                    'ty': 'LabeledScatter2Df',
                    'x': 'float',
                    'y': 'float',
                    'label': 'str',
                }).unwrap(),
                old_to_new
            )
            .and_then(lambda df: LabeledScatter2Df(df, new_to_old)),
    )

def scatter2_df_aggregate(df: Scatter2Df, groupby_method: str = "mean") -> Result[pd.DataFrame, str]:
    supported_methods = {"mean", "sum", "min", "max", "median", "none"}
    if groupby_method not in supported_methods:
        return Err(f"Unsupported groupby method: {groupby_method}")
    
    grouped_df = (
        df.df.df
        .groupby("x", as_index=False)
        .agg({"y": groupby_method})
            if groupby_method != "none" else
        df.df.df
    )

    return Ok(grouped_df)
    
def labeled_scatter2_df_aggregate(df: LabeledScatter2Df, groupby_method: str = "mean") -> Result[pd.DataFrame, str]:
    supported_methods = {"mean", "sum", "min", "max", "median", "none"}
    if groupby_method not in supported_methods:
        return Err(f"Unsupported groupby method: {groupby_method}")
    
    grouped_df = (
        df.df.df
        .groupby(["x", "label"], as_index=False)
        .agg({"y": groupby_method})
            if groupby_method != "none" else
        df.df.df
    )

    return Ok(grouped_df)

def alt_scatter2(df: Scatter2Df, groupby_method: str = "mean"):
    import altair as alt

    grouped_df = scatter2_df_aggregate(df, groupby_method).unwrap()

    return alt.Chart(grouped_df).mark_point().encode(
        x=alt.X("x", title=df.new_to_old["x"]),
        y=alt.Y("y", title=df.new_to_old["y"])
    )


def alt_colored_scatter2(df: LabeledScatter2Df, groupby_method: str = "mean"):
    import altair as alt

    grouped_df = labeled_scatter2_df_aggregate(df, groupby_method).unwrap()

    return alt.Chart(grouped_df).mark_point().encode(
        x=alt.X("x", title=df.new_to_old["x"]),
        y=alt.Y("y", title=df.new_to_old["y"]),
        color=alt.Color("label:N", title=df.new_to_old["label"])
    )


def mpl_scatter2(
    df: Scatter2Df, 
    opt_ax: Optional[Axes] = None, 
    opt_point_color: Optional[str] = None, 
    opt_label: Optional[str] = None, 
    groupby_method: str = "mean",
    s: int = 4, 
    polyfit: int = 0, 
    polyfit_stdev: int = 2, 
    polyfit_legend: str = 'None'):
    """
    Plots a scatter plot of the specified x and y columns from a DataFrame.
    
    Parameters:
    ax (matplotlib.axes.Axes, optional): The axes on which to plot the graph. If None, a new figure is created.
    color (str, optional): The color of the scatter points.
    label (str, optional): The label for the scatter points.
    s (int, optional): The size of the scatter points.
    polyfit (int, optional): The degree of the polynomial fit to perform. 1 for linear, 2 for quadratic, etc.
    polyfit_legend (str, optional): 'None', 'Text', 'Graph', 'Both'. How legend of coefficients is shown.
    """


    # Create a new figure if no axes are provided
    opt_fig = None
    if opt_ax is None:
        opt_fig, opt_ax = plt.subplots(figsize=(4, 3))
    ax = opt_ax

    df_agg = scatter2_df_aggregate(df, groupby_method).unwrap()

    ax.scatter(df_agg["x"], df_agg["y"], alpha=0.5, s=s, color=opt_point_color, label=opt_label)  # Creates the scatter plot on the specified ax
    ax.set_title(f'Scatter Plot of {df.new_to_old["x"]} vs {df.new_to_old["y"]}')  # Adds a title to the plot
    ax.set_xlabel(df.new_to_old["x"])  # Labels the x-axis
    ax.set_ylabel(df.new_to_old["y"])  # Labels the y-axis
    ax.grid(True)  # Adds a grid

    if polyfit != 0:
        # Perform polynomial interpolation
        coefficients = np.polyfit(df_agg["x"], df_agg["y"], polyfit)
        poly_fit = np.poly1d(coefficients)

        # Generate x values for the polynomial fit line
        x_fit = np.linspace(df_agg["x"].min(), df_agg["x"].max(), 100)
        y_fit = poly_fit(x_fit)

        # Choose a different color for the polynomial fit line
        line_color = (
            'blue' 
                if opt_point_color == 'red' else 
            'red'
        )

        # Plot the polynomial fit line
        ax.plot(x_fit, y_fit, color=line_color, label=f'{polyfit}-degree fit')

        # Calculate the residuals and standard deviation
        residuals = df_agg["y"] - poly_fit(df_agg["x"])
        std_dev = np.std(residuals)

        # Plot lines at the calculated standard deviation above and below the fit line
        ax.plot(x_fit, y_fit + polyfit_stdev*std_dev, color='grey', linestyle='--', alpha=0.5)
        ax.plot(x_fit, y_fit - polyfit_stdev*std_dev, color='grey', linestyle='--', alpha=0.5)

        # Format the coefficients for display
        coeff_str = (
            f'{coefficients[-1]:.2f} + ' + ' + '
                .join([f'{coeff:.2f}x^{i}' for i, coeff in enumerate(coefficients[-2::-1], start=1)])
        )
        textstr = f'Polyfit: {coeff_str}\nStd Dev: {std_dev:.2f}'

        if polyfit_legend == 'Text' or polyfit_legend == 'Both':
            print(textstr)

        if polyfit_legend == 'Graph' or polyfit_legend == 'Both':
            # Place the text box in the upper left corner
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

    if opt_fig is not None:
        plt.legend()
        plt.show()