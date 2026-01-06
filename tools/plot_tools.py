"""
Plotting utilities using matplotlib, plotly, and seaborn.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pandas as pd

from tools.data_tools import get_dataframe
from utils.logging import get_logger

logger = get_logger(__name__)


# Global figure storage
_figures: Dict[str, Any] = {}


# Configure matplotlib defaults
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


def create_matplotlib_plot(
    df_id: str,
    plot_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Create matplotlib plot from DataFrame.

    Args:
        df_id: DataFrame identifier
        plot_type: Type of plot (line, scatter, bar, hist, box, violin)
        x: X-axis column name
        y: Y-axis column name
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        **kwargs: Additional matplotlib parameters

    Returns:
        Figure ID

    Raises:
        KeyError: If df_id or columns don't exist
        ValueError: If plot_type is invalid
    """
    df = get_dataframe(df_id)

    fig, ax = plt.subplots()
    fig_id = f"fig_{uuid.uuid4().hex[:8]}"

    try:
        if plot_type == "line":
            if x and y:
                ax.plot(df[x], df[y], **kwargs)
            elif y:
                ax.plot(df.index, df[y], **kwargs)
            else:
                raise ValueError("line plot requires 'y' column")

        elif plot_type == "scatter":
            if not (x and y):
                raise ValueError("scatter plot requires both 'x' and 'y' columns")
            ax.scatter(df[x], df[y], **kwargs)

        elif plot_type == "bar":
            if x and y:
                ax.bar(df[x], df[y], **kwargs)
            elif y:
                ax.bar(df.index, df[y], **kwargs)
            else:
                raise ValueError("bar plot requires 'y' column")

        elif plot_type == "hist":
            if not x:
                raise ValueError("histogram requires 'x' column")
            ax.hist(df[x], **kwargs)

        elif plot_type == "box":
            if x:
                df.boxplot(column=x, ax=ax, **kwargs)
            else:
                df.boxplot(ax=ax, **kwargs)

        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        # Set labels and title
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        plt.tight_layout()

        _figures[fig_id] = {"fig": fig, "type": "matplotlib"}
        logger.info(f"Created matplotlib {plot_type} plot: {fig_id}")

        return fig_id

    except Exception as e:
        plt.close(fig)
        logger.error(f"Error creating matplotlib plot: {e}")
        raise


def create_plotly_plot(
    df_id: str,
    plot_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Create interactive plotly plot from DataFrame.

    Args:
        df_id: DataFrame identifier
        plot_type: Type of plot (line, scatter, bar, histogram, box)
        x: X-axis column name
        y: Y-axis column name
        title: Plot title
        **kwargs: Additional plotly parameters

    Returns:
        Figure ID

    Raises:
        KeyError: If df_id or columns don't exist
        ValueError: If plot_type is invalid
    """
    df = get_dataframe(df_id)
    fig_id = f"fig_{uuid.uuid4().hex[:8]}"

    try:
        if plot_type == "line":
            fig = px.line(df, x=x, y=y, title=title, **kwargs)

        elif plot_type == "scatter":
            fig = px.scatter(df, x=x, y=y, title=title, **kwargs)

        elif plot_type == "bar":
            fig = px.bar(df, x=x, y=y, title=title, **kwargs)

        elif plot_type == "histogram":
            fig = px.histogram(df, x=x, title=title, **kwargs)

        elif plot_type == "box":
            fig = px.box(df, x=x, y=y, title=title, **kwargs)

        else:
            raise ValueError(f"Unsupported plotly plot type: {plot_type}")

        _figures[fig_id] = {"fig": fig, "type": "plotly"}
        logger.info(f"Created plotly {plot_type} plot: {fig_id}")

        return fig_id

    except Exception as e:
        logger.error(f"Error creating plotly plot: {e}")
        raise


def create_seaborn_plot(
    df_id: str,
    plot_type: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Create seaborn plot from DataFrame.

    Args:
        df_id: DataFrame identifier
        plot_type: Type of plot (scatter, line, bar, box, violin, heatmap, pairplot)
        x: X-axis column name
        y: Y-axis column name
        hue: Grouping variable
        title: Plot title
        **kwargs: Additional seaborn parameters

    Returns:
        Figure ID

    Raises:
        KeyError: If df_id or columns don't exist
        ValueError: If plot_type is invalid
    """
    df = get_dataframe(df_id)
    fig, ax = plt.subplots()
    fig_id = f"fig_{uuid.uuid4().hex[:8]}"

    try:
        if plot_type == "scatter":
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)

        elif plot_type == "line":
            sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)

        elif plot_type == "bar":
            sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)

        elif plot_type == "box":
            sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)

        elif plot_type == "violin":
            sns.violinplot(data=df, x=x, y=y, hue=hue, ax=ax, **kwargs)

        elif plot_type == "heatmap":
            # For heatmap, use numeric columns or specified columns
            if x and y:
                pivot = df.pivot_table(values=y, index=x, aggfunc='mean')
                sns.heatmap(pivot, annot=True, fmt=".2f", ax=ax, **kwargs)
            else:
                numeric_df = df.select_dtypes(include=['number'])
                sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax, **kwargs)

        else:
            raise ValueError(f"Unsupported seaborn plot type: {plot_type}")

        if title:
            ax.set_title(title)

        plt.tight_layout()

        _figures[fig_id] = {"fig": fig, "type": "seaborn"}
        logger.info(f"Created seaborn {plot_type} plot: {fig_id}")

        return fig_id

    except Exception as e:
        plt.close(fig)
        logger.error(f"Error creating seaborn plot: {e}")
        raise


def save_figure(fig_id: str, file_path: str, **kwargs) -> str:
    """
    Save figure to file.

    Args:
        fig_id: Figure identifier
        file_path: Output file path
        **kwargs: Additional save parameters (dpi, bbox_inches, etc.)

    Returns:
        Absolute path to saved file

    Raises:
        KeyError: If fig_id doesn't exist
    """
    if fig_id not in _figures:
        raise KeyError(f"Figure not found: {fig_id}")

    fig_data = _figures[fig_id]
    path = Path(file_path)

    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if fig_data["type"] == "plotly":
            # Plotly figure
            fig = fig_data["fig"]
            if path.suffix == ".html":
                fig.write_html(str(path))
            elif path.suffix in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
                fig.write_image(str(path), **kwargs)
            else:
                raise ValueError(f"Unsupported file format for plotly: {path.suffix}")

        else:
            # Matplotlib/Seaborn figure
            fig = fig_data["fig"]
            default_kwargs = {"dpi": 300, "bbox_inches": "tight"}
            default_kwargs.update(kwargs)
            fig.savefig(path, **default_kwargs)

        logger.info(f"Saved figure {fig_id} to {path}")
        return str(path.absolute())

    except Exception as e:
        logger.error(f"Error saving figure {fig_id}: {e}")
        raise


def close_figure(fig_id: str) -> bool:
    """
    Close and remove figure from memory.

    Args:
        fig_id: Figure identifier

    Returns:
        True if successful

    Raises:
        KeyError: If fig_id doesn't exist
    """
    if fig_id not in _figures:
        raise KeyError(f"Figure not found: {fig_id}")

    fig_data = _figures[fig_id]

    if fig_data["type"] in ["matplotlib", "seaborn"]:
        plt.close(fig_data["fig"])

    del _figures[fig_id]
    logger.debug(f"Closed figure {fig_id}")
    return True


def list_figures() -> List[Dict[str, Any]]:
    """
    List all stored figures.

    Returns:
        List of figure info
    """
    return [
        {
            "fig_id": fig_id,
            "type": fig_data["type"],
        }
        for fig_id, fig_data in _figures.items()
    ]


def clear_all_figures() -> None:
    """Close and clear all figures from memory."""
    for fig_id in list(_figures.keys()):
        try:
            close_figure(fig_id)
        except:
            pass
    _figures.clear()
    plt.close('all')
    logger.info("Cleared all figures")
