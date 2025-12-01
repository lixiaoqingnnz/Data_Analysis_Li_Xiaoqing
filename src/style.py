import matplotlib.pyplot as plt
import seaborn as sns

def set_global_style():
    """
    Apply a unified global visualization style for all charts in the project.

    This function should be called once at the start of the program
    (for example, at the top of main.py):
        from visualization.style import set_global_style
        set_global_style()

    The style configuration includes:
    - Seaborn whitegrid theme
    - Consistent color palette (Set2)
    - Standardized font sizes, DPI, grid style, and axis formatting
    - Updated Matplotlib default color cycle
    """
    set2 = plt.get_cmap("Set2")

    sns.set_theme(
        style="whitegrid",
        palette=set2.colors,
        rc={
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 12,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.5,
        }
    )

    # Use Set2 as the default Matplotlib color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=set2.colors)


def apply_common_format(ax, title: str, xlabel: str, ylabel: str):
    """
    Apply common formatting to a standard 2D chart.

    This helper function ensures consistent:
    - Titles
    - Axis labels
    - Tick label size
    - Slight rotation of x-axis labels for readability

    Note:
        Polar charts such as radar plots generally do not need this formatting.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.tick_params(axis="both", labelsize=12)

    # Apply slight rotation to x-axis labels for improved readability
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_horizontalalignment("right")
