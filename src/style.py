# visualization/style.py
import matplotlib.pyplot as plt
import seaborn as sns

def set_global_style():
    """
    在项目启动时调用一次，统一所有图表的风格。
    例如在 main.py 顶部： from visualization.style import set_global_style; set_global_style()
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

    # Matplotlib 默认颜色循环也用 Set2
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=set2.colors)


def apply_common_format(ax, title: str, xlabel: str, ylabel: str):
    """
    每个普通 2D 图在画完后调用一次，统一标题、坐标轴与刻度样式。
    （雷达图这种 polar 的图可以不用它）
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.tick_params(axis="both", labelsize=12)

    # x 轴标签统一轻微倾斜（对年份、region 等都比较友好）
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_horizontalalignment("right")
