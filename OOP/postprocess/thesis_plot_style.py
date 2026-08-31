from __future__ import annotations

from dataclasses import dataclass
import numbers

import matplotlib as mpl


@dataclass(frozen=True)
class ThesisPlotStyle:
    base: float = 18.0
    axes_title: float = 22.0
    axes_label: float = 20.0
    ticks: float = 17.0
    legend: float = 14.0
    figure_title: float = 24.0
    colorbar_label: float = 19.0
    annotation: float = 12.0
    min_dpi: int = 300


def _numeric(value, fallback):
    if isinstance(value, numbers.Real):
        return float(value)
    return fallback


def apply_thesis_style(style: ThesisPlotStyle | None = None):
    """
    Apply minimum font sizes for thesis-ready figures.

    This affects plotting only. It does not modify simulation data,
    numerical diagnostics, spectra, PDFs, or solver behavior.
    """
    style = style or ThesisPlotStyle()

    mpl.rcParams.update(
        {
            "font.size": style.base,
            "axes.titlesize": style.axes_title,
            "axes.labelsize": style.axes_label,
            "xtick.labelsize": style.ticks,
            "ytick.labelsize": style.ticks,
            "legend.fontsize": style.legend,
            "legend.title_fontsize": style.legend,
            "figure.titlesize": style.figure_title,
            "lines.linewidth": max(
                float(mpl.rcParams.get("lines.linewidth", 1.5)),
                2.0,
            ),
            "savefig.dpi": style.min_dpi,
        }
    )

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.colorbar import Colorbar

    if getattr(Axes, "_thesis_font_patch", False):
        return style

    old_title = Axes.set_title
    old_xlabel = Axes.set_xlabel
    old_ylabel = Axes.set_ylabel
    old_legend = Axes.legend
    old_ticks = Axes.tick_params
    old_text = Axes.text
    old_suptitle = Figure.suptitle
    old_savefig = Figure.savefig
    old_figure_colorbar = Figure.colorbar
    old_cbar_label = Colorbar.set_label

    def set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        kwargs["fontsize"] = max(
            _numeric(kwargs.get("fontsize"), style.axes_title),
            style.axes_title,
        )
        return old_title(
            self,
            label,
            fontdict=fontdict,
            loc=loc,
            pad=pad,
            y=y,
            **kwargs,
        )

    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        kwargs["fontsize"] = max(
            _numeric(kwargs.get("fontsize"), style.axes_label),
            style.axes_label,
        )
        return old_xlabel(
            self,
            xlabel,
            fontdict=fontdict,
            labelpad=labelpad,
            loc=loc,
            **kwargs,
        )

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        kwargs["fontsize"] = max(
            _numeric(kwargs.get("fontsize"), style.axes_label),
            style.axes_label,
        )
        return old_ylabel(
            self,
            ylabel,
            fontdict=fontdict,
            labelpad=labelpad,
            loc=loc,
            **kwargs,
        )

    def legend(self, *args, **kwargs):
        kwargs["fontsize"] = max(
            _numeric(kwargs.get("fontsize"), style.legend),
            style.legend,
        )
        return old_legend(self, *args, **kwargs)

    def tick_params(self, axis="both", **kwargs):
        kwargs["labelsize"] = max(
            _numeric(kwargs.get("labelsize"), style.ticks),
            style.ticks,
        )
        return old_ticks(self, axis=axis, **kwargs)

    def text(self, x, y, s, fontdict=None, **kwargs):
        if "fontsize" in kwargs:
            kwargs["fontsize"] = max(
                _numeric(kwargs["fontsize"], style.annotation),
                style.annotation,
            )
        return old_text(self, x, y, s, fontdict=fontdict, **kwargs)

    def suptitle(self, t, **kwargs):
        kwargs["fontsize"] = max(
            _numeric(kwargs.get("fontsize"), style.figure_title),
            style.figure_title,
        )
        return old_suptitle(self, t, **kwargs)

    def savefig(self, fname, *args, **kwargs):
        dpi = kwargs.get("dpi")
        if dpi is None:
            kwargs["dpi"] = style.min_dpi
        elif isinstance(dpi, numbers.Real):
            kwargs["dpi"] = max(int(dpi), style.min_dpi)
        return old_savefig(self, fname, *args, **kwargs)

    def figure_colorbar(self, *args, **kwargs):
        cbar = old_figure_colorbar(self, *args, **kwargs)
        cbar.ax.tick_params(labelsize=style.ticks)
        return cbar

    def colorbar_label(self, label, *, loc=None, **kwargs):
        kwargs["fontsize"] = max(
            _numeric(kwargs.get("fontsize"), style.colorbar_label),
            style.colorbar_label,
        )
        result = old_cbar_label(self, label, loc=loc, **kwargs)
        self.ax.tick_params(labelsize=style.ticks)
        return result

    Axes.set_title = set_title
    Axes.set_xlabel = set_xlabel
    Axes.set_ylabel = set_ylabel
    Axes.legend = legend
    Axes.tick_params = tick_params
    Axes.text = text
    Figure.suptitle = suptitle
    Figure.savefig = savefig
    Figure.colorbar = figure_colorbar
    Colorbar.set_label = colorbar_label

    Axes._thesis_font_patch = True

    return style
