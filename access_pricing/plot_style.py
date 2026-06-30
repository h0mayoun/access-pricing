"""Matplotlib settings for paper-ready figure output.

Matches the IFAC/conference PDF requirements used in the original notebooks:
  - TrueType fonts (so embedded fonts can be edited downstream)
  - LaTeX rendering off by default (avoids font-availability issues)
"""

import matplotlib as mpl


def configure(use_tex: bool = False) -> None:
    """Apply paper-ready matplotlib defaults.

    Call once near the top of any notebook before creating figures.
    """
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["text.usetex"] = use_tex
