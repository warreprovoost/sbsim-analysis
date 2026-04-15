#!/usr/bin/env python3
"""Render floorplan images for thesis figures.

Saves one PNG per floorplan to images/.
Walls are drawn as lines on cell boundaries for clean rendering.

Usage:
    python scripts/plot_floorplans.py
"""
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from smart_control_analysis.floorplans import (
    corporate_floor,
    headquarters_floor,
    office_4room,
    single_room,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
CV_SIZE_CM = 50

COLOR_ROOM     = "#dce8f5"
COLOR_CORRIDOR = "#f0ede8"
COLOR_EXTERIOR = "#3a3a3a"
COLOR_INT_WALL = "#555555"
COLOR_EXT_WALL = "#1a1a1a"


def _find_rooms(floorplan):
    """Find connected air regions (value=0) not separated by walls.
    Returns list of (row_indices, col_indices) for each region."""
    from collections import deque
    TH, TW = floorplan.shape
    visited = np.zeros((TH, TW), dtype=bool)
    regions = []
    for r in range(TH):
        for c in range(TW):
            if floorplan[r, c] == 0 and not visited[r, c]:
                # BFS
                queue = deque([(r, c)])
                visited[r, c] = True
                cells = []
                while queue:
                    rr, cc = queue.popleft()
                    cells.append((rr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < TH and 0 <= nc < TW:
                            if not visited[nr, nc] and floorplan[nr, nc] == 0:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                rows = [c[0] for c in cells]
                cols = [c[1] for c in cells]
                regions.append((rows, cols))
    return regions


def _render(floorplan, name, title, cv_cm=CV_SIZE_CM, figsize=(8, 5)):
    TH, TW = floorplan.shape

    # Physical size in metres
    w_m = (TW - 2) * cv_cm / 100
    h_m = (TH - 2) * cv_cm / 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(COLOR_EXTERIOR)

    # --- Colour all interior cells (air + walls) with room colour ---
    # Wall cells are then overdrawn with thin lines so they appear as thin partitions
    for r in range(1, TH-1):
        for c in range(1, TW-1):
            rect = plt.Rectangle((c, r), 1, 1, facecolor=COLOR_ROOM, edgecolor="none")
            ax.add_patch(rect)

    # --- Draw interior walls as thin filled rectangles centred in each wall cell ---
    w = 0.2  # wall visual thickness (fraction of cell)
    half = w / 2
    for r in range(1, TH-1):
        for c in range(1, TW-1):
            if floorplan[r, c] == 1:
                cx, cy = c + 0.5, r + 0.5
                rect = plt.Rectangle((cx - half, cy - half), w, w,
                                     facecolor=COLOR_INT_WALL, edgecolor="none")
                ax.add_patch(rect)
                # Extend toward each neighbouring wall or exterior cell
                if floorplan[r, c+1] in (1, 2):   # right neighbour is wall/ext
                    rect = plt.Rectangle((cx - half, cy - half), 0.5 + half, w,
                                         facecolor=COLOR_INT_WALL, edgecolor="none")
                    ax.add_patch(rect)
                if floorplan[r, c-1] in (1, 2):   # left neighbour
                    rect = plt.Rectangle((cx - 0.5 - half, cy - half), 0.5 + half, w,
                                         facecolor=COLOR_INT_WALL, edgecolor="none")
                    ax.add_patch(rect)
                if floorplan[r+1, c] in (1, 2):   # below neighbour
                    rect = plt.Rectangle((cx - half, cy - half), w, 0.5 + half,
                                         facecolor=COLOR_INT_WALL, edgecolor="none")
                    ax.add_patch(rect)
                if floorplan[r-1, c] in (1, 2):   # above neighbour
                    rect = plt.Rectangle((cx - half, cy - 0.5 - half), w, 0.5 + half,
                                         facecolor=COLOR_INT_WALL, edgecolor="none")
                    ax.add_patch(rect)

    # --- Draw exterior border ---
    ext_lw = 3.0
    ax.plot([1, TW-1], [1,    1],    color=COLOR_EXT_WALL, linewidth=ext_lw)  # top
    ax.plot([1, TW-1], [TH-1, TH-1],color=COLOR_EXT_WALL, linewidth=ext_lw)  # bottom
    ax.plot([1,    1], [1, TH-1],    color=COLOR_EXT_WALL, linewidth=ext_lw)  # left
    ax.plot([TW-1, TW-1],[1, TH-1], color=COLOR_EXT_WALL, linewidth=ext_lw)  # right

    # --- Axes in physical metres ---
    ax.set_xlim(0, TW)
    ax.set_ylim(TH, 0)
    ax.set_aspect("equal")

    n_xticks = min(7, TW // 5 + 1)
    n_yticks = min(6, TH // 5 + 1)
    x_ticks = np.linspace(1, TW-1, n_xticks)
    y_ticks = np.linspace(1, TH-1, n_yticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{(x-1)*cv_cm/100:.0f}" for x in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{(y-1)*cv_cm/100:.0f}" for y in y_ticks])
    ax.set_xlabel("Width (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"{title}  ({w_m:.0f} m × {h_m:.0f} m)",
                 fontsize=11, fontweight="bold")

    legend = [
        mpatches.Patch(color=COLOR_ROOM,     label="Interior space"),
        mpatches.Patch(color=COLOR_INT_WALL, label="Interior wall"),
        mpatches.Patch(color=COLOR_EXTERIOR, label="Exterior wall"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    configs = [
        ("single_room",        single_room(10, 10),   "Single Room",          (4, 4)),
        ("office_4room",       office_4room(),         "4-Room Office Floor",  (6, 5)),
        ("corporate_floor",    corporate_floor(),      "Corporate Floor",      (10, 5)),
        ("headquarters_floor", headquarters_floor(),   "Headquarters Floor",   (13, 6)),
    ]

    for name, (floorplan, _), title, figsize in configs:
        fig = _render(floorplan, name, title, figsize=figsize)
        out = os.path.join(OUT_DIR, f"floorplan_{name}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
