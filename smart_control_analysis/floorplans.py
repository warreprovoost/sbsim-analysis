import numpy as np
from typing import Tuple


def corporate_floor() -> Tuple[np.ndarray, np.ndarray]:
    """
    Large corporate office floor — 9 zones, realistic layout at 50 cm/cell.

    Physical dimensions (at cv_size_cm=50):
      Interior grid: 60 × 30 cells  →  30 m × 15 m  =  450 m²

    Layout (interior cells, walls shown as | / ─):

      col:  0        16 17      33 34      50 51  55 56  59
            +─────────+─+────────+─+────────+─+───+─+───+ row 0
            │ Open Plan│ │ Large  │ │ Open   │ │Srv│ │Mee│
            │  A       │ │ Conf.  │ │ Plan B │ │ . │ │t. │
            │ (16×13)  │ │ Room   │ │(16×13) │ │Rm │ │Rm │
            │          │ │ (16×13)│ │        │ │(4)│ │(3)│
            +─────────+─+────────+─+────────+─+───+─+───+ row 13
            │         corridor / hallway  (60 × 2)       │ row 13-14
            +──────+──+─+───+───+─+───+───+─+────────────+ row 15
            │Off. 1│  │ │O2 │O3 │ │O4 │O5 │ │  Kitchen / │
            │      │  │ │   │   │ │   │   │ │  Break Room│
            │(13×13)  │ │(9)│(9)│ │(9)│(9)│ │  (16×13)   │
            │      │  │ │   │   │ │   │   │ │            │
            +──────+──+─+───+───+─+───+───+─+────────────+ row 28

    Zones (9 total):
      0  Open Plan A        — large open workspace, north-west   (16×13)
      1  Large Conf. Room   — boardroom, north-centre            (16×13)
      2  Open Plan B        — large open workspace, north-east   (16×13)
      3  Server Room        — no comfort penalty needed           (4×13)
      4  Meeting Room       — small meeting, north corner         (3×13)
      5  Office 1           — private office, south-west        (13×13)
      6  Office 2           — private office, south-centre-left   (9×13)
      7  Office 3           — private office, south-centre-right  (9×13)
      8  Office 4           — private office                      (9×13)  [unused col label O4]
      9  Kitchen/Break Room — south-east                         (16×13)

    Note: the corridor (rows 13-14) is interior air but assigned to no zone (-1).
          Interior walls are 1 cell thick.

    floorplan: 0=interior air, 1=interior wall, 2=exterior
    zone_map:  0-9=zone id, -1=wall/exterior/corridor
    """
    W, H = 60, 30           # interior cells (excl. 1-cell exterior border)
    TW, TH = W + 2, H + 2

    floorplan = np.full((TH, TW), 2, dtype=int)
    zone_map  = np.full((TH, TW), -1, dtype=int)

    # ── Interior air (everything inside border) ──────────────────────────────
    floorplan[1:-1, 1:-1] = 0

    # ── Key row/col positions (full-array coords = interior + 1) ─────────────
    # Horizontal walls
    r_corr_top = 14   # top of corridor    (interior row 13)
    r_corr_bot = 16   # bottom of corridor (interior row 15 = first south row)

    # Vertical walls — north half (cols in full-array)
    v1 = 17   # between Open A and Conf. Room  (interior col 16)
    v2 = 34   # between Conf. Room and Open B  (interior col 33)
    v3 = 51   # between Open B and Server Rm   (interior col 50)
    v4 = 56   # between Server Rm and Meeting  (interior col 55)

    # Vertical walls — south half
    v5 = 14   # east edge of Office 1          (interior col 13)
    v6 = 24   # between Office 2 and 3         (interior col 23)
    v7 = 34   # between Office 3 and 4/Kitchen (interior col 33)  (reuse v2 col)
    v8 = 44   # between Office 4 and Kitchen   (interior col 43)

    # ── Horizontal dividing walls ─────────────────────────────────────────────
    # North/south split: corridor top and bottom walls
    floorplan[r_corr_top, 1:-1] = 1   # top wall of corridor
    floorplan[r_corr_bot,  1:-1] = 1   # bottom wall of corridor

    # ── Vertical walls — north half (rows 1..r_corr_top) ─────────────────────
    floorplan[1:r_corr_top, v1] = 1
    floorplan[1:r_corr_top, v2] = 1
    floorplan[1:r_corr_top, v3] = 1
    floorplan[1:r_corr_top, v4] = 1

    # ── Vertical walls — south half (rows r_corr_bot..TH-1) ──────────────────
    floorplan[r_corr_bot:-1, v5] = 1
    floorplan[r_corr_bot:-1, v6] = 1
    floorplan[r_corr_bot:-1, v7] = 1
    floorplan[r_corr_bot:-1, v8] = 1

    # ── Zone assignments ──────────────────────────────────────────────────────
    # North rooms (rows 1..r_corr_top-1)
    nr = slice(1, r_corr_top)
    zone_map[nr, 1:v1]       = 0   # Open Plan A
    zone_map[nr, v1+1:v2]    = 1   # Large Conf. Room
    zone_map[nr, v2+1:v3]    = 2   # Open Plan B
    zone_map[nr, v3+1:v4]    = 3   # Server Room
    zone_map[nr, v4+1:-1]    = 4   # Meeting Room

    # Corridor: left as -1 (no zone, no comfort penalty)

    # South rooms (rows r_corr_bot+1..TH-2)
    sr = slice(r_corr_bot + 1, TH - 1)
    zone_map[sr, 1:v5]       = 5   # Office 1
    zone_map[sr, v5+1:v6]    = 6   # Office 2
    zone_map[sr, v6+1:v7]    = 7   # Office 3
    zone_map[sr, v7+1:v8]    = 8   # Office 4
    zone_map[sr, v8+1:-1]    = 9   # Kitchen / Break Room

    return floorplan, zone_map


def single_room(room_width: int = 10, room_height: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Single square/rectangular room. The original baseline layout."""
    TW, TH = room_width + 2, room_height + 2
    floorplan = np.full((TH, TW), 2, dtype=int)
    zone_map  = np.full((TH, TW), -1, dtype=int)
    floorplan[1:-1, 1:-1] = 0
    zone_map[1:-1, 1:-1]  = 0
    return floorplan, zone_map


def office_4room() -> Tuple[np.ndarray, np.ndarray]:
    """
    4-room office floor with rooms of different sizes separated by interior walls.

    Interior layout (cv cells, wall cells shown as |/─):

        col:  0    6   12   17
              +----+----+----+  row 0
              |  Room 1 | R2 |
              | (12x8)  |(5x8)
              |         |    |  row 8
              +---+─────+────+  ← horizontal wall at interior row 8
              | R3|   Room 4 |  row 9
              |(5x|(12x7)    |
              |x7)|          |  row 15
              +---+──────────+

    Room sizes (in cv cells, excluding wall cells):
      Room 1 (zone 0): large office      12 × 8 = 96 cells  (top-left)
      Room 2 (zone 1): small office       5 × 8 = 40 cells  (top-right)
      Room 3 (zone 2): meeting room       5 × 7 = 35 cells  (bottom-left)
      Room 4 (zone 3): open-plan space   12 × 7 = 84 cells  (bottom-right)

    floorplan encoding: 0 = interior air, 1 = interior wall, 2 = exterior
    zone_map encoding:  0..3 = zone id, -1 = wall / exterior
    """
    W, H = 18, 16          # interior grid (excl. outer border)
    TW, TH = W + 2, H + 2  # total array size including 1-cell exterior border

    floorplan = np.full((TH, TW), 2, dtype=int)
    zone_map  = np.full((TH, TW), -1, dtype=int)

    # Fill entire interior as air
    floorplan[1:-1, 1:-1] = 0

    # ── Dividing walls (in full-array coordinates) ──────────────────────────
    h_wall = 9    # horizontal wall row (interior row 8 → array row 9)
    v_top  = 13   # vertical wall col splitting top half    (interior col 12)
    v_bot  = 6    # vertical wall col splitting bottom half (interior col 5)

    floorplan[h_wall, 1:-1]          = 1   # full-width horizontal wall
    floorplan[1:h_wall, v_top]       = 1   # vertical wall top half
    floorplan[h_wall + 1:-1, v_bot]  = 1   # vertical wall bottom half

    # ── Zone assignments ────────────────────────────────────────────────────
    # Room 1: top-left
    zone_map[1:h_wall, 1:v_top]           = 0
    # Room 2: top-right
    zone_map[1:h_wall, v_top + 1:-1]      = 1
    # Room 3: bottom-left
    zone_map[h_wall + 1:-1, 1:v_bot]      = 2
    # Room 4: bottom-right
    zone_map[h_wall + 1:-1, v_bot + 1:-1] = 3

    return floorplan, zone_map
