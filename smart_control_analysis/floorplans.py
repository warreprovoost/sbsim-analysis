import numpy as np
from typing import Tuple


def headquarters_floor() -> Tuple[np.ndarray, np.ndarray]:
    """
    Large corporate headquarters floor — 16 zones, dual-wing layout at 50 cm/cell.

    Physical dimensions (cv_size_cm=50):
      Interior grid: 100 × 60 cells  →  50 m × 30 m  =  1500 m²

    Layout — 3 horizontal bands separated by corridors:

    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │ NORTH WING  (rows 0–17, 50m × 9m)                                                   │
    │ ┌──────────┬──┬──────────┬──┬──────────┬──┬──────────┬──┬──────────┬──┬──────────┐  │
    │ │ Open     │  │Conf. A   │  │Conf. B   │  │Conf. C   │  │Open Plan │  │R&D Lab   │  │
    │ │ Plan NW  │  │(14×16)   │  │(14×16)   │  │(14×16)   │  │NE        │  │(14×16)   │  │
    │ │ (14×16)  │  │          │  │          │  │          │  │(14×16)   │  │          │  │
    │ └──────────┴──┴──────────┴──┴──────────┴──┴──────────┴──┴──────────┴──┴──────────┘  │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │ CORRIDOR N (rows 18–19, 50m × 1m)                                                   │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │ CENTRAL CORE (rows 20–39, 50m × 10m)                                                │
    │ ┌───────────────┬──┬────────────────┬──┬────────────────┬──┬───────────────────────┐ │
    │ │ Reception /   │  │ Large Board-   │  │ IT / Server    │  │ Open Plan Central     │ │
    │ │ Lobby         │  │ room           │  │ Hub            │  │                       │ │
    │ │ (24×18)       │  │ (18×18)        │  │ (14×18)        │  │ (38×18)               │ │
    │ └───────────────┴──┴────────────────┴──┴────────────────┴──┴───────────────────────┘ │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │ CORRIDOR S (rows 40–41, 50m × 1m)                                                   │
    ├──────────────────────────────────────────────────────────────────────────────────────┤
    │ SOUTH WING  (rows 42–59, 50m × 9m)                                                  │
    │ ┌──────────┬──┬──────────┬──┬──────────┬──┬──────────┬──┬──────────┬──┬──────────┐  │
    │ │ Open     │  │Office    │  │Office    │  │Office    │  │Office    │  │ Kitchen/ │  │
    │ │ Plan SW  │  │Suite A   │  │Suite B   │  │Suite C   │  │Suite D   │  │ Canteen  │  │
    │ │ (14×16)  │  │(14×16)   │  │(14×16)   │  │(14×16)   │  │(14×16)   │  │(14×16)   │  │
    │ └──────────┴──┴──────────┴──┴──────────┴──┴──────────┴──┴──────────┴──┴──────────┘  │
    └──────────────────────────────────────────────────────────────────────────────────────┘

    Zones (16 total):
      North wing  (0-5):  0=Open NW, 1=Conf A, 2=Conf B, 3=Conf C, 4=Open NE, 5=R&D Lab
      Central core(6-9):  6=Reception/Lobby, 7=Boardroom, 8=IT/Server Hub, 9=Open Central
      South wing (10-15): 10=Open SW, 11=Office Suite A, 12=B, 13=C, 14=D, 15=Kitchen

    Corridors: interior air, no zone (comfort-free thermal buffer between wings and core).

    floorplan: 0=interior air, 1=interior wall, 2=exterior
    zone_map:  0-15=zone id, -1=wall/exterior/corridor
    """
    W, H = 100, 60
    TW, TH = W + 2, H + 2

    floorplan = np.full((TH, TW), 2, dtype=int)
    zone_map  = np.full((TH, TW), -1, dtype=int)
    floorplan[1:-1, 1:-1] = 0  # all interior is air

    # ── Band boundaries (full-array row indices) ─────────────────────────────
    r_north_bot  = 18   # bottom wall of north wing
    r_corrN_bot  = 20   # bottom wall of north corridor  (corridor = rows 18-19)
    r_core_bot   = 40   # bottom wall of central core
    r_corrS_bot  = 42   # bottom wall of south corridor  (corridor = rows 40-41)
    # south wing: rows 42..TH-2

    # ── Horizontal band walls ─────────────────────────────────────────────────
    floorplan[r_north_bot, 1:-1] = 1
    floorplan[r_corrN_bot, 1:-1] = 1
    floorplan[r_core_bot,  1:-1] = 1
    floorplan[r_corrS_bot, 1:-1] = 1

    # ── Vertical walls — north wing (rows 1..r_north_bot) ────────────────────
    # 6 equal-ish rooms across 100 cols, walls at cols 15,16,31,32,47,48,63,64,79,80
    nv = [16, 32, 48, 64, 80]   # 5 walls → 6 north rooms (each ~15-16 cols wide)
    for c in nv:
        floorplan[1:r_north_bot, c] = 1

    # ── Vertical walls — central core (rows r_corrN_bot..r_core_bot) ─────────
    # 4 rooms: Reception(24), Boardroom(18), IT(14), Open Central(remainder=38+walls)
    cv = [25, 44, 59]           # walls at interior cols 24, 43, 58
    for c in cv:
        floorplan[r_corrN_bot:r_core_bot, c] = 1

    # ── Vertical walls — south wing (rows r_corrS_bot..TH-1) ─────────────────
    sv = [16, 32, 48, 64, 80]   # same column positions as north wing
    for c in sv:
        floorplan[r_corrS_bot:-1, c] = 1

    # ── Zone map ─────────────────────────────────────────────────────────────
    # All interior air = 0; walls/corridors = -1. Simulator auto-discovers rooms
    # via connected components on the 0-regions split by wall boundaries.
    nr = slice(1, r_north_bot)
    zone_map[nr, 1:nv[0]]          = 0   # Open Plan NW
    zone_map[nr, nv[0]+1:nv[1]]    = 0   # Conf. A
    zone_map[nr, nv[1]+1:nv[2]]    = 0   # Conf. B
    zone_map[nr, nv[2]+1:nv[3]]    = 0   # Conf. C
    zone_map[nr, nv[3]+1:nv[4]]    = 0   # Open Plan NE
    zone_map[nr, nv[4]+1:-1]       = 0   # R&D Lab

    # corridors remain -1 (rows r_north_bot+1..r_corrN_bot-1 and r_core_bot+1..r_corrS_bot-1)

    cr = slice(r_corrN_bot + 1, r_core_bot)
    zone_map[cr, 1:cv[0]]          = 0   # Reception / Lobby
    zone_map[cr, cv[0]+1:cv[1]]    = 0   # Boardroom
    zone_map[cr, cv[1]+1:cv[2]]    = 0   # IT / Server Hub
    zone_map[cr, cv[2]+1:-1]       = 0   # Open Plan Central

    sr = slice(r_corrS_bot + 1, TH - 1)
    zone_map[sr, 1:sv[0]]          = 0   # Open Plan SW
    zone_map[sr, sv[0]+1:sv[1]]    = 0   # Office Suite A
    zone_map[sr, sv[1]+1:sv[2]]    = 0   # Office Suite B
    zone_map[sr, sv[2]+1:sv[3]]    = 0   # Office Suite C
    zone_map[sr, sv[3]+1:sv[4]]    = 0   # Office Suite D
    zone_map[sr, sv[4]+1:-1]       = 0   # Kitchen / Canteen

    return floorplan, zone_map


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
    # All interior air = 0; walls/corridor = -1. Simulator auto-discovers rooms.
    # North rooms (rows 1..r_corr_top-1)
    nr = slice(1, r_corr_top)
    zone_map[nr, 1:v1]       = 0   # Open Plan A
    zone_map[nr, v1+1:v2]    = 0   # Large Conf. Room
    zone_map[nr, v2+1:v3]    = 0   # Open Plan B
    zone_map[nr, v3+1:v4]    = 0   # Server Room
    zone_map[nr, v4+1:-1]    = 0   # Meeting Room

    # Corridor: left as -1 (no zone, no comfort penalty)

    # South rooms (rows r_corr_bot+1..TH-2)
    sr = slice(r_corr_bot + 1, TH - 1)
    zone_map[sr, 1:v5]       = 0   # Office 1
    zone_map[sr, v5+1:v6]    = 0   # Office 2
    zone_map[sr, v6+1:v7]    = 0   # Office 3
    zone_map[sr, v7+1:v8]    = 0   # Office 4
    zone_map[sr, v8+1:-1]    = 0   # Kitchen / Break Room

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
    # All interior air = 0; walls = -1. The simulator auto-discovers rooms via
    # connected components on the 0-regions split by wall (-1) boundaries.
    zone_map[1:h_wall, 1:v_top]           = 0  # Room 1: top-left
    zone_map[1:h_wall, v_top + 1:-1]      = 0  # Room 2: top-right
    zone_map[h_wall + 1:-1, 1:v_bot]      = 0  # Room 3: bottom-left
    zone_map[h_wall + 1:-1, v_bot + 1:-1] = 0  # Room 4: bottom-right

    return floorplan, zone_map
