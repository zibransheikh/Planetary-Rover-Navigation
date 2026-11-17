"""Procedural 32x32 terrain + energy-table-aware shortest paths (generator/evaluator side; never imported by agents)."""
from __future__ import annotations

import heapq

import numpy as np

from src.environment.terrain import DELTAS, INF, NORMAL, OBSTACLE, ROCK, SAND


def _blur(a: np.ndarray, passes: int) -> np.ndarray:
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        n, m = a.shape
        a = sum(p[i:i + n, j:j + m] for i in range(3) for j in range(3)) / 9.0
    return a


def _ranks01(values: np.ndarray) -> np.ndarray:
    """Replace values by their rank in (0,1): exact, distribution-free composition control."""
    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    r = np.empty(flat.size)
    r[order] = (np.arange(flat.size) + 0.5) / flat.size
    return r.reshape(values.shape)


def generate_terrain(rng: np.random.Generator, size: int, p_normal: float, p_sand: float, p_rock: float,
                     corr: int) -> np.ndarray:
    """Spatially clustered terrain with (almost) exact composition.

    Two independent smoothed noise fields: one decides where obstacles are (top `1-p_normal-p_sand-p_rock` ranks),
    the other splits the remaining free cells into normal/sand/rock in proportion p_normal:p_sand:p_rock.
    `corr` = number of 3x3 box-blur passes (0 -> i.i.d. cells; larger -> bigger patches).
    """
    fa = _blur(rng.random((size, size)), corr)
    fb = _blur(rng.random((size, size)), corr)
    p_obs = max(0.0, 1.0 - (p_normal + p_sand + p_rock))
    obstacle = _ranks01(fb) < p_obs
    grid = np.full((size, size), OBSTACLE, dtype=np.int64)
    free = ~obstacle
    ua = np.empty(int(free.sum()))
    ua[np.argsort(fa[free], kind="stable")] = (np.arange(ua.size) + 0.5) / ua.size
    tot = p_normal + p_sand + p_rock
    f_n, f_s = p_normal / tot, p_sand / tot
    cls = np.where(ua < f_n, NORMAL, np.where(ua < f_n + f_s, SAND, ROCK))
    grid[free] = cls
    return grid


def clear_blocks(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], k: int = 3) -> None:
    """Start and goal corners: k x k NORMAL blocks, so the rover can never be walled in at the endpoints."""
    n = grid.shape[0]
    grid[:k, :k] = NORMAL
    grid[n - k:, n - k:] = NORMAL
    grid[start] = NORMAL
    grid[goal] = NORMAL


def energy_lut(costs) -> np.ndarray:
    c = tuple(int(v) for v in costs)
    if len(c) != 3 or min(c) < 1:
        raise ValueError(f"energy table needs 3 integers >= 1 (normal, sand, rock), got {costs!r}")
    return np.array([c[0], c[1], c[2], 0], dtype=np.int64)


def dijkstra_lut(grid: np.ndarray, source: tuple[int, int], lut: np.ndarray) -> np.ndarray:
    """Min battery from `source` to every cell; cost is paid on entering a cell. FULL MAP: generator/evaluator only."""
    n = grid.shape[0]
    dist = np.full((n, n), INF, dtype=np.int64)
    dist[source] = 0
    heap = [(0, source[0], source[1])]
    while heap:
        c, x, y = heapq.heappop(heap)
        if c > dist[x, y]:
            continue
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[nx, ny] != OBSTACLE:
                nc = c + lut[grid[nx, ny]]
                if nc < dist[nx, ny]:
                    dist[nx, ny] = nc
                    heapq.heappush(heap, (nc, nx, ny))
    return dist


def dijkstra_path_lut(grid: np.ndarray, source, goal, lut: np.ndarray):
    """Optimal path (source..goal) under `lut`, or None. FULL MAP: evaluator/tests only."""
    n = grid.shape[0]
    to_goal = np.full((n, n), INF, dtype=np.int64)
    to_goal[goal] = 0
    heap = [(0, goal[0], goal[1])]
    while heap:
        c, x, y = heapq.heappop(heap)
        if c > to_goal[x, y]:
            continue
        for dx, dy in DELTAS:
            px, py = x - dx, y - dy
            if 0 <= px < n and 0 <= py < n and grid[px, py] != OBSTACLE:
                nc = c + lut[grid[x, y]]
                if nc < to_goal[px, py]:
                    to_goal[px, py] = nc
                    heapq.heappush(heap, (nc, px, py))
    if to_goal[source] >= INF:
        return None
    path, (x, y) = [tuple(source)], tuple(source)
    while (x, y) != tuple(goal):
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[nx, ny] != OBSTACLE and to_goal[x, y] == lut[grid[nx, ny]] + to_goal[nx, ny]:
                x, y = nx, ny
                path.append((x, y))
                break
    return path
