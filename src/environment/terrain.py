"""Terrain constants and the pure energy/geometry helpers shared by env and evaluator."""
from __future__ import annotations

import heapq
from collections import deque

import numpy as np

NORMAL, SAND, ROCK, OBSTACLE = 0, 1, 2, 3
ENERGY = {NORMAL: 1, SAND: 3, ROCK: 5}  # OBSTACLE is impassable
ENERGY_LUT = np.array([1, 3, 5, 0], dtype=np.int64)
# action index -> (d_row, d_col); identical to the original (0:-x, 1:+x, 2:-y, 3:+y)
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# observation encoding of a terrain cell in the local patch (identical to original)
PATCH_VALUE = np.array([0.0, 0.3, 0.6, 1.0])
INF = np.iinfo(np.int64).max // 4


def generate_grid(rng: np.random.Generator, size: int, p_normal: float, p_sand: float, p_rock: float) -> np.ndarray:
    """Vectorised i.i.d. terrain (same distribution as the original per-cell loop)."""
    r = rng.random((size, size))
    grid = np.full((size, size), OBSTACLE, dtype=np.int64)
    grid[r < p_normal + p_sand + p_rock] = ROCK
    grid[r < p_normal + p_sand] = SAND
    grid[r < p_normal] = NORMAL
    return grid


def clear_endpoints(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> None:
    """Original behaviour: start/goal are NORMAL and obstacles in their 2x2 corner blocks are cleared."""
    n = grid.shape[0]
    for di in range(2):
        for dj in range(2):
            for (i, j) in ((di, dj), (n - 1 - di, n - 1 - dj)):
                if grid[i, j] == OBSTACLE:
                    grid[i, j] = NORMAL
    grid[start] = NORMAL
    grid[goal] = NORMAL


def dijkstra(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """Minimum battery cost from `source` to every cell (INF if unreachable). FULL-MAP: evaluator/generator only."""
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
                nc = c + ENERGY_LUT[grid[nx, ny]]
                if nc < dist[nx, ny]:
                    dist[nx, ny] = nc
                    heapq.heappush(heap, (nc, nx, ny))
    return dist


def dijkstra_path(grid: np.ndarray, source: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Optimal path as a list of cells (source..goal) or None. FULL-MAP: evaluator/tests only."""
    # Cost of entering a cell is paid on arrival (asymmetric), so compute cost-to-go on the reverse graph.
    n = grid.shape[0]
    to_goal = np.full((n, n), INF, dtype=np.int64)
    to_goal[goal] = 0
    heap = [(0, goal[0], goal[1])]
    while heap:
        c, x, y = heapq.heappop(heap)
        if c > to_goal[x, y]:
            continue
        # predecessor (px,py) -> (x,y) costs ENERGY[(x,y)]
        for dx, dy in DELTAS:
            px, py = x - dx, y - dy
            if 0 <= px < n and 0 <= py < n and grid[px, py] != OBSTACLE:
                nc = c + ENERGY_LUT[grid[x, y]]
                if nc < to_goal[px, py]:
                    to_goal[px, py] = nc
                    heapq.heappush(heap, (nc, px, py))
    if to_goal[source] >= INF:
        return None
    path, (x, y) = [source], source
    while (x, y) != goal:
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and grid[nx, ny] != OBSTACLE and \
                    to_goal[x, y] == ENERGY_LUT[grid[nx, ny]] + to_goal[nx, ny]:
                x, y = nx, ny
                path.append((x, y))
                break
    return path


def is_connected(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    n = grid.shape[0]
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in DELTAS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen and grid[nx, ny] != OBSTACLE:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False
