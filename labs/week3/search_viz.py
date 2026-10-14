"""
CS-31 Week 3 / Lab Part 1 --- INSTRUCTOR SCAFFOLD
Grid search: BFS vs A* (vs Greedy) on two map families.

Design commitments baked in (change these deliberately, not accidentally):
  * Coordinates are (row, col), NOT (x, y). Row grows downward. Start is always
    (0,0). Default goal depends on map family: object maps default to the
    centre ((n-1)//2, (n-1)//2), mazes default to the bottom-right corner
    (n-1, n-1). Each generator stamps its grid with this convention (see the
    `Grid` ndarray subclass below) so bfs/A*/stepper pick it up automatically
    without every call site having to repeat it. Rationale: matches numpy
    indexing and matplotlib imshow, so students never hit a silent transpose bug.
  * 4-connected movement, unit step cost. This is what makes Manhattan distance
    admissible. If you switch to 8-connected, Manhattan becomes INADMISSIBLE and
    A* can return a suboptimal path -- the lab's whole story breaks.
  * "Expanded" == popped off the frontier and its successors generated.
    Goal test happens ON EXPANSION for every strategy, so counts are comparable.
    (AIMA's early-goal-test BFS is faster but makes BFS/A* counts apples-to-oranges.)
  * Deterministic tie-breaking. Priority key = (f, h, insertion_counter).
    Ties broken toward smaller h == larger g == "push deeper toward the goal".
    Without this, A* on open maps expands a huge equal-f plateau and the
    BFS/A* ratio becomes an artifact of dict/heap ordering.

Dependencies: numpy, matplotlib. ipywidgets optional (filmstrip fallback).
No pandas (deferred to CS35 per course design).
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

FREE, WALL = 0, 1
Cell = tuple  # (row, col)

# Fixed neighbor order => deterministic runs. Down, Right, Up, Left.
_DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))


class Grid(np.ndarray):
    """ndarray subclass that remembers a map's intended default goal.

    Plain arrays behave identically (view/slice/index all pass through);
    the only addition is a `default_goal` attribute that survives slicing,
    so a grid returned by generate_object_map or generate_maze tells
    bfs/A*/stepper which goal convention it was built for.
    """

    def __new__(cls, arr, default_goal: "Cell | None" = None):
        obj = np.asarray(arr).view(cls)
        obj.default_goal = default_goal
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.default_goal = getattr(obj, "default_goal", None)


def _default_goal(grid) -> Cell:
    """Bottom-right corner, unless `grid` carries its own convention
    (object maps stamp themselves with a centre goal -- see generate_object_map)."""
    n = grid.shape[0]
    return getattr(grid, "default_goal", None) or (n - 1, n - 1)


# ----------------------------------------------------------------------------
# 1. MAP GENERATORS
# ----------------------------------------------------------------------------

def neighbors(grid: np.ndarray, cell: Cell):
    """Free 4-neighbors of `cell`, in fixed order."""
    r, c = cell
    n_rows, n_cols = grid.shape
    for dr, dc in _DELTAS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < n_rows and 0 <= cc < n_cols and grid[rr, cc] == FREE:
            yield (rr, cc)


def _reachable(grid: np.ndarray, start: Cell, goal: Cell) -> bool:
    if grid[start] == WALL or grid[goal] == WALL:
        return False
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            return True
        for nxt in neighbors(grid, cur):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return False


def generate_object_map(n: int, density: float = 0.30, seed: int | None = None,
                        max_obj: int = 6, max_tries: int = 200) -> np.ndarray:
    """Open field littered with rectangular 'objects' (crates, buildings, walls).

    NOT i.i.d. per-cell noise: real obstacles are contiguous, and i.i.d. noise
    near the site-percolation threshold (p_c ~= 0.5927 for the square lattice)
    produces mazes-by-accident, which destroys the contrast with the maze family.

    `density` is target *wall fraction*, approximate (rectangles may overlap).
    Start and goal neighborhoods are always cleared. Regenerates until the goal
    is reachable; raises if it cannot after `max_tries`.

    Default goal is the CENTRE of the map, ((n-1)//2, (n-1)//2) -- stamped onto
    the returned grid so bfs/A*/stepper use it automatically (see `Grid`).
    """
    rng = np.random.default_rng(seed)
    start, goal = (0, 0), ( (n - 1)//2, (n - 1)//2)
    gr, gc = goal
    r0, r1 = max(0, gr - 1), min(n, gr + 2)
    c0, c1 = max(0, gc - 1), min(n, gc + 2)
    for _ in range(max_tries):
        grid = np.zeros((n, n), dtype=np.int8)
        target = density * n * n
        while grid.sum() < target:
            h = int(rng.integers(1, max_obj + 1))
            w = int(rng.integers(1, max_obj + 1))
            r = int(rng.integers(0, max(1, n - h)))
            c = int(rng.integers(0, max(1, n - w)))
            grid[r:r + h, c:c + w] = WALL
        grid[0:2, 0:2] = FREE      # clear start pocket
        grid[r0:r1, c0:c1] = FREE  # clear goal pocket (centred on `goal`)
        if _reachable(grid, start, goal):
            return Grid(grid, default_goal=goal)
    raise RuntimeError(f"No connected object map at n={n}, density={density}. Lower density.")


def generate_maze(n: int, seed: int | None = None, braid: float = 0.0) -> np.ndarray:
    """Perfect maze via randomized-DFS (recursive backtracker), carved on an odd grid.

    n is forced ODD (walls on even indices). Wall fraction lands near 0.5 and the
    corner-to-corner path is long and tortuous -- which is exactly why Manhattan
    distance is a bad estimate here.

    `braid` in [0,1]: fraction of dead ends to knock open, creating loops.
    braid=0 -> exactly one start->goal path. braid>0 -> multiple paths.

    Default goal is the BOTTOM-RIGHT corner (n-1, n-1) -- stamped onto the
    returned grid so bfs/A*/stepper use it automatically (see `Grid`).
    """
    if n % 2 == 0:
        n += 1
    rng = np.random.default_rng(seed)
    grid = np.ones((n, n), dtype=np.int8)

    stack = [(0, 0)]
    grid[0, 0] = FREE
    while stack:
        r, c = stack[-1]
        opts = []
        for dr, dc in _DELTAS:
            rr, cc = r + 2 * dr, c + 2 * dc
            if 0 <= rr < n and 0 <= cc < n and grid[rr, cc] == WALL:
                opts.append((rr, cc, r + dr, c + dc))
        if not opts:
            stack.pop()
            continue
        rr, cc, mr, mc = opts[int(rng.integers(len(opts)))]
        grid[mr, mc] = FREE
        grid[rr, cc] = FREE
        stack.append((rr, cc))

    if braid > 0:
        dead_ends = [(r, c) for r in range(n) for c in range(n)
                     if grid[r, c] == FREE and sum(1 for _ in neighbors(grid, (r, c))) == 1]
        rng.shuffle(dead_ends)
        for (r, c) in dead_ends[:int(braid * len(dead_ends))]:
            cands = []
            for dr, dc in _DELTAS:
                mr, mc, rr, cc = r + dr, c + dc, r + 2 * dr, c + 2 * dc
                if 0 <= rr < n and 0 <= cc < n and grid[mr, mc] == WALL and grid[rr, cc] == FREE:
                    cands.append((mr, mc))
            if cands:
                grid[cands[int(rng.integers(len(cands)))]] = FREE
    return Grid(grid, default_goal=(n - 1, n - 1))


# ----------------------------------------------------------------------------
# 2. SEARCH (instrumented, shared bookkeeping)
# ----------------------------------------------------------------------------

@dataclass
class SearchResult:
    strategy: str
    start: Cell
    goal: Cell
    path: list | None
    path_cost: int | None
    expanded: int              # nodes popped and expanded
    generated: int             # nodes pushed onto the frontier
    max_frontier: int
    trace: list = field(repr=False, default_factory=list)
    # trace[i] = (popped_cell, [cells newly pushed at step i])
    g_values: dict = field(repr=False, default_factory=dict)   # cell -> cost-so-far

    @property
    def solved(self) -> bool:
        return self.path is not None


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def null_heuristic(a: Cell, b: Cell):
    return 0


def _reconstruct(parent: dict, goal: Cell) -> list:
    path, cur = [goal], goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    return path[::-1]


def bfs(grid, start=None, goal=None, **_) -> SearchResult:
    start = start or (0, 0)
    goal = goal or _default_goal(grid)
    frontier = deque([start])
    parent = {start: None}
    depth = {start: 0}
    expanded, generated = 0, 1
    max_frontier = 1
    trace = []
    while frontier:
        cur = frontier.popleft()
        expanded += 1
        added = []
        if cur == goal:
            trace.append((cur, added))
            path = _reconstruct(parent, goal)
            return SearchResult("BFS", start, goal, path, len(path) - 1, expanded,
                                generated, max_frontier, trace, depth)
        for nxt in neighbors(grid, cur):
            if nxt not in parent:
                parent[nxt] = cur
                depth[nxt] = depth[cur] + 1
                frontier.append(nxt)
                generated += 1
                added.append(nxt)
        trace.append((cur, added))
        max_frontier = max(max_frontier, len(frontier))
    return SearchResult("BFS", start, goal, None, None, expanded, generated,
                        max_frontier, trace, depth)


def best_first(grid, start=None, goal=None, h=null_heuristic, w_g=1.0,
               label="A*", tie_break="h") -> SearchResult:
    """Generic best-first graph search.

    w_g = 1.0 -> A*      (priority = g + h)
    w_g = 0.0 -> Greedy  (priority = h)          [not optimal -- that's the point]
    h = lambda a,b: 0 -> Uniform-cost (== BFS on unit costs)
    tie_break: 'h' breaks equal-priority ties by h value (default);
    'order' breaks them by order put on the frontier (FIFO).
    """
    start = start or (0, 0)
    goal = goal or _default_goal(grid)
    counter = 0
    g = {start: 0}
    parent = {start: None}
    h0 = h(start, goal)
    tie_by_h = tie_break == "h"
    heap = [(w_g * 0 + h0, h0, counter, start)] if tie_by_h else [(w_g * 0 + h0, counter, start)]
    closed = set()
    expanded = 0
    generated = 1
    max_frontier = 1
    trace = []
    while heap:
        cur = heapq.heappop(heap)[-1]
        if cur in closed:
            continue                      # stale duplicate; not an expansion
        closed.add(cur)
        expanded += 1
        added = []
        if cur == goal:
            trace.append((cur, added))
            path = _reconstruct(parent, goal)
            return SearchResult(label, start, goal, path, g[cur], expanded,
                                generated, max_frontier, trace, g)
        for nxt in neighbors(grid, cur):
            if nxt in closed:
                continue
            ng = g[cur] + 1
            if nxt not in g or ng < g[nxt]:
                g[nxt] = ng
                parent[nxt] = cur
                counter += 1
                hn = h(nxt, goal)
                entry = (w_g * ng + hn, hn, counter, nxt) if tie_by_h else (w_g * ng + hn, counter, nxt)
                heapq.heappush(heap, entry)
                generated += 1
                added.append(nxt)
        trace.append((cur, added))
        max_frontier = max(max_frontier, len(heap))
    return SearchResult(label, start, goal, None, None, expanded, generated,
                        max_frontier, trace, g)


def astar(grid, h, **kw):
    return best_first(grid, h=h, w_g=1.0, label="A*", **kw)


def greedy(grid, h, **kw):
    return best_first(grid, h=h, w_g=0.0, label="Greedy", **kw)


# ----------------------------------------------------------------------------
# 3. STEPPER VISUALISATION  (Red Blob Games style)
# ----------------------------------------------------------------------------

def _state_at(grid, result: SearchResult, step: int, goal: Cell):
    """Replay the trace up to `step`; return (closed set, frontier set, parents).

    `parents[child]` is the node that pushed `child` (the current parent, since
    later relaxations overwrite earlier ones -- consistent with g_values)."""
    closed, frontier = set(), set()
    parents: dict = {result.start: None}
    for i in range(min(step + 1, len(result.trace))):
        popped, added = result.trace[i]
        closed.add(popped)
        frontier.discard(popped)
        for nxt in added:
            frontier.add(nxt)
            parents[nxt] = popped
    frontier -= closed
    return closed, frontier, parents


def _arrow_scale(n: int) -> int:
    """Arrow head size in points, auto-shrinking with map size."""
    return max(3, min(12, int(0.35 * (4000 / n) ** 0.5)))


def _draw_grid(grid, ax, start: Cell, goal: Cell):
    """Draw the walls/free cells plus start/goal markers. Shared by draw_step
    and the result-less stepper view."""
    from matplotlib.colors import ListedColormap

    base = np.where(grid == WALL, 0, 1).astype(float)  # 0=wall,1=free
    ax.imshow(base, cmap=ListedColormap(["#2b2b2b", "#f2f2f2"]), interpolation="nearest")
    ax.scatter([start[1]], [start[0]], marker="o", s=60, c="#00a878", zorder=5)
    ax.scatter([goal[1]], [goal[0]], marker="*", s=140, c="#d1495b", zorder=5)
    ax.set_xticks([]), ax.set_yticks([])


LABEL_MAX_N = 40  # frontier value labels only on maps at most this wide/tall


def draw_step(grid, result: SearchResult, step: int, ax=None,
              display: str | None = None, show_path: bool = True,
              show_value: bool = False, show_parent: bool = False):
    """Render one iteration.

    display: None (single uniform color) or one of 'h' (heuristic estimate),
    'g' (cost so far), 'f' (g+h) -- colours each frontier node by that quantity.
    h/f follow the strategy's heuristic, so for BFS h renders as 0 and f reduces to g;
    for Greedy f reduces to h.
    NOTE: 'f' is what actually drives A*'s
     pop order. Colour by 'h' to show what
    the heuristic *believes*; colour by 'f' to show what the algorithm *does*.

    show_value: annotate each frontier node with the value it is coloured by
    (`display` must be set). Auto-skipped on large maps (unreadable labels).
    show_parent: arrow from each visible node toward its parent (back to start).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n = grid.shape[0]
    start, goal = result.start, result.goal
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    closed, frontier, parents = _state_at(grid, result, step, goal)

    _draw_grid(grid, ax, start, goal)

    if closed:
        rr, cc = zip(*closed)
        ax.scatter(cc, rr, marker="s", s=max(4, 4000 / n), c="#b8c6db", linewidths=0)

    if frontier:
        rr, cc = zip(*frontier)
        gv = result.g_values
        fontsize = max(6, min(11, 300 // n))
        if display in ("g", "h", "f"):
            heur = null_heuristic if result.strategy == "BFS" else manhattan
            if display == "f" and result.strategy == "Greedy":
                display = "h"
            if display == "h":
                vals = [heur(cell, goal) for cell in frontier]
            elif display == "g":
                vals = [gv.get(cell, 0) for cell in frontier]
            else:  # 'f' -- the quantity that actually drives A*'s pop order
                vals = [gv.get(cell, 0) + heur(cell, goal) for cell in frontier]
            ax.scatter(cc, rr, marker="s", s=max(4, 4000 / n), c=vals, cmap="viridis_r",
                       norm=Normalize(min(vals), max(vals) + 1e-9), linewidths=0)
            if show_value and n <= LABEL_MAX_N:
                for cell, c_, r_, v in zip(frontier, cc, rr, vals):
                    ax.text(c_, r_, str(v), fontsize=fontsize, zorder=6,
                            ha="center", va="center",
                            bbox=dict(fc="white", alpha=0.6, ec="none", pad=0.5))
        else:
            ax.scatter(cc, rr, marker="s", s=max(4, 4000 / n), c="#b8c6db", linewidths=0)

    if show_parent:
        for cell in closed | frontier:
            par = parents.get(cell)
            if par is None:
                continue  # start cell (or a cell whose parent is not yet visible)
            r, c = cell
            pr, pc = par
            ax.annotate("", xy=(pc, pr), xytext=(c, r),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.0,
                                        mutation_scale=_arrow_scale(n)), zorder=4)

    if show_path and result.path:
        done = step >= len(result.trace) - 1
        if done:
            pr, pc = zip(*result.path)
            ax.plot(pc, pr, lw=2.0, color="#d1495b")

    ax.set_title(f"{result.strategy} | step {step+1}/{len(result.trace)} | "
                 f"expanded={min(step+1, result.expanded)}", fontsize=10)
    return ax


def filmstrip(grid, result: SearchResult, k: int = 6, display=None, **kw):
    """Static fallback: k evenly spaced snapshots. Works with zero widgets."""
    import matplotlib.pyplot as plt
    steps = np.linspace(0, len(result.trace) - 1, k).astype(int)
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.2))
    for ax, s in zip(np.atleast_1d(axes), steps):
        draw_step(grid, result, int(s), ax=ax, display=display, **kw)
    fig.tight_layout()
    return fig


def stepper(grid, result: SearchResult | None = None, display: str | None = None,
            show_value: bool = False, show_parent: bool = False):
    """Interactive slider if ipywidgets is present; otherwise a filmstrip.

    With no `result`, just shows the grid with the start and goal markers.
    display: None (single uniform colour) or one of 'g', 'h', 'f' -- colours
    frontier nodes by cost-so-far, heuristic estimate, or f (= g+h) value.
    show_value: annotate each frontier node with the value it is coloured by.
    show_parent: arrow from each visible node toward its parent (back to start).
    """
    if result is None:
        import matplotlib.pyplot as plt

        start, goal = (0, 0), _default_goal(grid)
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_grid(grid, ax, start, goal)
        ax.set_title("start / goal", fontsize=10)
        plt.show()
        return

    try:
        import ipywidgets as widgets
        from IPython.display import display as _ipydisplay
        import matplotlib.pyplot as plt
    except ImportError:
        return filmstrip(grid, result, display=display, show_value=show_value,
                         show_parent=show_parent)

    slider = widgets.IntSlider(0, 0, len(result.trace) - 1, 1,
                               description="step", continuous_update=False,
                               layout=widgets.Layout(width="100%", max_width="600px"))
    play = widgets.Play(min=0, max=len(result.trace) - 1, interval=60)
    widgets.jslink((play, "value"), (slider, "value"))
    out = widgets.Output()

    def _step(delta):
        def handler(_):
            slider.value = min(max(slider.value + delta, 0), slider.max)
        return handler

    prev_btn = widgets.Button(description="\u25c0 Previous",
                              tooltip="Previous step",
                              layout=widgets.Layout(width="105px"))
    next_btn = widgets.Button(description="Next step \u25b6",
                              tooltip="Next step",
                              layout=widgets.Layout(width="105px"))
    prev_btn.on_click(_step(-1))
    next_btn.on_click(_step(1))

    def render(change=None):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            draw_step(grid, result, slider.value, ax=ax, display=display, show_value=show_value,
                  show_parent=show_parent)
            plt.show()

    slider.observe(render, names="value")
    render()
    _ipydisplay(widgets.VBox([widgets.HBox([play, prev_btn, slider, next_btn]), out]))


# ----------------------------------------------------------------------------
# 4. SCALING EXPERIMENT HARNESS
# ----------------------------------------------------------------------------

def scaling_study(map_kind: str, ns, seeds=range(5), density=0.30, braid=0.0,
                  strategies=("BFS", "A*")) -> dict:
    """Run each strategy on `len(seeds)` independent maps per n.

    ONE map per n is not enough: the BFS/A* ratio has large map-to-map variance,
    and students will otherwise 'explain' noise. Returns raw per-run records so
    students compute their own medians.
    """
    fns = {"BFS": bfs, "A*": astar, "Greedy": greedy}
    out = {s: {"n": [], "expanded": [], "cost": [], "seed": []} for s in strategies}
    for n in ns:
        for seed in seeds:
            if map_kind == "object":
                grid = generate_object_map(n, density=density, seed=seed)
            elif map_kind == "maze":
                grid = generate_maze(n, seed=seed, braid=braid)
            else:
                raise ValueError(map_kind)
            for s in strategies:
                r = fns[s](grid)
                assert r.solved, f"unsolvable {map_kind} n={n} seed={seed}"
                out[s]["n"].append(grid.shape[0])
                out[s]["expanded"].append(r.expanded)
                out[s]["cost"].append(r.path_cost)
                out[s]["seed"].append(seed)
    return out


def median_by_n(record: dict) -> tuple:
    ns = sorted(set(record["n"]))
    med = [float(np.median([e for e, nn in zip(record["expanded"], record["n"]) if nn == n]))
           for n in ns]
    return ns, med


def ratio_by_n(study: dict, num="BFS", den="A*") -> tuple:
    ns, a = median_by_n(study[num])
    _, b = median_by_n(study[den])
    return ns, [x / y for x, y in zip(a, b)]


if __name__ == "__main__":
    import time
    for kind, ns in (("object", [21, 31, 41, 61, 81, 101]),
                     ("maze", [21, 31, 41, 61, 81, 101])):
        t0 = time.time()
        st = scaling_study(kind, ns, seeds=range(5))
        ns_, ratio = ratio_by_n(st)
        nb, mb = median_by_n(st["BFS"])
        na, ma = median_by_n(st["A*"])
        print(f"\n{kind.upper()}  ({time.time()-t0:.1f}s)")
        print(f"{'n':>5}{'BFS':>10}{'A*':>10}{'ratio':>8}{'A*/free':>9}")
        for i, n in enumerate(ns_):
            print(f"{n:>5}{mb[i]:>10.0f}{ma[i]:>10.0f}{ratio[i]:>8.2f}{ma[i]/(n*n):>9.3f}")
