"""
tree_uf.py

Tree-based Union-Find (Disjoint Set Union) implementation.

Representation:
- Each element has a parent pointer.
- Roots are representatives of sets.
- Optionally tracks size or rank for weighted union.

Heuristics:
- union_by_size: attach smaller tree under larger.
- union_by_rank: attach smaller rank under larger rank.
- path_compression: during find, flatten paths.

Instrumentation:
- parent_updates: number of parent pointer changes.
"""

from __future__ import annotations


class TreeUF:
    def __init__(self, n: int, use_rank: bool = False, path_compression: bool = True):
        """
        Initialize n singleton sets {0}, {1}, ..., {n-1}.

        :param use_rank: if True, use union-by-rank; otherwise union-by-size.
        :param path_compression: if True, compress paths during find.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        self.n = n
        self.parent = list(range(n))
        self.size = [1] * n       # used when use_rank == False
        self.rank = [0] * n       # used when use_rank == True

        self.use_rank = use_rank
        self.path_compression = path_compression

        # instrumentation
        self.parent_updates = 0

    # ------------- core operations -------------

    def find(self, x: int) -> int:
        """
        Find with optional path compression.

        If path_compression is True, compresses the path from x to its root.
        Otherwise, standard "find root" without compression.
        """
        if x < 0 or x >= self.n:
            raise IndexError("x out of range")

        # find root
        r = x
        while self.parent[r] != r:
            r = self.parent[r]

        if not self.path_compression:
            return r

        # path compression
        while self.parent[x] != r:
            p = self.parent[x]
            self.parent[x] = r
            self.parent_updates += 1
            x = p

        return r

    def union(self, x: int, y: int) -> int:
        """
        Union the sets containing x and y.

        - If use_rank is False: union-by-size
        - If use_rank is True: union-by-rank

        Returns root of the resulting set.
        """
        rx = self.find(x)
        ry = self.find(y)

        if rx == ry:
            return rx

        if self.use_rank:
            # union-by-rank
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            self.parent_updates += 1
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
        else:
            # union-by-size
            if self.size[rx] < self.size[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            self.parent_updates += 1
            self.size[rx] += self.size[ry]

        return rx

    # ------------- convenience / instrumentation helpers -------------

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

    def num_sets(self) -> int:
        """
        Count number of distinct roots.
        """
        roots = set(self.find(i) for i in range(self.n))
        return len(roots)

    def _depth_of(self, x: int) -> int:
        """
        Depth of node x (distance to root), without path compression.
        Mainly used for instrumentation (do not call inside hot loops).
        """
        depth = 0
        while self.parent[x] != x:
            x = self.parent[x]
            depth += 1
        return depth

    def max_depth(self) -> int:
        """
        Compute maximum depth over all nodes (expensive; use sparingly).
        """
        return max(self._depth_of(i) for i in range(self.n))
