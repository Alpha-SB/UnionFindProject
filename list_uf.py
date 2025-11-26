"""
list_uf.py

List-based Union-Find (Disjoint Set Union) implementation.

Representation:
- Each set is stored as a "header" id and a member list.
- head[i] = header id of the set containing element i.
- members[header] = list of all elements in that set.
- size[header] = size of the set (valid only at header indices).

Weighted-union heuristic:
- Always merge the smaller list into the larger list.

Instrumentation:
- head_updates: number of times an element's head pointer changes.
"""

from __future__ import annotations


class ListUF:
    def __init__(self, n: int):
        """
        Initialize n singleton sets {0}, {1}, ..., {n-1}.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        self.n = n
        # head[i] = header id of the set containing i
        self.head = list(range(n))
        # size[h] = size of set whose header is h
        self.size = [1] * n
        # members[h] = list of members in the set whose header is h
        self.members = [[i] for i in range(n)]

        # instrumentation
        self.head_updates = 0

    # ------------- core operations -------------

    def find(self, x: int) -> int:
        """
        Return the header (representative) of the set containing x.
        Complexity: O(1)
        """
        if x < 0 or x >= self.n:
            raise IndexError("x out of range")
        return self.head[x]

    def union(self, x: int, y: int) -> int:
        """
        Union the sets containing x and y.
        Uses weighted union: append smaller list into larger list.
        Returns the header of the resulting set.

        Amortized complexity: O(log n) across a sequence of operations.
        """
        hx = self.find(x)
        hy = self.find(y)

        if hx == hy:
            return hx

        # ensure hx is the larger set
        if self.size[hx] < self.size[hy]:
            hx, hy = hy, hx

        # move all members from hy into hx
        for v in self.members[hy]:
            self.head[v] = hx
            self.head_updates += 1
            self.members[hx].append(v)

        self.members[hy].clear()
        self.size[hx] += self.size[hy]
        self.size[hy] = 0

        return hx

    # ------------- convenience / instrumentation helpers -------------

    def connected(self, x: int, y: int) -> bool:
        """
        Return True if x and y are in the same set.
        """
        return self.find(x) == self.find(y)

    def num_sets(self) -> int:
        """
        Return the number of non-empty sets.
        """
        return sum(1 for h in range(self.n) if self.size[h] > 0)

    def max_set_size(self) -> int:
        """
        Return the size of the largest set.
        """
        return max(self.size) if self.size else 0
