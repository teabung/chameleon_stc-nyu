"""
Recursive grouping of a source design.

Algorithm:
  * Visit elements by increasing z_index, preserving input order for ties.
  * An unassigned element may be a child only if its z_index is strictly
    greater than its parent's and their rotated rectangular bound is contained by the parent.
  * Claim the child BEFORE recursing. The first eligible depth-first parent
    wins. Finish its descendants before considering the next sibling.
  * After recursion, bucket similar siblings by alignment AND size. Every
    child appears in exactly one Similarity instance, including singletons.

Conventions and limits:
  * The canvas root has id=None and z_index=None; it accepts all
    remaining elements, including negative layers and off-canvas elements.
  * Missing z_index means 0.
  * position is the element's center relative to its parent's original
    top-left, in canvas-axis pixels.
  * size is the original element size, not a union of descendant bounds.
  * Overlap uses rotated rectangular boxes, not image alpha or shape pixels.
    Edge contact and zero-area boxes do not count as positive-area overlap.
  * group_id is retained in element but does not override this requested
    overlap/z-index algorithm. This is an inferred tree, not a replacement
    for explicit designer groups when implementing the resizing engine.
  * Same-layer elements cannot be parent/child; they can be similar siblings.
  * The input is never modified. All original fields are deep-copied.
  * This performs grouping only, with no resizing or importance scoring.
  * Deeply nested inputs are subject to Python's normal recursion limit.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from math import cos, isfinite, radians, sin
from typing import Any, Iterator


@dataclass(eq=False)
class Similarity:
    """A bucket of similar sibling groups; it is not an ownership parent."""

    similar: list[Group] = field(default_factory=list)


@dataclass(eq=False)
class Group:
    id: str | None
    z_index: int | None
    size: tuple[float, float]
    position: tuple[float, float]
    absolute_position: tuple[float, float]
    parent: Group | None = field(default=None, repr=False)
    element: dict[str, Any] | None = field(default=None, repr=False)
    source_index: int | None = None
    members: list[Similarity] = field(default_factory=list, repr=False)

    @property
    def children(self) -> list[Group]:
        """Direct child Groups, without the Similarity wrappers."""
        return [child for bucket in self.members for child in bucket.similar]

    @property
    def source_bbox(self) -> tuple[float, float, float, float]:
        """Original unrotated (x, y, width, height), in canvas coordinates."""
        cx, cy = self.absolute_position
        width, height = self.size
        return cx - width / 2, cy - height / 2, width, height

    def walk(self) -> Iterator[Group]:
        """Yield this group and each descendant once."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe export: parent ID instead of a cyclic parent object.

        Do not use dataclasses.asdict: parent references would recurse forever.
        A null parent_id on a returned top-level group refers to the canvas.
        """
        return {
            "id": self.id,
            "z_index": self.z_index,
            "parent_id": self.parent.id if self.parent is not None else None,
            "size": list(self.size),
            "position": list(self.position),
            "absolute_position": list(self.absolute_position),
            "source_index": self.source_index,
            "element": deepcopy(self.element),
            "members": [
                {"similar": [child.to_dict() for child in bucket.similar]}
                for bucket in self.members
            ],
        }

@dataclass(frozen=True)
class Region:
    cx: float
    cy: float
    width: float
    height: float
    angle: float = 0.

    def sub(self, x, y, width, height):
        dx, dy = rotate(x + width/2 - self.width/2,
                        y + height/2 - self.height/2, self.angle)
        return Region(self.cx+dx, self.cy+dy, width, height, self.angle)


def groups_to_dict(groups: list[Group]) -> list[dict[str, Any]]:
    return [group.to_dict() for group in groups]

def _corners(element: dict[str, Any]) -> list[tuple[float, float]]:
    """Clockwise rotation about center in screen coordinates (positive y down)."""
    width, height = float(element["width"]), float(element["height"])
    cx = float(element["x"]) + width / 2
    cy = float(element["y"]) + height / 2
    angle = radians(float(element.get("rotation", 0)) % 360)
    c, s = cos(angle), sin(angle)
    return [
        (cx + dx * c - dy * s, cy + dx * s + dy * c)
        for dx, dy in (
            (-width / 2, -height / 2), (width / 2, -height / 2),
            (width / 2, height / 2), (-width / 2, height / 2),
        )
    ]

def _similar(a: Group, b: Group, alignment_tolerance: float,
             size_tolerance: float) -> bool:
    # Alignment is measured on original unrotated boxes.
    ax, ay, aw, ah = a.source_bbox
    bx, by, bw, bh = b.source_bbox
    equal_size = all(
        abs(v1 - v2) <= size_tolerance * max(v1, v2)
        for v1, v2 in ((aw, bw), (ah, bh))
    )
    aligned = any(
        abs(v1 - v2) <= alignment_tolerance
        for v1, v2 in (
            (ax, bx), (ax + aw / 2, bx + bw / 2), (ax + aw, bx + bw),
            (ay, by), (ay + ah / 2, by + bh / 2), (ay + ah, by + bh),
        )
    )
    return equal_size and aligned


def _bucket_similar(children: list[Group], enabled: bool,
                    alignment_tolerance: float,
                    size_tolerance: float) -> list[Similarity]:
    buckets: list[Similarity] = []
    for child in children:
        if enabled:
            for bucket in buckets:
                # Require similarity to EVERY member, preventing a chain of
                # pairwise matches from joining dissimilar endpoints.
                if all(_similar(child, other, alignment_tolerance, size_tolerance)
                       for other in bucket.similar):
                    bucket.similar.append(child)
                    break
            else:
                buckets.append(Similarity([child]))
        else:
            buckets.append(Similarity([child]))
    return buckets

def rotate(x, y, angle):
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return x*c-y*s, x*s+y*c


def region(element):
    return Region(element["x"]+element["width"]/2,
                  element["y"]+element["height"]/2,
                  element["width"], element["height"], element.get("rotation", 0))

def contains(parent, child, tolerance=1e-6):
    p = region(parent)
    for x, y in _corners(child):
        lx, ly = rotate(x-p.cx, y-p.cy, -p.angle)
        if abs(lx) > p.width/2+tolerance or abs(ly) > p.height/2+tolerance:
            return False
    return True



def build_containment_groups(source: dict[str, Any], *,
                             detect_similarity=True) -> list[Group]:
    """The existing DFS rule, with FULL rotated containment instead of overlap.

    Uses the original Group/Similarity classes. It does not mutate an existing
    tree. Explicit group_id is preserved, and checked by the arrangement stage.
    Out-of-canvas source elements are permitted as canvas-level children.
    """
    canvas = source["canvas"]
    w, h = canvas["width"], canvas["height"]
    root = Group(None, None, (w,h), (w/2,h/2), (w/2,h/2))
    ordered = sorted(enumerate(deepcopy(source["elements"])),
                     key=lambda pair: pair[1].get("z_index", 0))
    # Reserve explicit containment links before DFS. Otherwise a label inside
    # the canvas could be claimed before its slightly off-canvas rotated panel.
    preferred_parent = {}
    for ci,child in ordered:
        gid = child.get("group_id")
        if gid is None: continue
        possible = [(pi,p) for pi,p in ordered if p["id"] != child["id"]
                    and p.get("group_id") == gid and p["type"] != "text"
                    and (p.get("z_index",0),pi) < (child.get("z_index",0),ci)
                    and contains(p,child)]
        if possible:
            preferred_parent[child["id"]] = min(possible,key=lambda pair:
                (pair[1]["width"]*pair[1]["height"],pair[0]))[1]["id"]
    assigned = set()
    def populate(parent, depth=0):
        if depth > 64:
            raise ValueError("Hierarchy exceeds 64 levels")
        children = []
        px, py, _, _ = parent.source_bbox
        for index, e in ordered:
            if e["id"] in assigned:
                continue
            if e["id"] in preferred_parent and preferred_parent[e["id"]] != parent.id:
                continue
            if parent.element is not None and (
                (e.get("z_index",0) <= parent.z_index and
                 preferred_parent.get(e["id"]) != parent.id) or not contains(parent.element,e)
            ):
                continue
            r = region(e)
            child = Group(e["id"], e.get("z_index",0), (r.width,r.height),
                          (r.cx-px,r.cy-py), (r.cx,r.cy), parent, e, index)
            assigned.add(child.id)
            children.append(child)
            populate(child, depth+1)
        parent.members = _bucket_similar(children, detect_similarity, 2., .05)
    populate(root)
    return root.children