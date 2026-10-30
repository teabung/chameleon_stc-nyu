"""Recursive grouping of a CHAMELEON v0.9 source design (Python 3.10+).

Usage:
    groups = build_groups(source)       # list[Group], canvas's direct children
    print(json.dumps(groups_to_dict(groups), indent=2))

Command line (accepts either a source design or a task containing "source"):
    python recursive_groups.py input.json
    python recursive_groups.py input.json --no-similarity

Algorithm:
  * Visit elements by increasing z_index, preserving input order for ties.
  * An unassigned element may be a child only if its z_index is strictly
    greater than its parent's and their rectangular footprints overlap.
  * Claim the child BEFORE recursing. The first eligible depth-first parent
    wins. Finish its descendants before considering the next sibling.
  * After recursion, bucket similar siblings by alignment AND size. Every
    child appears in exactly one Similarity instance, including singletons.

Conventions and limits:
  * The synthetic canvas root has id=None and z_index=None; it accepts all
    remaining elements, including negative layers and off-canvas elements.
  * Missing z_index means 0 (a local convention; the schema has no default).
  * position is the element's CENTER relative to its parent's original
    top-left, in canvas-axis pixels. Parent rotation is not applied to axes.
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
import json
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


def groups_to_dict(groups: list[Group]) -> list[dict[str, Any]]:
    return [group.to_dict() for group in groups]


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result


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


def _overlap(a: list[tuple[float, float]],
             b: list[tuple[float, float]], epsilon: float) -> bool:
    """Separating-axis test for two nondegenerate rotated rectangles."""
    for polygon in (a, b):
        for i in (0, 1):
            x1, y1 = polygon[i]
            x2, y2 = polygon[i + 1]
            nx, ny = -(y2 - y1), x2 - x1
            length = (nx * nx + ny * ny) ** 0.5
            if length == 0:
                return False
            nx, ny = nx / length, ny / length
            pa = [x * nx + y * ny for x, y in a]
            pb = [x * nx + y * ny for x, y in b]
            if min(max(pa), max(pb)) - max(min(pa), min(pb)) <= epsilon:
                return False
    return True


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


def build_groups(
    source: dict[str, Any],
    *,
    detect_similarity: bool = True,
    alignment_tolerance: float = 2.0,
    size_tolerance: float = 0.05,
    overlap_epsilon: float = 1e-9,
) -> list[Group]:
    """Build the requested tree and return its top-level Group instances.

    Each returned group has the synthetic canvas as parent. Access the canvas
    with groups[0].parent, or wrap it in a list if a root-inclusive export is
    preferred. The function expects source, not the full task envelope.

    alignment_tolerance: pixels; corresponding edges or centers may differ
        by at most this amount.
    size_tolerance: fractional difference relative to the larger dimension;
        both width AND height must match within this threshold (0.05 = 5%).
    overlap_epsilon: pixel tolerance used by the rectangle intersection test.

    Input is expected to conform to the supplied schema. Geometry, IDs and
    layer values are checked here; this is not a full JSON Schema validator.
    The overlap phase is O(n^2); rotated corners are computed once per element.
    """
    alignment_tolerance = _number(alignment_tolerance, "alignment_tolerance", 0)
    size_tolerance = _number(size_tolerance, "size_tolerance", 0)
    overlap_epsilon = _number(overlap_epsilon, "overlap_epsilon", 0)
    if size_tolerance > 1:
        raise ValueError("size_tolerance must be between 0 and 1")
    canvas = source["canvas"]
    W = _number(canvas["width"], "canvas.width", 1)
    H = _number(canvas["height"], "canvas.height", 1)
    elements = deepcopy(source["elements"])
    if not isinstance(elements, list) or not elements:
        raise ValueError("source.elements must be a nonempty list")

    seen: set[str] = set()
    for element in elements:
        element_id = element["id"]
        if not isinstance(element_id, str) or element_id in seen:
            raise ValueError(f"Invalid or duplicate element ID: {element_id!r}")
        seen.add(element_id)
        for name in ("x", "y", "width", "height"):
            _number(element[name], f"{element_id}.{name}",
                    0 if name in ("width", "height") else None)
        _number(element.get("rotation", 0), f"{element_id}.rotation")
        z = element.get("z_index", 0)
        if isinstance(z, bool) or not isinstance(z, int):
            raise ValueError(f"{element_id}.z_index must be an integer")

    # Sort only layer values that exist: gaps, large values and negative
    # values require no integer-by-integer search.
    ordered = sorted(enumerate(elements), key=lambda item: item[1].get("z_index", 0))
    footprints = {element["id"]: _corners(element) for element in elements}
    assigned: set[str] = set()
    root = Group(None, None, (W, H), (W / 2, H / 2), (W / 2, H / 2))

    def populate(parent: Group) -> None:
        children: list[Group] = []
        parent_x, parent_y, _, _ = parent.source_bbox

        for source_index, element in ordered:
            element_id = element["id"]
            if element_id in assigned:
                continue

            z = element.get("z_index", 0)
            if parent.element is not None:
                if z <= parent.z_index:
                    continue
                if not _overlap(footprints[parent.id], footprints[element_id],
                                overlap_epsilon):
                    continue

            width, height = float(element["width"]), float(element["height"])
            cx = float(element["x"]) + width / 2
            cy = float(element["y"]) + height / 2
            child = Group(
                id=element_id,
                z_index=z,
                size=(width, height),
                position=(cx - parent_x, cy - parent_y),
                absolute_position=(cx, cy),
                parent=parent,
                element=element,
                source_index=source_index,
            )

            # Claim, attach, then recurse: no second parent can claim it.
            assigned.add(element_id)
            children.append(child)
            populate(child)

        # Post-order similarity pass: child subtrees are already complete.
        parent.members = _bucket_similar(
            children, detect_similarity, alignment_tolerance, size_tolerance
        )

    populate(root)
    return root.children