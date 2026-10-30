"""
Deterministic, importance-aware arrangement.

Only x, y, width, height and an existing text font_size are edited.
No model is trained, downloaded, or silently substituted for human supervision.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import itertools
import logging
import math
import re
from typing import Any, Protocol, Sequence
import unicodedata

from .groupformation import Group, Similarity, _bucket_similar, _corners

LOG = logging.getLogger(__name__)
EPS = 1e-7
EDITABLE = frozenset({"x", "y", "width", "height"})
FEATURE_NAMES = ("text_prominence", "contrast", "sqrt_area",
                 "centrality", "visibility", "contrast_available")

class TextMeasurer(Protocol):
    """Return unwrapped text width in pixels, including style_runs/spacing.

    start is the fragment's character offset in the unchanged content string.
    Implementations should use the same fonts, shaping and metrics as rendering.
    """
    exact: bool
    def __call__(self, element: dict[str, Any], text: str,
                 font_size: float, start: int) -> float: ...


class ConservativeTextMeasurer:
    """Dependency-free estimate, NOT a font renderer or legibility certificate."""
    exact = False

    def __call__(self, element, text, font_size, start=0):
        units = 0.
        for char in text:
            if unicodedata.combining(char):
                continue
            if char.isspace():
                units += .36
            elif unicodedata.east_asian_width(char) in ("W", "F"):
                units += 1.1
            elif char in "il.,:;!'|":
                units += .36
            elif char in "MW@%":
                units += 1.02
            else:
                units += .69 if char.isupper() else .61
        # Conservative allowance for bold/style runs and unknown font metrics.
        return max(0., units * font_size * 1.08 +
                   max(0, len(text)-1) * float(element.get("letter_spacing", 0)))


@dataclass
class Config:
    min_font_size: float = 12. # minimum font size that ensures readability of the design
    margin_ratio: float = .025
    gap_ratio: float = .025
    max_candidates: int = 72 # number of different predicted layout candidates
    max_depth: int = 64 # max z-index of the design (used as termination condition)
    max_font_scale: float = 2.
    preserve_image_aspect: bool = True


@dataclass
class ArrangementResult:
    design: dict[str, Any]
    report: dict[str, Any]


class ConstraintConflict(ValueError):
    """The supplied group constraints cannot be satisfied as configured."""


class LayoutInfeasibleError(ConstraintConflict):
    def __init__(self, message: str, result: ArrangementResult | None = None):
        super().__init__(message)
        self.result = result


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


@dataclass
class Node:
    group: Group
    element: dict[str, Any]
    parent: str | None
    buckets: list[list[str]]
    background: bool = False
    decoration: bool = False
    behavior: str = "container"


@dataclass(frozen=True)
class Plan:
    mode: str
    alignment: str = "center"
    weighting: float = .5
    similarity_mode: str = "auto"
    font_multiplier: float = 1.


def finite(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


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


def aabb(element):
    corners = _corners(element)
    xs, ys = zip(*corners)
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(poly):
    return abs(sum(x*y2-x2*y for (x,y),(x2,y2)
                   in zip(poly, poly[1:]+poly[:1])))/2 if poly else 0.


def intersection_area(a, b):
    """Convex polygon clipping: actual rotated rectangles, not their AABBs."""
    poly = _corners(a)
    clip = _corners(b)
    for p, q in zip(clip, clip[1:]+clip[:1]):
        old, poly = poly, []
        if not old:
            break
        def side(v):
            return (q[0]-p[0])*(v[1]-p[1])-(q[1]-p[1])*(v[0]-p[0])
        for u, v in zip(old, old[1:]+old[:1]):
            su, sv = side(u), side(v)
            if su >= -EPS:
                poly.append(u)
            if (su > EPS and sv < -EPS) or (su < -EPS and sv > EPS):
                t = su/(su-sv)
                poly.append((u[0]+t*(v[0]-u[0]), u[1]+t*(v[1]-u[1])))
    return polygon_area(poly)

def prepare_tree(source, groups, config):
    """Accept roots OR a flat collection including descendants; reject ambiguity."""
    src = {e["id"]: e for e in source["elements"]}
    objects, paths = {}, set()
    def visit(g, depth):
        if depth > config.max_depth:
            raise ConstraintConflict("Hierarchy exceeds max_depth")
        if id(g) in paths:
            raise ConstraintConflict("Cycle in Group hierarchy")
        if g.id not in src or g.element is None:
            raise ConstraintConflict(f"Unknown/non-drawable group {g.id!r}")
        if g.id in objects:
            if objects[g.id] is not g:
                raise ConstraintConflict(f"Different Group instances use ID {g.id}")
            return
        if g.element != src[g.id]:
            raise ConstraintConflict(f"Stale Group.element for {g.id}; rebuild groups from source")
        objects[g.id] = g
        paths.add(id(g))
        child_ids = [c.id for c in g.children]
        if len(child_ids) != len(set(child_ids)):
            raise ConstraintConflict(f"Duplicate child in {g.id}")
        for c in g.children:
            if c.parent is not g:
                raise ConstraintConflict(f"Inconsistent parent pointer for {c.id}")
            visit(c, depth+1)
        paths.remove(id(g))
    for g in groups:
        visit(g, 0)
    if set(objects) != set(src):
        raise ConstraintConflict(f"Group/source IDs differ: missing={sorted(set(src)-set(objects))}")
    roots = [g for g in objects.values() if g.parent is None or g.parent.element is None]
    if objects and not roots:
        raise ConstraintConflict("Hierarchy has no canvas-level roots")
    nodes = {}
    for eid,g in objects.items():
        parent_id = g.parent.id if g.parent is not None and g.parent.element is not None else None
        if parent_id is not None:
            if parent_id not in objects or all(c is not g for c in objects[parent_id].children):
                raise ConstraintConflict(f"Missing ownership link for {eid}")
            if not contains(src[parent_id], src[eid]):
                raise ConstraintConflict(f"{eid} is not fully inside {parent_id}. "
                                         "Rebuild with build_containment_groups(source).")
        e = src[eid]
        cw,ch = source["canvas"]["width"],source["canvas"]["height"]
        bg = (e["type"] in ("image","shape") and abs(e["x"])<EPS and abs(e["y"])<EPS
              and abs(e["width"]-cw)<EPS and abs(e["height"]-ch)<EPS
              and abs(e.get("rotation",0) % 360)<EPS
              and all(e.get("z_index",0) < o.get("z_index",0)
                      for o in src.values() if o["id"] != eid))
        decor = e["type"] == "shape" and e.get("opacity",1) <= .25
        children = [[c.id for c in b.similar] for b in g.members if b.similar]
        gids = {src[c.id].get("group_id") for c in g.children}
        composite = bool(g.children) and (e["type"] == "shape" or
                     (e.get("group_id") is not None and gids == {e["group_id"]}))
        behavior = "composite" if composite else "container"
        nodes[eid] = Node(g,e,parent_id,children,bg,decor,behavior)
    # The canvas's Similarity buckets also matter, even when roots were supplied.
    parents = {id(g.parent):g.parent for g in roots if g.parent is not None}
    if len(parents) == 1:
        parent = next(iter(parents.values()))
        buckets = [[c.id for c in b.similar] for b in parent.members if b.similar]
        flattened = [eid for b in buckets for eid in b]
        if set(flattened) != {g.id for g in roots} or len(flattened) != len(set(flattened)):
            raise ConstraintConflict("Canvas Similarity buckets do not match roots")
    else:
        if any(len(b.similar)>1 for p in parents.values() for b in p.members):
            raise ConstraintConflict("Multiple synthetic roots contain ambiguous Similarities")
        buckets = [[g.id] for g in roots]
    # Explicit designer groups must be connected in this supplied tree. Splitting
    # a same-layer button into independent siblings would lose its association.
    explicit = {}
    for eid,n in nodes.items():
        if n.element.get("group_id") is not None:
            explicit.setdefault(n.element["group_id"], []).append(eid)
    for gid,ids in explicit.items():
        tops = [eid for eid in ids if nodes[eid].parent not in ids]
        if len(ids)>1 and len(tops)!=1:
            raise ConstraintConflict(f"Explicit group_id {gid!r} is split in supplied hierarchy")
    def check_buckets(bs):
        for bucket in bs:
            ratios = [nodes[eid].element["width"]/nodes[eid].element["height"]
                      for eid in bucket if aspect_locked(nodes[eid],config)
                      and nodes[eid].element["height"]>EPS]
            if ratios and max(ratios)-min(ratios)>1e-6:
                raise ConstraintConflict("Equal-size Similarity conflicts with locked aspect "
                                         f"ratios: {bucket}")
    check_buckets(buckets)
    for n in nodes.values():
        check_buckets(n.buckets)
    return nodes, buckets


def aspect_locked(node, config):
    return (config.preserve_image_aspect and node.element["type"] == "image" and not node.background)


def rgba(value):
    if not isinstance(value,str):
        return None
    match = re.fullmatch(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)",value)
    if not match:
        return None
    r,g,b = [float(v)/255 for v in match.groups()[:3]]
    alpha = float(match.group(4) or 1)
    if not all(0<=v<=1 for v in (r,g,b,alpha)):
        return None
    return r,g,b,alpha


def luminance(color):
    rgb = [v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in color[:3]]
    return sum(v*w for v,w in zip(rgb,(.2126,.7152,.0722)))


def extract_importance(source, nodes, config):
    w = source["canvas"]["width"]
    h = source["canvas"]["height"]
    max_font = max([e.get("font_size", 0) for e in source["elements"]] + [1])
    features, evidence = {}, {}
    for eid, node in nodes.items():
        element = node.element
        # Use contrast only when both colors are known and fully opaque.
        contrast = None
        foreground = rgba(element.get("color"))
        parent = nodes.get(node.parent)
        background = (
            rgba(parent.element.get("fill_color"))
            if parent is not None
            else None
        )
        if (
            foreground is not None
            and background is not None
            and foreground[3] == 1
            and background[3] == 1
            and parent.element.get("opacity", 1) == 1
            and element.get("opacity", 1) == 1
        ):
            foreground_luminance = luminance(foreground)
            background_luminance = luminance(background)

            contrast_ratio = (max(foreground_luminance, background_luminance) + 0.05) / (min(foreground_luminance, background_luminance) + 0.05)
            contrast = math.log(contrast_ratio) / math.log(21)
        cx = (element["x"] + element["width"] / 2) / w
        cy = (element["y"] + element["height"] / 2) / h
        centrality = max(0.0,1.0 - min(1.0,math.hypot(cx - 0.5, cy - 0.5) / math.sqrt(0.5)))
        text_prominence = (
            element.get("font_size", 0) / max_font
            if element["type"] == "text"
            else 0.0
        )
        relative_area = min(1.0, element["width"] * element["height"] / (w * h))

        row = [
            text_prominence,
            contrast if contrast is not None else 0.0,
            math.sqrt(relative_area),
            centrality,
            1.0,  # Neutral placeholder; actual visibility is unknown.
            float(contrast is not None),
        ]
        features[eid] = [
            max(0.0, min(1.0, float(value)))
            for value in row
        ]
        evidence[eid] = {
            "contrast_missing": contrast is None,
            "visibility_missing": True,
        }

    # Deterministic scoring
    scores = {}
    for eid, row in features.items():
        contrast_available = bool(row[5])
        weights = [
            0.20,                                  # Text prominence
            0.20 if contrast_available else 0.0,   # Contrast
            0.15,                                  # Square root of area
            0.10,                                  # Centrality
            0.0,                                   # Unknown visibility
            0.0,                                   # Availability flag
        ]
        # Renormalize over available evidence.
        scores[eid] = sum(
            value * weight
            for value, weight in zip(row, weights)
        ) / sum(weights)
    # Propagate maximum descendant importance to each parent.
    def aggregate(eid):
        child_scores = [
            aggregate(child_id)
            for bucket in nodes[eid].buckets
            for child_id in bucket
        ]
        scores[eid] = max([scores[eid]] + child_scores)
        return scores[eid]
    for eid, node in nodes.items():
        if node.parent is None:
            aggregate(eid)
    return scores, {
        eid: {
            "features": dict(zip(FEATURE_NAMES, features[eid])),
            "score": scores[eid],
            **evidence[eid],
        }
        for eid in nodes
    }


def text_layout(e, width, font, measure):
    """Greedy wrapping, including explicit newlines and breakable long tokens.

    Layout-only line breaks never change content. Renderer integration should
    replace this routine too if its line-breaking policy differs.
    """
    lines, offset, overflow = [], 0, False
    for paragraph in e["content"].split("\n"):
        line, start = "", offset
        for match in re.finditer(r"\s+|\S+",paragraph):
            token = match.group()
            token_start = offset+match.start()
            candidate = line+token
            if line and measure(e,candidate.rstrip(),font,start)>width+EPS:
                lines.append((line.rstrip(),start))
                line,start = "",token_start
                token = token.lstrip()
            if measure(e,token,font,token_start)>width+EPS:
                for i,char in enumerate(token):
                    if line and measure(e,line+char,font,start)>width+EPS:
                        lines.append((line.rstrip(),start))
                        line,start = "",token_start+i
                    line += char
                    if measure(e,line,font,start)>width+EPS:
                        overflow = True
            else:
                line += token
        lines.append((line.rstrip(),start))
        offset += len(paragraph)+1
    cap = e.get("max_lines")
    if cap is not None and len(lines)>cap: overflow = True
    height = len(lines)*font*e.get("line_height",1.2)
    return {"lines":len(lines),"height":height,"width_overflow":overflow}


def fit_font(e, width, height, desired, measure):
    def fits(f):
        layout = text_layout(e,width,f,measure)
        return not layout["width_overflow"] and layout["height"]<=height+EPS
    if not e["content"]:
        return max(1.,desired)
    if not fits(1.):
        return 1.
    lo,hi = 1.,max(1.,desired)
    for _ in range(23):
        mid = (lo+hi)/2
        if fits(mid): lo = mid
        else: hi = mid
    return lo


class Engine:
    def __init__(self, source, nodes, buckets, target, scores, config, measure):
        self.source,self.nodes,self.buckets = source,nodes,buckets
        self.target,self.scores,self.config,self.measure = target,scores,config,measure
        self.scale = math.sqrt(target["width"]*target["height"] /
                               (source["canvas"]["width"]*source["canvas"]["height"]))
        self.output = {}

    def desired_font(self,e,plan):
        return min(e["font_size"]*self.config.max_font_scale,
                   max(self.config.min_font_size, e["font_size"]*self.scale)*plan.font_multiplier)

    def fit(self,e,slot, *, dims=None):
        w,h = dims if dims else (slot.width,slot.height)
        w,h = max(EPS,w),max(EPS,h)
        angle = e.get("rotation",0)-slot.angle
        c,s = abs(math.cos(math.radians(angle))),abs(math.sin(math.radians(angle)))
        factor = min(1.,slot.width/max(EPS,c*w+s*h),slot.height/max(EPS,s*w+c*h))
        w,h = max(EPS,w*factor),max(EPS,h*factor)
        return slot.cx-w/2,slot.cy-h/2,w,h

    def put(self,eid,slot,plan,shared=None,defer_children=False):
        n,e = self.nodes[eid],self.nodes[eid].element
        out = deepcopy(e)
        if shared is not None:
            dims = shared
        elif n.background:
            dims = (slot.width,slot.height)
        elif aspect_locked(n,self.config):
            dims = (max(EPS,e["width"]),max(EPS,e["height"]))
            k = min(slot.width/dims[0],slot.height/dims[1])
            if e["type"] == "image": k = min(k,self.scale*1.25)
            dims = (dims[0]*k,dims[1]*k)
        elif n.behavior == "composite":
            ar = e["width"]/max(EPS,e["height"])
            dims = (slot.width,min(slot.height,max(self.config.min_font_size*2.6,
                                                  slot.width/max(EPS,ar))))
            dims = (min(dims[0],dims[1]*ar),dims[1])
        else:
            dims = None
        x,y,w,h = self.fit(e,slot,dims=dims)
        if e["type"] == "text":
            f = fit_font(e,w,h,self.desired_font(e,plan),self.measure)
            if shared is None:
                used = min(h,text_layout(e,w,f,self.measure)["height"])
                h = max(EPS,used)
                y = slot.cy-h/2
            out["font_size"] = max(1.,f)
        out.update(x=x,y=y,width=w,height=h)
        self.output[eid] = out
        if not n.buckets or defer_children:
            return
        else:
            r = region(out)
            if n.behavior == "composite":
                # Padding accommodates unchanged strokes and label readability.
                pad = min(r.width,r.height)*.04
                r = r.sub(pad,pad,max(EPS,r.width-2*pad),max(EPS,r.height-2*pad))
            self.layout(n.buckets,r,plan)

    def slots(self,region_,count,mode,weights,alignment):
        if not count: return []
        r = region_
        gap = min(r.width,r.height)*self.config.gap_ratio
        if mode == "auto": mode = "row" if r.width/r.height>1.7 else "column"
        if mode.startswith("grid"):
            columns = min(count,max(1,int(mode[4:])))
            rows = math.ceil(count/columns)
            gap = min(gap,r.width/(columns*4),r.height/(rows*4))
            w,h = (r.width-gap*(columns-1))/columns,(r.height-gap*(rows-1))/rows
            return [r.sub((i%columns)*(w+gap),(i//columns)*(h+gap),w,h) for i in range(count)]
        horizontal = mode == "row"
        total = r.width if horizontal else r.height
        gap = min(gap,total/(count*4))
        available = max(EPS,total-gap*(count-1))
        allocations = [available*w/sum(weights) for w in weights]
        slots,offset = [],0.
        for v in allocations:
            slots.append(r.sub(offset,0,v,r.height) if horizontal else r.sub(0,offset,r.width,v))
            offset += v+gap
        return slots

    def layout(self,buckets,region_,plan):
        # Reserve regions jointly before visiting layers; later global candidate
        # validation can reject the whole plan and revise earlier allocations.
        decorations, regular = [],[]
        pending = list(buckets)
        while pending:
            b = pending.pop(0)
            if len(b)==1 and self.nodes[b[0]].background:
                # Full-canvas backgrounds are surfaces, not space-consuming
                # foreground units. Plan their children jointly with independent
                # canvas-level units, retaining actual ownership/containment.
                self.put(b[0],region_,plan,defer_children=True)
                pending.extend(self.nodes[b[0]].buckets)
            elif len(b)==1 and self.nodes[b[0]].decoration: decorations.append(b)
            else: regular.append(b)
        for b in decorations:
            eid = b[0]
            e = self.nodes[eid].element
            parent = self.nodes.get(self.nodes[eid].parent)
            old = parent.element if parent else {"x":0,"y":0,**self.source["canvas"]}
            sx,sy = region_.width/max(EPS,old["width"]),region_.height/max(EPS,old["height"])
            w,h = e["width"]*min(sx,sy),e["height"]*min(sx,sy)
            w,h = min(region_.width,w),min(region_.height,h)
            x = max(0,min(region_.width-w,(e["x"]-old["x"])*sx))
            y = max(0,min(region_.height-h,(e["y"]-old["y"])*sy))
            self.put(eid,region_.sub(x,y,max(EPS,w),max(EPS,h)),plan)
        if not regular: return
        margin = min(region_.width,region_.height)*self.config.margin_ratio
        r = region_.sub(margin,margin,max(EPS,region_.width-2*margin),max(EPS,region_.height-2*margin))
        regular.sort(key=lambda b:min((self.nodes[e].element["y"],self.nodes[e].element["x"])
                                      for e in b))
        if plan.mode == "source":
            slots = []
            for bucket in regular:
                es = [self.nodes[eid].element for eid in bucket]
                x0,y0 = min(e["x"] for e in es),min(e["y"] for e in es)
                x1,y1 = max(e["x"]+e["width"] for e in es),max(e["y"]+e["height"] for e in es)
                parent = self.nodes.get(self.nodes[bucket[0]].parent)
                old = region(parent.element) if parent else Region(
                    self.source["canvas"]["width"]/2,self.source["canvas"]["height"]/2,
                    self.source["canvas"]["width"],self.source["canvas"]["height"])
                sx,sy = r.width/max(EPS,old.width),r.height/max(EPS,old.height)
                w,h = min(r.width,max(EPS,(x1-x0)*sx)),min(r.height,max(EPS,(y1-y0)*sy))
                x = max(0,min(r.width-w,(x0-(old.cx-old.width/2))*sx))
                y = max(0,min(r.height-h,(y0-(old.cy-old.height/2))*sy))
                slots.append(r.sub(x,y,w,h))
        elif plan.mode == "feature":
            # Generic image-plus-content template; no particular element ID or
            # fixed badge count is assumed. Also works for portrait canvases.
            images = [i for i,b in enumerate(regular) if len(b)==1 and
                      self.nodes[b[0]].element["type"] == "image"]
            hero = max(images,key=lambda i:self.nodes[regular[i][0]].element["width"]*
                       self.nodes[regular[i][0]].element["height"]) if images else None
            if hero is not None and len(regular)>1:
                gap = min(r.width,r.height)*self.config.gap_ratio
                if r.width>r.height*1.5:
                    v = min(r.width*.42,r.height*1.2)
                    hero_slot = r.sub(0,0,v,r.height)
                    rest = r.sub(v+gap,0,max(EPS,r.width-v-gap),r.height)
                else:
                    v = r.height*.4
                    hero_slot = r.sub(0,0,r.width,v)
                    rest = r.sub(0,v+gap,r.width,max(EPS,r.height-v-gap))
                rest_slots = iter(self.slots(rest,len(regular)-1,"column",[1.]*(len(regular)-1),plan.alignment))
                slots = [hero_slot if i==hero else next(rest_slots) for i in range(len(regular))]
            else:
                slots = self.slots(r,len(regular),"auto",[1.]*len(regular),plan.alignment)
        else:
            weights = [(0.2+max(self.scores[e] for e in b))**plan.weighting for b in regular]
            slots = self.slots(r,len(regular),plan.mode,weights,plan.alignment)
        work = []
        for bucket,slot in zip(regular,slots):
            work.extend(self.bucket_assignments(bucket,slot,plan))
        # Layer-by-layer placement; Similarity regions/dimensions already shared.
        for eid,slot,shared in sorted(work,key=lambda v:self.nodes[v[0]].element.get("z_index",0)):
            self.put(eid,slot,plan,shared)

    def bucket_assignments(self,bucket,slot,plan):
        if len(bucket)==1: return [(bucket[0],slot,None)]
        mode = plan.similarity_mode
        if mode == "auto":
            mode = "row" if slot.width>slot.height*1.4 else "column"
        if mode == "grid": mode = "grid"+str(math.ceil(math.sqrt(len(bucket))))
        cells = self.slots(slot,len(bucket),mode,[1.]*len(bucket),plan.alignment)
        # One shared size variable for the WHOLE bucket, including rotations.
        locked = [self.nodes[e] for e in bucket if aspect_locked(self.nodes[e],self.config)]
        ratio = (locked[0].element["width"]/max(EPS,locked[0].element["height"])) if locked else None
        if ratio:
            w,h = cells[0].width,cells[0].width/ratio
        else:
            w,h = cells[0].width,cells[0].height
            # Compact repeated button rows/columns instead of giant rectangles.
            ratios = [self.nodes[e].element["width"]/max(EPS,self.nodes[e].element["height"]) for e in bucket]
            h = min(h,max(self.config.min_font_size*2.6,w/(sum(ratios)/len(ratios))))
        factors = []
        for eid,cell in zip(bucket,cells):
            a = math.radians(self.nodes[eid].element.get("rotation",0)-cell.angle)
            c,s = abs(math.cos(a)),abs(math.sin(a))
            factors.append(min(1.,cell.width/max(EPS,c*w+s*h),cell.height/max(EPS,s*w+c*h)))
        factor = min(factors)
        shared = max(EPS,w*factor),max(EPS,h*factor)
        # Equal raw box sizes plus common centers along each chosen row/column.
        return [(eid,cell,shared) for eid,cell in zip(bucket,cells)]

    def candidate(self,plan):
        self.output = {}
        r = Region(self.target["width"]/2,self.target["height"]/2,
                   self.target["width"],self.target["height"])
        self.layout(self.buckets,r,plan)
        design = deepcopy(self.source)
        design["canvas"].update(self.target)
        design["elements"] = [self.output[e["id"]] for e in self.source["elements"]]
        return design

    def baseline(self,stretch=False):
        sw,sh = self.source["canvas"]["width"],self.source["canvas"]["height"]
        tw,th = self.target["width"],self.target["height"]
        sx,sy = tw/sw,th/sh
        if not stretch: sx = sy = min(sx,sy)
        dx,dy = (tw-sw*sx)/2,(th-sh*sy)/2
        d = deepcopy(self.source)
        d["canvas"].update(self.target)
        for e in d["elements"]:
            e.update(x=e["x"]*sx+dx,y=e["y"]*sy+dy,width=e["width"]*sx,height=e["height"]*sy)
            if e["type"] == "text": e["font_size"] = max(1.,e["font_size"]*min(sx,sy))
        return d


def preserved_fields(source,design):
    """Independent allowlist check; never sanitize output by dropping fields."""
    if set(source)!=set(design) or len(source["elements"])!=len(design["elements"]):
        return False
    if set(source["canvas"])!=set(design["canvas"]): return False
    if any(source["canvas"][k]!=design["canvas"][k] for k in source["canvas"]
           if k not in ("width","height")): return False
    for original,changed in zip(source["elements"],design["elements"]):
        allowed = EDITABLE | ({"font_size"} if original["type"]=="text" else set())
        if set(original)!=set(changed): return False
        if any(original[k]!=changed[k] for k in original if k not in allowed): return False
    return all(source[k]==design[k] for k in source if k not in ("canvas","elements"))


def evaluate(source,design,nodes,buckets,scores,config,measure):
    es = {e["id"]:e for e in design["elements"]}
    canvas = {"x":0,"y":0,**design["canvas"]}
    hard,readability,collisions = [],[],[]
    if not preserved_fields(source,design): hard.append("immutable field changed")
    weighted_font,weight_sum,scale_loss,move_loss = 0.,0.,0.,0.
    visual_quality,visual_weights = 0.,0.
    similarity_checks = []
    def check(bs,parent_angle=0):
        for b in bs:
            if len(b)<2: continue
            ws,hs = [es[e]["width"] for e in b],[es[e]["height"] for e in b]
            if max(ws)-min(ws)>1e-5 or max(hs)-min(hs)>1e-5:
                hard.append(f"Similarity unequal size: {b}")
            centers = [rotate(region(es[e]).cx,region(es[e]).cy,-parent_angle) for e in b]
            xs,ys = [p[0] for p in centers],[p[1] for p in centers]
            row = max(ys)-min(ys)<1e-5
            col = max(xs)-min(xs)<1e-5
            # Grid rows have >=2 elements, except a possible trailing singleton.
            def clusters(vals):
                result = []
                for v in sorted(vals):
                    if not result or abs(v-result[-1])>1e-5: result.append(v)
                return result
            ux,uy = clusters(xs),clusters(ys)
            grid = len(ux)>1 and len(uy)>1 and len(ux)*len(uy)>=len(b)
            # A grid must have repeated axes, no arbitrary diagonal scatter.
            grid = grid and len(ux)<len(b) and len(uy)<len(b)
            if not (row or col or grid): hard.append(f"Similarity unaligned: {b}")
            similarity_checks.append({"ids":b,"equal_size":max(ws)-min(ws)<=1e-5 and max(hs)-min(hs)<=1e-5,
                                      "alignment":"row" if row else "column" if col else "grid" if grid else "invalid"})
    check(buckets)
    for eid,n in nodes.items():
        e = es[eid]
        if any(not math.isfinite(e[k]) for k in EDITABLE) or min(e["width"],e["height"])<=0:
            hard.append(f"Invalid geometry: {eid}")
            continue
        if not contains(canvas,e): hard.append(f"Off-canvas: {eid}")
        if n.parent and not contains(es[n.parent],e): hard.append(f"Outside parent: {eid}")
        if aspect_locked(n,config) and n.element["width"]>EPS and n.element["height"]>EPS:
            original_ratio = n.element["width"]/n.element["height"]
            if abs(math.log((e["width"]/e["height"])/original_ratio))>1e-6:
                hard.append(f"Aspect ratio changed: {eid}")
        check(n.buckets,e.get("rotation",0))
        s = scores[eid]
        if not n.background and not n.decoration and (e["type"] == "image" or n.behavior == "composite"):
            expected = n.element["width"]*n.element["height"]*(
                design["canvas"]["width"]*design["canvas"]["height"] /
                (source["canvas"]["width"]*source["canvas"]["height"]))
            visual_quality += s*min(1.,math.sqrt(e["width"]*e["height"]/max(EPS,expected)))
            visual_weights += s
        if e["type"]=="text":
            layout = text_layout(e,e["width"],e["font_size"],measure)
            if e["font_size"]<config.min_font_size-1e-4 or layout["height"]>e["height"]+1e-4 or layout["width_overflow"]:
                readability.append({"id":eid,"font_size":e["font_size"],
                                    "below_minimum":e["font_size"]<config.min_font_size-1e-4,
                                    "overflow":layout["height"]>e["height"]+1e-4 or layout["width_overflow"]})
            desired = max(config.min_font_size,n.element["font_size"]*math.sqrt(
                design["canvas"]["width"]*design["canvas"]["height"] /
                (source["canvas"]["width"]*source["canvas"]["height"])))
            weighted_font += s*min(1.,e["font_size"]/desired)
            weight_sum += s
        if not n.background and not n.decoration:
            old = n.element
            old_c,new_c = region(old),region(e)
            move_loss += s*((old_c.cx/source["canvas"]["width"]-new_c.cx/design["canvas"]["width"])**2+
                            (old_c.cy/source["canvas"]["height"]-new_c.cy/design["canvas"]["height"])**2)
            scale_loss += s*abs(math.log(max(EPS,e["width"]*e["height"])/max(EPS,old["width"]*old["height"])))
    def ancestors(eid):
        result = set()
        while nodes[eid].parent is not None:
            eid = nodes[eid].parent
            result.add(eid)
        return result
    ancestry = {eid:ancestors(eid) for eid in nodes}
    for aid,bid in itertools.combinations(nodes,2):
        a,b = nodes[aid],nodes[bid]
        if aid in ancestry[bid] or bid in ancestry[aid] or a.background or b.background:
            continue
        overlap = intersection_area(es[aid],es[bid])
        if overlap<EPS: continue
        if a.decoration or b.decoration: continue
        # Only decoration and ancestor composites receive an overlap exemption.
        # Existing text-on-text collisions are not treated as design intent.
        denom = max(EPS,min(es[aid]["width"]*es[aid]["height"],es[bid]["width"]*es[bid]["height"]))
        ratio = overlap/denom
        if ratio>.005: collisions.append({"ids":[aid,bid],"overlap_fraction":ratio})
    font_quality = weighted_font/max(EPS,weight_sum)
    visual_quality = visual_quality/visual_weights if visual_weights else 1.
    # Common target-quality objective across ALL templates. Candidate-specific
    # references cannot give uniform-fit an artificial zero-distortion win.
    loss = (100.*len(readability) + 40.*sum(c["overlap_fraction"] for c in collisions)
            + 8.*(1-font_quality) + 16.*(1-visual_quality) + .05*move_loss + .005*scale_loss)
    return {"hard_failures":hard,"readability_failures":readability,
            "unintended_overlaps":collisions,"font_quality":font_quality,
            "foreground_scale_quality":visual_quality,
            "movement_penalty":move_loss,"scale_penalty":scale_loss,
            "similarities":similarity_checks,"objective":loss,
            "feasible":not hard and not readability and not collisions}


def arrange_with_report(source: dict[str,Any], groups: list[Group],
                        target_canvas: dict[str,Any], *,
                        config: Config | None = None,
                        text_measurer: TextMeasurer | None = None) -> ArrangementResult:
    """Search a bounded set of layouts; return design and separate diagnostics.

    Strict geometry/identity/Similarity constraints are never relaxed. 
    A geometry-valid best effort may have small/overflowing
    text; report.feasible is then false and a warning is logged. This API makes
    no claim to a global optimum, human preference, or measured contrast when
    fonts/assets/rendering are unavailable.
    """
    config = config or Config()
    measure = text_measurer or ConservativeTextMeasurer()
    finite(config.min_font_size,"min_font_size",1)
    finite(config.max_font_scale,"max_font_scale",EPS)
    for name in ("margin_ratio","gap_ratio"):
        if not 0<=finite(getattr(config,name),name)<.25: raise ValueError(f"Invalid {name}")
    if config.max_candidates<1: raise ValueError("max_candidates must be positive")
    target = {}
    for k in ("width","height"):
        v = target_canvas[k]
        if isinstance(v,bool) or not isinstance(v,int) or v<1:
            raise ValueError("target_canvas width/height must be positive integers")
        target[k] = v
    nodes,buckets = prepare_tree(source,groups,config)
    scores,features = extract_importance(source,nodes,config)
    engine = Engine(source,nodes,buckets,target,scores,config,measure)
    candidates,summary = [],[]
    def consider(name,d,eligible=True):
        metrics = evaluate(source,d,nodes,buckets,scores,config,measure)
        summary.append({"name":name,**metrics,"eligible_for_selection":eligible})
        if eligible and not metrics["hard_failures"]: candidates.append((metrics["objective"],name,d,metrics))
    consider("uniform_fit",engine.baseline())
    consider("stretch_comparison",engine.baseline(True),False)
    # Fair coverage under the budget: try every mode before extra variants.
    modes = ["source","row","column","grid2","grid3","grid4","feature"]
    plans = [Plan(mode,weighting=weighting,similarity_mode=sim,font_multiplier=font_mult)
             for weighting,font_mult,sim in itertools.product((.5,0.,1.),(1.,1.25),("auto","row","column","grid"))
             for mode in modes]
    budget = min(len(plans),config.max_candidates)
    indices = sorted({round(i*(len(plans)-1)/max(1,budget-1)) for i in range(budget)})
    for index in indices:
        plan = plans[index]
        consider(f"{plan.mode}:weight={plan.weighting}:font={plan.font_multiplier}:similarity={plan.similarity_mode}",
                 engine.candidate(plan))
    if not candidates:
        raise LayoutInfeasibleError("No candidate satisfies geometry, containment, aspect and Similarity constraints")
    _,name,design,metrics = min(candidates,key=lambda c:(not c[3]["feasible"],c[0]))
    warnings = []
    if not getattr(measure,"exact",False):
        warnings.append("Font files/renderer not supplied: text fitting uses an estimate; verify in the actual renderer.")
    warnings.append("Asset pixels are not loaded: image visibility, transparency and actual contrast are unverified. Missing assets remain referenced unchanged.")
    if not metrics["feasible"]:
        warnings.append("No selected fully readable, collision-free result; returning the best geometry-valid candidate.")
    report = {"selected_candidate":name,**metrics,"candidate_count":len(summary),
              "scorer":"deterministic_heuristic",
              "text_measurement":"renderer_supplied" if getattr(measure,"exact",False) else "approximate",
              "warnings":warnings,"group_importance":features,"candidates":summary}
    result = ArrangementResult(design,report)
    if not metrics["feasible"]: LOG.warning(warnings[-1])
    return result


def arrange(source: dict[str,Any], groups: list[Group],
            target_canvas: dict[str,Any], **kwargs) -> dict[str,Any]:
    """Return {canvas, elements}; diagnostics never enter the design JSON."""
    return arrange_with_report(source,groups,target_canvas,**kwargs).design
