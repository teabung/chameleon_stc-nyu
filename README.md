# CHAMELEON — our approach

## Overview:

Convert a source design to a target canvas by combining recursive DFS grouping, importance scoring, and constrained layout search.
  - Input:
      design directory: `dict[src, Any]`
      target canvas size: `dict[src, Any]` {width and height}
  - Output:
      new design directory: `dict[src, Any]`

## The pipeline:

Grouping -> Importance scoring -> Layout search -> Output best design layout

1. Recursive grouping

The algorithm builds a hierarchy using overlap and stacking order (z-index)
  1. visit elements in increasing z-index
  2. an element that is fully contained by the current element become a child of the current element
  3. Mark the child assigned, ensuring that the child has one parent
  4. Make recursion call on the child to complete the child before looking for the next child of the current element
  5. After building the children, collect similiar sibllings into `Similarity` instances using shared alignment and (approximately) equal size.

This generates the list of `Group` instances.

2. Importance scoring

`prepare_tree()` indexes the hierarchy by element ID and keeps parent links and Similarity groups. `extract_importance()` builds one numerical row per Group's own element: text_prominence, contrast, sqrt_area, centrality, visibility, contrast_available.

The default scorer uses heuristic weights. Predictions are normalized within sibling set. Each parent's importance score becomes the maximum of its own and its descendants' scores. This prevents rewarding groups simply for having many members.

Importance influences space allocation and candidate evaluation. We use deterministic scoring based on the position and size of the element. This scores will be used when making inferences via ML interface (uniform-fit).

3. Layout search

The engine tries up to 72 alternative layouts + uniform-fit + diagnostic stretch baselines. Each candicate include placement relative to source, rows, columns, grids, and arrangements. It allocates regions for groups by layer, and recursively fits children inside parents.

Elements characterized by the same `Similarity` instance share exact width and height and alignment (can be row, column, or even grid).

Candidate layout is validated by checking Similarity constraints, text fit, and unwanted overlaps across layers. Feasible candidates take priority. Then, it selects the best candidate (lowest validation loss).

4. Output

Only element's `x`, `y`, `width`, `height`, and `font_size` change. Canvas dimensions become the target dimensions. All other properties are preserved.

## Usage

### Quick start:

In Chameleon main directory, run

`python -m engine.main`

results are stored under output directory.

### Configuration:

in `arrangement.py`,
```python
class Config:
    min_font_size: float
    margin_ratio: float
    gap_ratio: float
    max_candidates: int
    max_depth: int
    max_font_scale: float
    preserve_image_aspect: bool
```

`min_font_size`
- The minimum readable text size, in pixels. Setting this to 12 px ensures that the text generated in any design layout is readable.

`margin_ratio`
- Padding around the foreground layout inside the current canvas.

`gap_ratio`
- Preferred spacing between arranged units. Engine may reduce this when necessary to fit design objects.

`max_candidates`
- Maximum number of alterative layouts to evaluate. More candidates increase search time and may improve the result.

`max_depth`
- Maximum parent-child nesting depth when validating the Group hierarchy.

`max_font_scale`
- Caps the desired font size. if max_font_scale = 2, the original 20 px font has an upper limit of 40 px.

`preserve_image_aspect`
- Keeps foreground images' width-to-height ratios, avoiding stretching.