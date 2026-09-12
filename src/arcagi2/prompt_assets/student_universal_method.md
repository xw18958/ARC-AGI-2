# ARC Solving Guide

Use this guide as a reasoning checklist for a new ARC task. Infer the rule from the solved examples first. The pattern library below contains possible hypotheses only; do not force a task to match a listed pattern unless the examples support it.

## 1. Understand the task structure

- Identify the likely background color.
- Record input and output dimensions.
- Detect connected objects, isolated markers, frames, separators, repeated panels, repeated motifs, and obvious symmetry.
- Treat colors as temporary roles such as background, object, marker, frame, instruction, or payload. Do not assume a fixed meaning for a color across tasks.
- Note whether the output preserves the input size or instead crops, expands, compresses, or rearranges it.

## 2. Compare all solved examples

For every solved input-output pair, determine:

- what is preserved;
- what disappears;
- what moves;
- what is copied;
- what is recolored;
- what is reflected, rotated, translated, stretched, cropped, repeated, completed, or reordered;
- whether the same change is defined in absolute coordinates or relative to an object, frame, separator, panel, or bounding box.

Look for a rule that explains all solved examples, not just one.

## 3. Represent the problem at the right level

Before predicting individual cells, decide what the meaningful units are.

Possible units include:

- connected objects;
- object bounding boxes;
- rows or columns;
- quadrants or panels;
- symmetry axes;
- corners or anchor points;
- repeated motifs;
- coarse cells that later expand into larger blocks.

If literal sizes vary across demonstrations, compare topology, orientation, relative position, and role rather than exact pixel size.

## 4. Generate a small number of candidate rules

Prefer simple transformations supported directly by the demonstrations. Common possibilities include:

- select or crop a region;
- copy or move an object;
- reflect, rotate, transpose, or translate;
- recolor according to object role;
- repeat or complete a motif;
- stretch a template into a target region;
- connect anchors;
- use one object or region as an instruction for another;
- apply a cyclic shift or permutation;
- select the unique panel that satisfies or violates a geometric property.

Do not enumerate many unrelated rules. Focus on the few hypotheses suggested by the visible structure.

## 5. Infer which features are variables

Separate the general rule from task-specific parameters.

Possible parameters include:

- object color;
- marker location;
- frame dimensions;
- direction;
- distance;
- repetition period;
- number of objects;
- symmetry type;
- selected region.

When the same role uses different colors, sizes, or positions across examples, reason from the role rather than memorizing the literal value.

## 6. Verify before applying

Mentally execute the candidate rule on every solved example.

Reject or revise a rule if it fails any required output structure or cell.

When several rules seem possible:

- prefer the one supported by repeated evidence;
- prefer the one requiring fewer exceptions;
- check boundary cases and truncated patterns;
- distinguish meaningful repeated evidence from one-off coincidences.

Only after a rule explains all available demonstrations should it be applied to the query.

## 7. Construct the query output carefully

- Determine the output dimensions first.
- Apply the same role mapping and transformation used in the solved examples.
- Preserve invariant cells.
- Remove temporary markers, seeds, or instruction objects when the demonstrations show that they should disappear.
- Respect boundaries, spacing, repetition periods, and relative coordinates.

Before returning the answer, recheck:

- output height and width;
- every row length;
- colors;
- object positions;
- repeated spacing;
- symmetry;
- boundary behavior;
- whether any temporary instruction content was accidentally left in the output.

---

## Useful transformation patterns

Use these only when the task evidence matches them.

### Marker-triggered local change
A marker can trigger a fixed neighborhood operation. Infer the relative offsets from the solved examples and apply the same local operation independently to every matching marker.

### Complete and propagate a motif
Repeated panels may contain incomplete versions of the same motif. A more complete panel can reveal the common pattern, which is then completed or propagated to the other panels while preserving demonstrated special cells.

### Anchor and connector construction
Objects can act as anchors or corners of a larger structure. Repeated horizontal or vertical seed patterns may specify how to connect those anchors.

### Template stretching
A small object can encode topology rather than literal size. Its orientation and connectivity can be preserved while its runs are stretched to fit a frame or target region.

### Mirrored regions
A separator can define corresponding regions by reflection. A direction or displacement encoded on one side may need the reflected direction on the other side.

### Partition and select
Full separator rows or columns can divide a grid into panels. A marker, exceptional property, or dominant feature may identify which panel should be extracted or transformed.

### Odd-one-out by geometric property
When several candidate panels are shown, test simple invariants such as horizontal, vertical, diagonal, or rotational symmetry. The required output may be the unique candidate that differs.

### Position-encoded operation
The position of a marker can encode an operation parameter such as shift amount, direction, or selected row/column. Separate the instruction part from the payload before transforming the payload.

### Object-local trimming or editing
When several separate objects undergo the same change, define the operation relative to each object's own bounding box rather than global image coordinates.

### Inner bounding-box exception
Special markers inside a frame can define a smaller bounding box that is preserved while the surrounding framed region is filled or otherwise transformed.

### Obstacle-constrained connection
When one segment must reach another and a third color acts as an obstacle, extend from the open end in the demonstrated direction and turn only as needed through background cells without overwriting obstacles or the target.

### Corner-state rotation
For a central block with markers around its corners, marker roles may rotate among northwest, northeast, southeast, and southwest positions. Any outward continuation should follow the marker's new corner.

### Frame gap as an outlet
A missing cell in an otherwise complete frame can indicate an outlet direction. The task may transform the frame interior and extend content outward through that gap.

### Periodic continuation at boundaries
For a periodic motif, infer the underlying phase before considering the crop. A grid boundary can truncate the next repetition, so partial edge motifs may still be correct.

### Repeated evidence versus noise
If several colors or objects accidentally satisfy a local clue, compare how often each candidate satisfies the same relation. Repeated independent evidence is stronger than a single coincidence.

### Coarse reasoning followed by fine rendering
Some tasks define the transformation on a coarse grid and then render each coarse cell as a fixed-size block. Infer the coarse structure first, then reproduce the demonstrated rendering exactly.

### Count before packing or nesting
When objects of several sizes are compacted or nested, count each type first. Use those counts and the demonstrated nesting capacity to determine the output structure.

### Region selection by marker evidence
When one color forms barriers and another sparse color appears inside the resulting regions, define connectivity using the barrier color and compare marker evidence across regions to identify the target region.

### Symbolic quadrant compression
When a full row and column divide a noisy grid into four regions, the output may summarize the dominant role or color of each quadrant rather than reproduce pixel detail. The separator may become the central row and column of the compact output.
