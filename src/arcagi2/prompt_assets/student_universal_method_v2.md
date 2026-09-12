# ARC Reasoning Procedure V2

Infer one transformation rule from the solved demonstrations and apply it to the test input. The methods below are hypotheses only: use a method only when the demonstrations support its trigger.

## Core controller

1. Treat grid values `0-9` as categorical colors/symbols, not numerical quantities. Do not add, subtract, multiply, rank, or compare color IDs unless arithmetic is explicitly demonstrated.
2. Compare every solved input-output pair. Identify dimensions, invariants, and what changes: colors, cells, objects, positions, shapes, regions, repetition, or layout.
3. Test the simplest directly supported explanation first.
4. Verify a candidate against every solved demonstration. Reject it immediately if any required output cell or structure fails.
5. Do not repeatedly reconsider a rejected hypothesis unless new evidence gives a specific reason.
6. As soon as one rule reproduces all solved demonstrations exactly, STOP searching for alternatives, apply it to the query, verify the output, and answer.
7. Always finish with `<answer>[[...],[...],...]</answer>`.

## Priority checks

### Symbol/color substitution
Trigger: input/output dimensions and corresponding geometry are unchanged while symbols/colors change.
Infer a consistent `input_color -> output_color` mapping across all demonstrations and apply it cell-wise. Treat mappings as symbolic substitutions/permutations, never as arithmetic. If the mapping reproduces every demonstrated cell, stop.

### Simple geometric transformation
Trigger: clear structural change.
Test only directly supported operations such as crop/select, move/translate, copy, reflect, rotate, transpose, recolor by role, repeat/complete, expand, compress, or rearrange.

## Empirical method library

- **Marker-triggered local change:** If isolated markers repeatedly cause the same nearby edit, infer the relative offsets/local operation and apply it to corresponding markers.
- **Complete and propagate a motif:** If repeated regions contain partial versions of a shared pattern, infer the common motif and complete/propagate it while preserving demonstrated special cells.
- **Anchor and connector construction:** If objects act as endpoints/corners/anchors, infer and reproduce the demonstrated connection geometry.
- **Template stretching:** If the same topology appears at different sizes, preserve orientation/connectivity while stretching runs to the target region.
- **Mirrored regions:** If an axis/separator divides corresponding regions, reflect positions, directions, or displacements across it.
- **Partition and select:** If full rows/columns divide the grid into panels, infer which panel is selected and extract/transform it.
- **Geometric odd-one-out:** If several similar candidates differ systematically, test symmetry, rotation, connectivity, orientation, or another demonstrated invariant and select the unique candidate.
- **Position-encoded operation:** If marker position encodes direction, displacement, selected row/column, or another parameter, separate the instruction marker from the payload and apply the encoded operation.
- **Object-local editing:** If separate objects undergo the same edit, define the operation relative to each object's own bounding box rather than global coordinates.
- **Inner bounding-box exception:** If special cells inside a frame define an internal protected/exception region, infer the inner box and transform the surrounding region accordingly.
- **Obstacle-constrained connection:** If endpoints must connect around blocking cells, infer start, direction, and turns from the demonstrations; do not overwrite obstacles or targets.
- **Corner-state rotation:** If markers around a central object move among NW/NE/SE/SW corner roles, infer the corner-state transition and apply continuation relative to the new corner.
- **Frame gap as an outlet:** If a nearly complete frame has a meaningful gap, use the demonstrated gap direction as outlet/instruction information.
- **Periodic continuation at boundaries:** If a repeated motif reaches a boundary, infer period and phase first; allow the boundary to truncate the next repetition.
- **Repeated evidence versus noise:** If several hypotheses fit locally, prefer the relation supported independently across multiple examples/structures rather than a one-off coincidence.
- **Coarse reasoning followed by fine rendering:** If the task operates on a coarse symbolic grid rendered as larger pixel blocks, solve the coarse transformation first, then reproduce the demonstrated rendering exactly.
- **Count before packing or nesting:** If objects of several types/sizes are compacted or nested, count them first and infer the demonstrated capacity/packing rule before constructing output.
- **Region selection by marker evidence:** If barriers define connected regions and sparse markers distinguish them, infer connectivity and use marker evidence to choose the target region.
- **Symbolic quadrant compression:** If a full row and column divide a noisy grid into four regions and the output is compact, infer the symbolic property of each quadrant rather than copying pixel detail.

## Verification and termination

For each candidate, state it briefly and mentally execute it on every solved example. Check dimensions, positions, colors/symbols, spacing, symmetry, boundaries, and any temporary instruction markers. Prefer the simplest rule with no exceptions.

Avoid repeating observations or generating unrelated hypotheses. Once a rule exactly explains all demonstrations, stop reasoning, construct the query output, recheck every row/cell, and return exactly one rectangular JSON grid containing only integers `0-9` inside `<answer>...</answer>`.