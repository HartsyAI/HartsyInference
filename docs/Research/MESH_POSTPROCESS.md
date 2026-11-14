# Mesh post-processing and PBR baking

Scope: the model-independent toolkit in `HartsyInference.ThreeD` (`Geometry/Ops`, `Geometry`, `Io`) that turns a raw triangle mesh
plus a sparse voxel attribute volume into a decimated, UV-unwrapped mesh with baked PBR textures and a GLB. It replaces the
standard path of `to_glb` in `microsoft/TRELLIS.2` `o-voxel/o_voxel/postprocess.py` (commit `75fbf018`), which needs CUDA-only
cumesh, nvdiffrast, flex_gemm and OpenCV. Nothing here depends on a model.

## Chain

`PbrMeshPostProcessor.Process`: weld and fill holes (snapshot into `TriangleBvh`) -> `MeshDecimator.DecimateStaged` (3x target,
clean, target, clean, unify orientation) -> `UvUnwrapper` -> `TextureBaker` -> `Inpainter` -> `PbrMaterial` -> `PbrGlbWriter`.
Every stage is a public type of its own and usable alone.

## Algorithms and upstream mapping

| Upstream | Here | Notes |
|---|---|---|
| cumesh `simplify` | `MeshDecimator`: Garland-Heckbert quadric edge collapse | Link condition keeps the mesh manifold and its genus; boundary edges get penalty planes and boundary vertices never move; a collapse that turns a neighbor face past `MinNormalDot` is refused. Deterministic: ties break on edge keys. |
| `remove_duplicate_faces`, `repair_non_manifold_edges`, `remove_small_connected_components`, `fill_holes`, `unify_face_orientations` | `MeshCleaner` | Non-manifold edges are repaired by splitting vertices, as upstream's docstring says ("duplicates with the same coordinates"): the consistently wound, largest pair of faces stays joined, the others are cut off along the edge with copies of its two vertices, and no face is removed (`RemoveNonManifoldFaces` is the variant that drops them). Components are vertex-connected and judged by area ratio. Holes fill as a triangle or a centroid fan. Orientation floods across manifold edges, then flips closed components with negative signed volume (extra). |
| cumesh `uv_unwrap` | `UvUnwrapper` | Charts grow face by face (`UvChartGrower`): a face joins when its normal is within the cone of the chart's projection normal and its projected triangle overlaps nothing in the chart, so a chart is valid by construction. Small charts dissolve into large neighbours at a wider cone. Planar projection per chart (mean normal when that stays valid), MaxRects (up to 256 charts) or shelf packing at uniform texel density, with a gutter that never drops below `MinPaddingTexels`. |
| cuBVH `unsigned_distance` | `TriangleBvh` | Median-split BVH, exact closest point and barycentrics, thread-safe queries. |
| nvdiffrast UV rasterize | `Rasterizer` | CPU, banded over rows, lowest triangle index wins on overlap, so threading never changes the result. |
| flex_gemm `grid_sample_3d(trilinear)` | `SparseAttributeSampler` | See conventions. |
| cv2 `inpaint(INPAINT_TELEA)` | `Inpainter` | Telea fast marching with gradient-corrected weights on float data. |
| trimesh GLB export | `PbrGlbWriter` | Embedded PNGs, own PNG encoder (the engine's lives in a package that depends on ThreeD). |

## Conventions

- **Voxel centers.** Voxel `i` spans `[i, i + 1)` voxel units from `SparseAttributeVolume.Origin`; its center is `i + 0.5`.
  World position `p` maps to grid position `(p - Origin) / VoxelSize`. A sample blends the eight centers around it with tent
  weights `1 - |center - p|`; empty neighbors are dropped and the remaining weights renormalized; no occupied neighbor gives zeros
  (and an uncovered texel). This follows `flex_gemm/ops/grid_sample/grid_sample_torch.py`. That file truncates corner coordinates
  with `.int()`, which differs from `floor` only for samples within half a voxel of the grid's minimum faces; this code uses
  `floor`, which is what the CUDA kernel behind `grid_sample_3d(mode='trilinear')` runs (`floor(q - 0.5)`, FlexGEMM `6dd94a85`).
- **UV and texel.** Texel `(x, y)` samples UV `((x + 0.5) / W, (y + 0.5) / H)`; image row 0 is first in the PNG and v = 0. This is
  glTF's convention, so the writer emits UVs as they are. Upstream flips V in `to_glb`, but trimesh flips it again on glTF export,
  so the file's UV equals the internal one (inferred from trimesh behavior, not re-read from its source). `PbrGlbOptions.FlipV`
  exists for callers whose UVs have v pointing up.
- **Z-up to Y-up.** `PbrGlbOptions.ConvertZUpToYUp` applies `(x, y, z) -> (x, z, -y)` to positions and normals (a rotation, so
  winding is unchanged). It is off by default; `Trellis2ImageTo3DPipeline.ToGlb` sets it, and `DoubleSided`.
- **Material packing.** Texture 0 is base color RGB, with alpha in the fourth channel when the material has alpha; texture 1 is
  `R = 0, G = roughness, B = metallic`; both factors 1. `AlphaMode` defaults to OPAQUE as upstream does.
- **Attribute range.** Volume attributes are expected in [0,1] and are quantized as `clamp(round(v * 255))`.
- **Units.** Hole perimeter (`3e-2`) and weld tolerance are absolute mesh units, as upstream; callers scale the mesh to about one
  unit or change the options.

## Differences from upstream

- Remesh (narrow-band dual contouring) is not ported; only the standard path is.
- Inpainting runs on float data before quantization (upstream quantizes first), with radius 3 for base color and 1 for the rest.
- Texels whose eight volume neighbors are all empty are treated as uncovered and inpainted; upstream leaves them zero.
- Upstream marks the material double-sided unless remeshed; here `DoubleSided` defaults to false and the caller chooses.
- The cleaner also welds identical vertices and drops degenerate faces, and its orientation step can flip a closed component outward.
- Pinholes: the raw dual-grid surface keeps thousands of tiny boundary loops that the perimeter fill does not close, and they render as dark specks in a textured result (the same dots appear with no texture at all). `PbrPostProcessOptions.OpeningWidthRatio` (0.005 of the bounding-box diagonal by default, 0 restores the old behaviour) also makes the cleans between the decimation passes zip shut every boundary loop that triangulates within that width. On the saved 512 bust at 500,000 faces the welded boundary edges fell from 16,824 to 3,444 and the specks on the hair, forehead and shirt are gone; a width of 0.1 closes more but draws long crack lines, so it stays opt-in on the shape-only finisher. Upstream's own examples and Gradio app avoid the holes by running `to_glb` with `remesh=True`, which this engine does not port.
- Output is geometry-only after cleaning and decimation; normals are recomputed (area-weighted) before unwrapping and carried
  across seams.

## Why the shape mesh is not watertight, measured

Measured on the TRELLIS.2 512 bust (seed 1), raw mesher output of 736,637 vertices and 1,487,440 triangles, by vertex index unless stated:

| Stage | Boundary edges | Non-manifold edges | Largest boundary loop (bbox = 1) |
|---|---|---|---|
| Raw `DualGridMesher` output | 18,889 (4,887 loops, 4,856 of them under 0.03) | 30,844 | 0.165 |
| After the first clean (weld, hole fill 0.03) | 14,565 | 30,845 | |
| Decimation to 1M alone | 14,493 | 30,845 | 0.165 |
| Face-dropping repair, then decimation (earlier behaviour) | 38,634 | 20 | **25.8** (12,789 edges) |
| Vertex-splitting repair, then decimation (now) | 26,482 (the split copies sit at the same positions; one loop of 9.5 and 5,220 edges is a chain of them) | 0 | 9.5 by index, closed once welded |
| Final mesh after UV unwrap, vertices welded by position (now) | 12,545 | 10,881 | |

- The raw field itself is not closed: the dual grid emits a quad only where all four neighbouring voxels exist, so a missing neighbour leaves a
  small hole (about 4,900 loops, nearly all under 3% of the bounding box), and a voxel that several surfaces cross gives edges with more than
  two faces. `flexible_dual_grid_to_mesh` makes the same quads by its source (same
  neighbour offsets, same split rule); running it on the decoded field to confirm identical triangles is a separate check.
- Our cleaner made it worse: `RepairNonManifoldEdges` dropped faces, which cut cracks along the non-manifold chains (a 12,789-edge loop of
  perimeter 25.8 on the bust, visible as dark lines across the jacket). Upstream's `repair_non_manifold_edges` splits vertices instead. It does now, and
  checks its own result: when the pairs kept at neighbouring edges of a chain join all the faces of one edge back together at both ends, that edge is
  cut as well (its kept pair included, which can leave a one-edge slit where keeping a pair was possible) and the grouping redone, once, and no edge keeps
  more than two faces afterwards: a corner has two edges at its vertex and each gives it at most one join, so the corners of a vertex group into
  paths and cycles, and the faces of a cut edge are path ends, at most two per group (it assumes no degenerate faces, which `Clean` removes first;
  tested on all 19,683 chain patterns and 3,000 random junction meshes).
  The pair kept at an edge is the consistently wound one, then the larger by area (the area term used to be lost to a `1e30` bonus added to it).
- Fins cut off by the repair are vertex-disconnected from the surface, so `RemoveSmallComponents` drops the small ones, as upstream's order does
  (repair, small components, fill holes). Hole filling skips a lone triangle (its cap would be the same face reversed); a larger cut-off fin whose
  boundary loop is under the fill perimeter still gets a centroid fan.
- UV unwrap is not a cause: by vertex index the unwrapped mesh shows thousands of components and 140,000 boundary edges because seams
  split vertices, and welded by position it has the topology of the mesh that went in. A count of "components" taken on a GLB measures UV
  charts, not the surface.

## Remaining boundary, classified

`MeshBoundary.Trace` walks the boundary into loops: each loop follows one fan of faces at every vertex, so loops that only touch at a pinch vertex are one walk
that passes it twice (`BoundaryLoop.SimpleCycles` cuts it into its lobes), and a walk that meets an edge with more than two faces ends there as an open chain.
The numbers below come from a throwaway probe outside the repository over saved TRELLIS.2 decoder output (seed 1; bust and mug at 512, mug at 1024; raw
`DualGridMesher` output, then `MeshCleaner.Clean` with its defaults). A loop is a sliver when its vertices span at most 3 voxels, a slit when its
second principal spread is under 0.35 voxel or 12% of the first, a patch when the third is under 0.5 voxel or 10% of the second, else a 3D opening. "By index" is the mesh as a
GLB stores it; "welded" merges vertices at the same position, so the copies made by the non-manifold repair read as one vertex again.

| Raw mesher output | Vertices / triangles | Boundary edges | Non-manifold edges | Sliver (edges) | Slit (edges) | Patch (edges) | 3D opening (edges) |
|---|---|---|---|---|---|---|---|
| Bust 512 | 736,637 / 1,487,440 | 18,889 (5,664 walks: 1,700 closed, 3,964 open) | 30,844 | 18,607 | 44 | 224 | 14 |
| Mug 512 | 1,282,266 / 2,528,776 | 41,231 (1,119 walks: 630 closed, 489 open) | 7,249 | 4,299 | 27,748 | 8,756 | 428 |
| Mug 1024 | 5,596,157 / 11,070,214 | 160,408 (1,294 walks: 346 closed, 948 open) | 43,097 | 4,011 | 105,308 | 44,244 | 6,845 |

| After `Clean` defaults (by index) | Boundary edges | Loops | Sliver (edges) | Slit (edges) | Patch (edges) | 3D opening (edges) | Welded boundary / non-manifold | Time |
|---|---|---|---|---|---|---|---|---|
| Bust 512 | 30,198 | 2,521, all closed | 16,274 | 264 | 7,424 | 6,236 | 9,611 / 19,708 | 1.4 s |
| Mug 512 | 46,678 | 545, all closed | 1,842 | 34,686 | 9,442 | 708 | 38,243 / 7,146 | 2.5 s |
| Mug 1024 | 204,558 | 205, all closed | 480 | 76,276 | 46,812 | 80,990 | 156,336 / 42,378 | 10.6 s |

- **The raw openings are parity defects of the decoded edge flags, not missing voxels.** Re-meshing the same decoder output while counting why each quad was kept
  or dropped: the bust has 743,727 flagged grid edges and 7 of their quads are dropped for a missing corner voxel (14 of 18,889 boundary edges touch one); the mug at
  512 has 1,268,526 and 4,138 (5,820 of 41,231 boundary edges, 14.1%). No non-manifold edge touches a dropped quad. The rest are voxel faces with an odd number of flagged
  edges (a boundary edge) or three or four (a non-manifold one): the network predicts each edge's flag on its own, so the flags are not the crossings of one sign field.
  Nothing is cut by a narrow band (the mesher has none); at 512 no loop lies on the faces of the grid cube, at 1024 the mug has 3 walks (1,351 edges) that do, where it touches the grid limit.
- **Raw bust boundaries are all tiny**: no walk is longer than 64 edges, and 5,639 of 5,664 walks span at most 3 voxels. They read as open chains because
  the non-manifold edges stop the walks. Pinch vertices, where walks meet: 616 on the bust, 101 on the 512 mug and 1,004 on the 1024 mug.
- **The mug's openings are slits and strips**: the body is a stack of flat terraces and the gaps run along the steps. The longest walk at 1024 has 6,816 edges and
  spreads 243 voxels along one axis and 144 along the other, while 105,308 of its raw boundary edges are on slits only a few voxels wide; a perimeter limit cannot tell
  a long thin slit from a real opening.
- **The default repair makes the by-index count worse**: the non-manifold repair cuts every surplus face loose along its edge, so the bust goes from 18,889 to 30,198
  boundary edges and the 1024 mug from 160,408 to 204,558. What it removes (non-manifold edges to 0) it turns into boundary. Of the 30,198, about 10,000 sit on 61 loops
  of 65 to 1,024 edges that the perimeter limit (3e-2 is 15 voxels at 512) leaves open.
- **Measuring**: the welded count is what `MeshStatistics` reports and can never read as watertight while the repair duplicates positions; the by-index count is the topology a
  consumer that does not merge vertices sees. Both are reported.

## Decimation target and texture size, measured (2026-11-13)

Upstream's own defaults (read from `microsoft/TRELLIS.2` on 2026-11-13): `o_voxel.postprocess.to_glb` takes `decimation_target = 1000000` and `texture_size = 2048`; `example.py` calls it with 1,000,000 and
4096 (and `remesh=True`); the Gradio `app.py` offers a decimation slider from 100,000 to 1,000,000 that starts at **500,000** and a texture slider from 1,024 to 4,096 that starts at 2,048. This engine's
`PbrPostProcessOptions` defaults to 1,000,000 and 2,048, the function defaults.

CPU run of `PbrMeshPostProcessor` on the saved textured TRELLIS.2 bust (1,007,560 faces) with every other setting at its default. The attribute volume is a stand-in rebuilt from the saved atlas (the colour
of each vertex's texel, averaged on a 512 grid, 379,483 voxels), not the original volume, so the look is a proxy; the coverage, chart and time columns do not depend on the volume.

| Decimation target | Texture | Faces | Charts | Faces per chart | Atlas covered by UV triangles | Post-process time | GLB size |
|---|---|---|---|---|---|---|---|
| 1,000,000 | 2048 | 999,969 | 44,752 | 22 | 20.0 % | 38 s | 40.1 MB |
| 500,000 | 2048 | 499,840 | 29,968 | 17 | 20.3 % | 33 s | 24.1 MB |
| 200,000 | 2048 | 199,788 | 27,751 | 7 | 20.1 % | 32 s | 14.5 MB |
| 1,000,000 | 4096 | 999,969 | 44,752 | 22 | 22.6 % | 97 s | 51.2 MB |
| 500,000 | 4096 | 499,840 | 29,968 | 17 | 22.8 % | 102 s | 35.1 MB |
| 200,000 | 4096 | 199,788 | 27,751 | 7 | 22.4 % | 100 s | 25.3 MB |

- **About 80 % of the atlas stays uncovered whatever the target.** Coverage is 20 to 23 % at every target, so it is not a face-count effect. The cost is the space between charts: the unwrapper makes
  28,000 to 45,000 small charts (a coarser mesh leaves their count almost flat, 29,968 at 500 k against 27,751 at 200 k, while the faces per chart fall from 17 to 7), each with a gutter (4 texels, 1 for
  charts under 8 texels, halved when the packed scale falls under 0.2 of ideal), and above 1,000 charts the shelf packer also loses space at the end of every shelf. The split between gutters and packing loss
  was not measured; going to 4096 halves the gutters' share yet adds only 2.3 to 2.6 points, which suggests the gutter is not the main cost. The lever may be fewer, larger charts, not the texture size. The
  texture side doubled (4096) takes about 2.6 times as long at 1 M faces and about 3.1 times at 500 k and 200 k, and adds about 11 MB to the GLB.
- **The 1 M-face results show blotches, at both sizes; the cause is open.** In the front-view renders of all six runs, 1 M faces at 2048 and at 4096 shows pale blotches on the forehead and hair that 500 k and 200 k do
  not; 4096 over 2048 adds little at any target. The covered texels per face do not explain it (0.84 at 1 M and 2048, 1.7 at 500 k and 2048, 3.8 at 1 M and 4096: the 4096 run has twice the density of the clean 500 k run
  and still blotches). Two candidates: the 1 M row barely decimates its input (1,007,560 faces in, 999,969 out), so it keeps the raw mesh and its sliver faces, and the volume is a stand-in rebuilt from that
  mesh's own atlas, which may reproduce the atlas's artefacts. The run with the real attribute volume settles it. The look comparison is a single subject.
- **Proposed default for textured output, not applied:** 500,000 faces and a 2048 texture, which is the value upstream's own app starts at, gives a 40 % smaller GLB than the current default and cleaner renders in
  this test; 4096 stays an explicit choice. Changing the default needs the same comparison on a run with the real attribute volume (one GPU run of TRELLIS.2 at 512).
- The saved render sheet of the six runs is `trellis2-atlas-defaults-compare.png` in the shared notes folder of the Hunyuan3D session group (`hunyuan-gpu-block-results`, outside the repository). Upstream's defaults were read from the `main` branch of `microsoft/TRELLIS.2` on 2026-11-13 (no commit pinned).

## Closing slits and gaps (`HoleFiller`)

State of the shape-only finisher, plainly: the default finisher turns non-manifold edges into boundary edges. It splits vertices, so each split leaves boundary edges by index (Pixal3D bust 1024: 891,942
non-manifold edges become 995,170 boundary edges; welded by position the non-manifold count is still 785,313). That bust is the worst case measured; on the mugs the change is small (512: 41,231 to 46,678). Only upstream's
dual-contouring remesh (`remesh_narrow_band_dc`) would remove it, and it is not ported. `HoleFiller` and `CloseOpenings` below close most of those cuts and slits, but by index only: on that bust width 0.1 takes the boundary
from 995,170 to 340,960 while the welded non-manifold count rises from 785,313 to 872,800 and the time from 7.3 to 25 s, so they are off by default and stay off.

An opt-in step after the perimeter fill (`MeshCleanOptions.HoleFillMaxWidth`, default 0 = off, so the default output is unchanged). It traces the boundary with `MeshBoundary`, cuts walks that pass a pinch vertex
into simple cycles, and triangulates each cycle by ear clipping without adding a vertex. Ears are ranked by roundness (a sliver along one side of a slit comes last, so the clipping zips across the slit), with a
small penalty for a long new edge and a large one for an ear that folds against the surrounding surface; an ear whose new edge already exists in the mesh is skipped (it would give that edge a third face). A cycle is
filled only when no new edge is longer than the width limit, tried at 1/16, 1/8, 1/4, 1/2 and the full limit so a long edge allowed early does not strand the rest of the loop; a triangle hole counts its own edges.
That is what separates a slit from an opening: a slit of any length zips shut across its width, a cut base or a mouth needs edges as long as itself and stays open. Orientation is unified first, because the patches take
their winding from the faces around the hole.

Wiring: `ShapeMeshFinisher.Finish(mesh, openingWidthRatio:)` (share of the bounding-box diagonal; `DefaultOpeningWidthRatio` is 0.1), `Trellis2Options.CloseOpenings` and `Pixal3DOptions.CloseOpenings` (off by default),
and `fill-holes` takes a `maxWidth` share of the diagonal besides `maxPerimeter` (0 by default). `HoleFiller.Fill` reports what it filled and what stayed open (too wide, unclosable, walks that end at a non-manifold edge).

Measured on the saved meshes of the section above (CPU, welded counts as `MeshStatistics` takes them; the counts repeat exactly between runs, the times are from a shared, loaded machine and vary up to twofold, for example 2.3 and 5.8 s for the SAM 3D bust):

| Mesh, finisher | Boundary edges, by index | Non-manifold, by index | Welded boundary / non-manifold | Triangles | Time |
|---|---|---|---|---|---|
| Bust 512 raw | 18,889 | 30,844 | 18,889 / 30,844 | 1,487,440 | |
| Bust 512, defaults | 30,198 | 0 | 9,611 / 19,708 | 1,485,720 | 1.6 s |
| Bust 512, width 0.1 | 0 | 0 | 0 / 22,407 | 1,510,876 | 3.2 s |
| Mug 512 raw | 41,231 | 7,249 | 41,231 / 7,249 | 2,528,776 | |
| Mug 512, defaults | 46,678 | 0 | 38,243 / 7,146 | 2,530,286 | 2.6 s |
| Mug 512, width 0.1 | 0 | 0 | 0 / 7,466 | 2,575,874 | 8.3 s |
| Mug 1024 raw | 160,408 | 43,097 | 160,408 / 43,097 | 11,070,214 | |
| Mug 1024, defaults | 204,558 | 0 | 156,336 / 42,378 | 11,067,728 | 21.8 s |
| Mug 1024, width 0.1 | 92,464 (10 loops) | 0 | 66,429 / 42,796 | 11,179,432 | 27.4 s |
| Pixal3D bust 1024 raw (a real run, below) | 124,853 | 891,942 | 124,853 / 891,942 | 6,237,152 | |
| Pixal3D bust 1024, defaults | 995,170 | 0 | 76,651 / 785,313 | 6,241,216 | 7.3 s |
| Pixal3D bust 1024, width 0.1 | 340,960 (5 loops) | 0 | 28,080 / 872,800 | 6,881,564 | 25.0 s |
| SAM 3D bust, saved after `Sam3dObjectsMesh.Finish` | 466 | 0 | 466 / 0 | 676,758 | |
| SAM 3D bust, width 0.1 (area ratio 1e-3 as in the engine) | 0 | 0 | 0 / 0 | 677,202 | 2.3 s (1.1 s without) |

- The by-index count is the topology of the mesh as stored; the welded one is what `MeshStatistics.Measure` returns. Width 0.1 leaves the SAM 3D bust with no boundary and no non-manifold edge by either count, so that mesh is closed
  in the full sense (it has four components). The TRELLIS.2 meshes are closed by index only: the non-manifold repair splits vertices, two sheets that touch along an edge keep sharing positions, and welding
  merges them again (the welded non-manifold column). That count is the geometry, not a leftover of the fill, and only a change of geometry (the upstream remesh) removes it.
- The width limit decides the result, not time: on the mug at 512, ratios 0.05, 0.1 and 0.2 leave 432, 0 and 0 boundary edges; on the bust 0.03 already leaves 0. On the 1024 mug width 0.1 leaves ten loops, 7 of them
  patches and 3 of them 3D openings, the largest of 48,566 and 21,736 edges (a spread of about 290 by 260 by 135 voxels), and one on the grid face: those are real openings in the decoded shape, and
  they stay open. Width 0.2 leaves 74,768 edges.
- Pixal3D bust 1024 is a real run (2026-11-13, RTX 3060, seed 1, `CleanMesh = false`, 42 progress steps (the split per flow was not logged), 266 s from start to the written GLB, whole-card `memory.used` peak 10,471 MiB with about 2 GB
  of it other processes) on a stand-in image, the SAM 3D session's `bust35_rgba.png`, not the image behind the earlier Pixal3D bust figures; the output is a 2.6 M vertex, 6.2 M triangle mesh with 891,942 non-manifold edges. The defaults turn
  those into 995,170 boundary edges by index; width 0.1 closes 654,210 of them and leaves five loops, three of 223,418, 84,914 and 23,828 edges that spread 130 to 145 voxels (3D openings), all away from the grid faces. The raw mesh was written
  in the export frame, after the winding unification of the `CleanMesh = false` path. It was not rendered. The defaults raise its boundary edges eightfold because they split the 891,942 non-manifold edges (welded 76,651), and why this subject has six times the non-manifold edges of the earlier bust on about half the triangles is not explained.
- Winding: counting edges whose two faces run the same way, the defaults leave 3,373 (bust), 2,254 (mug 512), 5,987 (mug 1024) and 121,725 (Pixal3D bust); width 0.1 raises them to 8,197, 6,448, 12,488 and 248,384. The SAM 3D bust has none.
  Boundary and non-manifold counts do not depend on it. Traced on the 512 meshes: the rise is not the patches. Of the 8,197 disagreements on the bust 6,820 are between two original faces, 780 between an original face and a patch and 597 between two patches
  (mug: 6,038, 226 and 184); a patch takes its winding from the faces around its hole, and every loop lies inside one edge-connected component (the repair leaves no pinch vertices), so a patch cannot disagree with a consistently wound
  neighbourhood (tests: capped sphere, slit grid, pinched octahedron, mixed winding through `Clean`, all consistent). The disagreements are the repaired surface itself: 3,202 of the bust's 3,373 are between nearly coplanar faces, so a seam of twisted
  joins that the flood places wherever its fronts meet; adding patches joins more of the surface and moves and lengthens those seams. Local search does not remove them (flipping, one at a time and re-checking, each face with two disagreeing edges: 8,197 to 6,790; an earlier all-at-once version oscillated near 7,400), a
  flatness-first flood (smoothest continuations first) halves the bust (3,373 to 1,031, 8,197 to 4,168) but makes the mug worse (2,254 to 7,558), and joining the faces of a non-manifold edge in smooth pairs lowers the defaults only from 3,373 to 2,923,
  so none was kept. Faces on a seam are back faces from one side, the likely cause of the see-through streak at the mug's rim (that streak's faces were not identified). Removing them needs the repair to produce an orientable surface, which per-edge vertex splitting does not guarantee.
- The slits and terraces of the mug are filled with flat strips of triangles, so the patches are as rough as the surface around them; nothing fairs or relaxes them. Rendered with the engine's thumbnail renderer, the bust's
  dark specks in the hair are gone; on the mug most white streaks along the inner wall are closed, but one at the rim stays see-through (a seam of faces wound against their neighbours, the winding bullet above; the exact faces of that streak were not identified).
- Tried and not kept: joining all faces of a non-manifold edge in pairs (wound consistently, most coplanar) instead of one pair. It lowered the by-index boundary of the defaults only from 30,198 to 26,778 on the bust, raised the welded
  non-manifold count, and gained nothing once the fill ran.

### Feasibility of the narrow-band dual-contouring remesh (not implemented)

Upstream's `to_glb` runs `remesh_narrow_band_dc` only with `remesh=True`; its default is the standard path above. Read from CuMesh `remeshing.py`: a sparse grid is refined from 32 up to the grid resolution, keeping a voxel while the
unsigned distance at its centre (`bvh.unsigned_distance`) is under 0.87 of the cell; the distance is shifted by `eps = band * voxel` (band 1) and the zero set is dual-contoured (`simple_dual_contour`, source not read), with one quad per
crossing edge through the four voxels around it (the offsets and the split rule of `DualGridMesher`, so that code could be reused) and quads with a missing neighbour dropped; `project_back` blends each vertex toward the closest point of the original.
There is no flood fill and no sign: the result is the offset surface at distance `eps`, a closed shell about two voxels thick around the original, and a hole wider than `2 * eps` stays a hole through the shell.

Cost here, measured with `TriangleBvh` on the finished 512 bust (1,485,720 faces, 8 threads, a loaded machine): build 3.6 s, 1.56 million exact distance queries per second within 1.5 voxels of the surface. The band holds one to three
times as many voxels as the surface has (736,637 at 512 for the bust, 5,596,157 for the 1024 mug), so an estimated few million to fifteen million queries (2 to 10 s at the rate above) plus a voxel hash (estimated at 40 bytes a voxel: 100 MB at 512, 700 MB at 1024); the estimates are not measured. It is bounded and testable
(a sphere must come out closed with the volume of the offset ball), so cost is not the reason it is left out. The reasons are what it produces: twice the faces and a shell displaced by a voxel on each side, or, projected back, two coincident
sheets that weld to non-manifold edges again, so the welded count would not be the thing it fixes; and the by-index closure above already reaches no boundary and no non-manifold edge on three of the four meshes. It is the next step only if a consumer needs
position-welded manifold geometry; a single-layer alternative (flood fill from the grid's outside through voxels farther than a gap tolerance from the surface, then contour the solid) is a separate design that changes topology.

## Known limits against cumesh and xatlas

- **Parametrization is planar projection per chart**, not LSCM, ABF or xatlas' iterative chart optimization. Stretch is bounded by the
  cone angle (default 50 degrees, `UvUnwrapOptions`), so texel density on steep parts of a chart is lower than on its flat parts.
  A chart is grown from its largest unassigned face and a face joins only if it keeps the projection valid (no overlap, normal within
  the cone), so a noisy surface costs only the facets that really fold, not the whole chart. The first version split a chart at half the cone
  angle whenever one facet folded, down to single triangles. Charts under `AbsorbBelowFaces` (32) are then dissolved: large charts take their
  faces at a wider cone (80 degrees), and a last round takes small steep leftovers at 87 degrees (a sliver of texels, inpainted from the
  neighbours). Faces that fold back on their neighbours (normals past 87 degrees) still end as single-face charts: on the bust about 5,800 faces,
  0.3% of its area.
  Measured on three TRELLIS.2 meshes of 1M faces (welded), first version -> now: bust at 512, 111,770 -> 11,785 charts, 5.6% -> 22.5% of the
  2048x2048 atlas covered by triangles, 2.4 s -> 4.6 to 6.6 s; bust at 1024, 275,550 -> 52,167 charts, 9.5% -> 20.5%, 3.0 s -> 4.6 to 5.7 s; mug,
  126,147 -> 3,261 charts, 9.3% -> 39.3%, 2.0 s -> 4.1 to 5.3 s (CPU, unwrap only, two runs). Decimating 11M triangles, unwrapping, baking and inpainting took
  46 to 68 s.
- No seam-length minimization and no chart optimization beyond the grow-and-absorb above; charts are only rotated to the best of 24
  bounding-box angles before packing.
- Packing: MaxRects best-short-side-fit only for up to 256 charts; larger counts use a height-sorted shelf packer (roughly 65-75%
  efficient). The gutter around each chart is `PaddingTexels` (4); it halves while the charts do not fit at a useful scale but never below
  `MinPaddingTexels` (1), and a chart whose longest side is under 8 texels keeps the minimum gutter, so thousands of slivers do not each cost
  a full gutter. `UvUnwrapResult.PaddingTexels` reports what was kept. Sorts break ties on the index, so the atlas is the same on every run.
- Decimation is quadric collapse only: no attribute-aware metric, no preservation of sharp features beyond the quadric itself, and
  non-manifold edges are locked (vertices where several manifold fans meet are not detected) rather than repaired. Cleaner hole filling does not fair or relax the patch.
- Inpainting follows Telea's method and OpenCV's structure but is not bit-identical to OpenCV.
- Everything is CPU. A full-size run (1M faces, 2048 texture) has not been measured; see the pull request for what was.

## Tests

`tests/HartsyInference.ThreeD.Tests/MeshPostProcess`, synthetic meshes only, run with
`dotnet test tests/HartsyInference.ThreeD.Tests --filter "FullyQualifiedName~MeshPostProcess"`. They check topology, bounds,
determinism, UV range and overlap (with an independent polygon-clipping check), BVH against brute force, hand-computed trilinear
values, texel-exact bakes, inpainting smoothness and GLB structure (chunks, accessors, views, PNG payloads). They do not compare
against upstream output; no upstream dump was available.
