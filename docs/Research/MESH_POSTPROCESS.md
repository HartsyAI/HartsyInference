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
| `remove_duplicate_faces`, `repair_non_manifold_edges`, `remove_small_connected_components`, `fill_holes`, `unify_face_orientations` | `MeshCleaner` | Non-manifold edges keep the consistently wound, largest pair of faces. Components are vertex-connected and judged by area ratio. Holes fill as a triangle or a centroid fan. Orientation floods across manifold edges, then flips closed components with negative signed volume (extra). |
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
- Output is geometry-only after cleaning and decimation; normals are recomputed (area-weighted) before unwrapping and carried
  across seams.

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
