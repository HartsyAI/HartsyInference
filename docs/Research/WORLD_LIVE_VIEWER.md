# World live viewer: design, and how to add a control scheme

`hartsy world --view` opens a window that plays a world model's frames as the session generates them and steers the run from
the keyboard. Hold a key to move; release it to stop. The run still writes its PNGs and MP4 exactly as before. Esc or Q, or
closing the window, ends the run the way `q` does, so what was generated is saved.

**What the model moves is the camera, not the character.** On SANA-WM the keys move the camera through the scene. `w` dollies
toward whatever is ahead, which smears a close subject within a few dozen frames, and `s` pulls back and keeps the subject
framed. For a third-person view, start from a seed still framed from behind the character, looking down the street. Do not
expect the character to respond to the keys.

## Layers

| Layer | Project | Owns |
|---|---|---|
| Control schemes | `HartsyInference.Engine` (`Worlds/`) | `IWorldControlScheme`, the built-in schemes for SANA-WM, Oasis and DIAMOND, and `WorldCapabilities.LiveControls` |
| Core | `HartsyInference.WorldViewer` | `WorldSteering` (held keys to session actions), `WorldFrameStream` (per-step feeding), `WorldPlayback` and `WorldPacer` (bounded buffer and frame clock), `WorldPlaybackText` (status text) |
| Window | `HartsyInference.WorldViewer.Avalonia` | `WorldViewerHost` (Avalonia on its own UI thread) and `WorldViewerWindow` (frames, keys, status block) |

The core depends only on `IWorldSession`, the frame rate and the frame size. It has no window toolkit, so a web or remote front
end can reuse it; an HTTP client for `/v1/native/world/sessions/{id}/stream` is not built yet. Only the window project references
Avalonia, pinned to 11.3.22, the version in the local NuGet cache.

The CLI keeps its frame path: `WorldAsync` writes each frame to the spool (PNGs, then the MP4 on quit) and, with `--view`, also
hands it to the playback buffer. Without `--view` nothing changes.

## Cadence: how keys become actions

A control scheme declares how its session takes actions:

- **`OnChange`** (SANA-WM streaming). The held key set is sent when it changes, and the session applies it to every later
  chunk. Holding `w` sends one action; auto-repeat changes nothing.
- **`PerStep`** (Oasis, DIAMOND). A session that ends whenever its action queue drains needs one action per step. The stream sends
  the current held action before each step and asks again. Oasis re-seeds each step from its last frame, so control lags by one
  generated frame; DIAMOND renders one frame per action.

Only keys the scheme declares are tracked. The window reports key names as they are pressed (`W`, `Space`, `LeftShift`, `Up`).

## Pacing and lag

Frames are played at the plan's frame rate. Playback starts after one second of frames (or the whole buffer if smaller), and a
rebuffer after an underrun resumes at a quarter second. The buffer holds two seconds of video. When it is full the stream waits,
so generation is held back rather than letting control lag grow. The window shows the buffered seconds, the frames shown and
generated, and the delivered frame rate against the play rate. Once the buffer is full the stream waits on the viewer, so the
delivered rate reads at the play rate; it drops below that only when generation is the bottleneck, and then the buffer drains
and the window says it is waiting for generation.

A key press reaches the picture after the buffered video plays out and the next chunk generates: at most the two seconds of
buffer plus one chunk's generation time. On SANA-WM streaming a chunk is 24 frames, 1.5 seconds of video at 16 fps. On the RTX 4090
with a seed still framed from behind the character, the runs delivered about 10 frames per second, so a chunk took about 2.4 seconds and playback stalled
between chunks. The window says so ("Waiting for generation") instead of hiding it. The control lag on that run is therefore
closer to 3 to 5 seconds than the 2 to 3 seconds the design expected.

## Limits, stated plainly

- Oasis batches each step: a key change reaches the picture a frame later, and the model sees only the last frame, not its
  history.
- DIAMOND's picture moves at the model's pace, not the frame rate.
- `--view` and `--interactive` cannot be combined. `--actions` is refused with `--view` for a per-step model; steer it with the keys.
- Avalonia on macOS wants the process's main thread, and the window runs on its own thread. Only Linux (X11 or Wayland) has been
  run here; Windows and macOS are unverified.
- A batch model with no live control (SANA-WM stage 1) plays its clip when the run completes, and the window says so.

## Adding a control scheme for a new world model

A new model adds a scheme and nothing else; the viewer reads the scheme from the catalog and never names a model.

1. Add a class in `src/HartsyInference.Engine/Worlds/` that implements `IWorldControlScheme`: a `Name`, the `Cadence`, the
   `Controls` list (key name and label for the legend), a one-line `Note`, and `ActionFor(heldKeys)`, which returns the action
   string the session takes. Pick `OnChange` if the session holds the last action, `PerStep` if it consumes one action per step.
   Expose it as a shared `Instance` the way `SanaWmControlScheme` does.
2. Declare it on the model's `WorldCapabilities` in `ModelCatalog`: `LiveControls = YourScheme.Instance`. Models without live
   control leave it null.
3. Test the mapping in `tests/HartsyInference.Engine.Tests/Worlds/WorldControlSchemeTests.cs`. Check every action string
   against the model's own action parser (for SANA-WM, `SanaWmAction.ParseKeys`), not only against a string you wrote.

Esc and Q are reserved for stopping a run; a scheme must not declare them. Key names compare case-insensitively.

## Running it

```bash
hartsy world -m sana-wm-streaming --model-path <dit.safetensors> --image <seed.png> \
    --prompt "<prompt>" --intrinsics 800,800,640,352 --actions "s-24" --unbounded --view -o <outdir>
```

The window needs a display (`DISPLAY` or `WAYLAND_DISPLAY`); without one, `--view` fails before the model loads. The window takes
keys only while it has focus, so click it after it opens. Esc or Q, or closing the window, ends the run.

## Tests

- `tests/HartsyInference.WorldViewer.Tests`: steering, pacer, playback buffer, frame stream with a scripted session (chunk
  bursts, a failing stream, cancellation, per-step re-enumeration), and the status text. No GPU, no window.
- `tests/HartsyInference.WorldViewer.Avalonia.Tests`: headless Avalonia. Key down and up through the window reach the steering,
  Esc requests the stop once, and an undeclared key changes nothing.
- `tests/HartsyInference.Engine.Tests/Worlds/WorldControlSchemeTests.cs`: key mapping for the three schemes, and the catalog's
  declarations.

## Evidence

Two live runs on the RTX 4090 (`CUDA_VISIBLE_DEVICES=0`, the 4090 under the default device order), both under the GPU lock,
both `sana-wm-streaming --unbounded --view` with the seed still framed from behind the character and the opening script `s-24`:

- Run 1: keys `w` (3 s), `a` (2 s) and `l` (2 s) sent with real X key events. The window showed the live frames, `Held` changed
  with each key, Esc stopped the run, and the CLI wrote 560 PNGs and an MP4 of 35.0 s at 16 fps, 1280x704. The process exited 0.
- Run 2, the same command: keys `w`, `a` and `l` again, with screenshots taken while each key was held. The HUD read `Held: W`
  and `Held: L` at those moments. The stream delivered about 10 frames per second against 16 to play, so the buffer ran down to
  a tenth of a second and the window showed "Waiting for generation". The run wrote 223 PNGs and a 13.9 s MP4, and Esc stopped it.

Those figures say the plumbing works: the key state reaches the window, the actions reach the session, and the playback and
spool agree. They do not say the 4090 keeps up with real time on this seed: it delivered about 10 frames per second, which is
below the 16 the stream plays at.
