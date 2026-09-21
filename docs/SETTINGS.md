# Settings

Everything the engine can be configured with is a **setting**: a dotted id (`paths.modelsRoot`,
`vram.keepModels`, `numerics.sageAttn`), declared once in `src/HartsyInference.Core/Configuration/EngineKnobs.*.cs`
with its type, default, scope and domain. The id is the only name a setting has — it is what the file, the CLI
and the API all use.

**The engine reads no environment variables.** Exporting `HARTSY_ANYTHING` does nothing. The only environment
variables anything here reads are third-party conventions we honour but do not own (`HF_TOKEN`, `HF_ENDPOINT`,
`ESPEAK_DATA_DIR`, `NO_COLOR`, `COLORFGBG`), and `tests/HartsyInference.Core.Tests/env-read-allowlist.txt` fails
the build if that list grows.

## Where settings live

One file:

```
~/.config/hartsyinference/settings.json
```

`hartsy settings path` prints it. A host that keeps its settings elsewhere sets `KnobFile.ExplicitPath` before
the first setting is read; there is no search path beyond that, so which file applies never depends on the
directory a process was started from.

```json
{
  "profile": "reference",
  "settings": {
    "paths.modelsRoot": "/mnt/model-storage/Models",
    "vram.keepModels": false
  }
}
```

`profile` is optional and applied first; `settings` is applied on top. A malformed file **throws** rather than
being skipped — a settings file that is silently ignored is how someone benchmarks a configuration they never
actually applied.

## Changing a setting

**CLI** — `set` persists to the file:

```bash
hartsy settings list                                    # every setting (--all adds diagnostics)
hartsy settings get paths.modelsRoot                    # effective value + where it came from
hartsy settings set paths.modelsRoot /mnt/models        # persists
hartsy settings path                                    # which file
```

For one run only, without touching the file: `--set id=value` and `--profile <name>`.

**API**:

```
GET  /settings                     server options + every engine setting with its value and source
GET  /settings/engine/{id}         one setting
PUT  /settings/engine/{id}         {"value": "..."} — persists, returns the new state
```

The value is a string in both cases and is parsed with the file's rules, so `"true"`, `"1"` and `"256"` mean
the same thing everywhere. An unknown id, a wrong type or an out-of-range value is rejected at the point of
setting, not at the next startup.

## Which value wins

```
per-request profile  →  host (KnobStore.Set)  →  settings file  →  declared default
```

`settings get` and the API both report the **source**, which matters because a value can legitimately differ
from what the file says:

**Inside SwarmUI, SwarmUI owns the models folder.** The backend extension calls
`KnobStore.Set(EngineKnobs.ModelsRoot, …)` from SwarmUI's own `ModelRoot` when it initialises, so that is what
governs there, and it persists in SwarmUI's `Settings.fds` like the rest of its configuration. The engine's
settings file governs the CLI, the API and any headless host. Both are deliberate: SwarmUI lists models from
its own tree, so loading them from a different one would be worse than useless. `settings get` reports `host`
for such a value.

## A long-running process reads the file once

The file is loaded on the first setting read and then cached for the process. A CLI invocation is a fresh
process, so it always sees the current file; a **running server does not** — write to it through
`PUT /settings/engine/{id}` and it applies immediately, but an edit made elsewhere (the CLI, an editor) is
not picked up until that server restarts.

This is deliberate rather than an omission. Re-reading the file would have to discard the in-process
overrides to apply cleanly, and those include the ones a host set: SwarmUI pushes `paths.modelsRoot` at
backend init, so a reload triggered by an unrelated edit would drop the models folder out from under a
running generation. A server changes its own settings through its own API.

## Scope: when a setting takes effect

| scope | meaning |
|---|---|
| `Runtime` | read per generation; a per-request profile can override it |
| `Construction` | bound when the engine or a backend is built — **restart to apply** |

`settings set` and the API both say so when the setting is `Construction`. A per-request profile naming a
`Construction` setting is rejected (400) rather than accepted and silently ignored.

One caveat worth knowing: `Runtime` is approximate. Some consumers bind a value into a `static readonly` field
at type-initialisation, which freezes it for the process even though the scope says otherwise.
`KnobScopeIsEnforcedTests` covers the declared cases; the residual gap is real and documented rather than
claimed away.

## Adding a setting

Declare it in the `EngineKnobs.*.cs` partial for its domain and read it through `Knob.Value` at the point of
use. Do not cache a `Runtime` knob in a `static readonly` field — that is the freeze above, and
`KnobScopeIsEnforcedTests` will say so.
