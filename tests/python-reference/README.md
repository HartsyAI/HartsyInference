# Reference tools

Offline Python oracles cover multiple model families and operations. They are validation tooling, not runtime dependencies. Find the matching script/test pair with rg; each script defines its dependencies, checkpoint paths, dump layout and options.

Create an isolated virtual environment for the relevant reference implementation. Do not install one universal dependency set for this heterogeneous corpus. Unix environments use bin/python; Windows environments use Scripts/python.exe.

Preserve checkpoint/reference revisions, input hashes, shape/dtype/layout metadata and resolved settings with dumps. Raw tensors are not universally F32; read the writer and consuming test. Share actual noise/embeddings between runtimes rather than only matching seeds.

Reference artifacts can be large or machine-local. A missing fixture or skipped comparison does not establish parity. Record verified results in [PARITY_VERIFICATION](../../docs/Checklists/PARITY_VERIFICATION.md) and the relevant modality status; keep detailed diagnostics only when they preserve a reusable finding.
