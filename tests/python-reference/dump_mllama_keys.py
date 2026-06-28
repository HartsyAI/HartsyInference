"""Reconcile helper for Llama-3.2-Vision (mllama). The community GGUFs are converted by Ollama's fork, so the
tensor naming may differ from mainline llama.cpp. Run this on BOTH the text GGUF and the mmproj GGUF to dump the
exact architecture, metadata, and tensor names so the C# MllamaKeyMapper / MllamaVisionEncoder / config can be
reconciled to reality before building against guessed keys (the parity-loop "inspect real keys first" step).

Usage:
    pip install gguf            # already present if you've run the other dump_*_ref.py scripts
    python dump_mllama_keys.py /path/to/llama-3.2-11b-vision.Q4_K_M.gguf /path/to/mmproj-f16.gguf

Paste the output back. Only tensor names + small metadata are read (the header), not the weights, so it is fast
and uses no real memory.
"""
import sys
from gguf import GGUFReader

def dump(path, label):
    r = GGUFReader(path)
    print(f"\n================ {label}: {path} ================")
    # Architecture + a curated set of interesting metadata keys.
    print("--- metadata (architecture + dims + counts) ---")
    for f in r.fields.values():
        name = f.name
        if any(k in name for k in ("architecture", "block_count", "head_count", "embedding_length",
                                   "feed_forward", "context_length", "cross_attn", "clip", "vision",
                                   "image", "patch", "projector", "tile", "layer_norm", "rope", "attention")):
            try:
                val = f.contents()
            except Exception:
                val = "<unreadable>"
            print(f"  {name} = {val}")
    # All tensor names + shapes (this is what we reconcile the mapper / encoder against).
    print(f"--- {len(r.tensors)} tensors (name : shape : dtype) ---")
    for t in r.tensors:
        print(f"  {t.name} : {list(t.shape)} : {t.tensor_type.name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    dump(sys.argv[1], "TEXT MODEL")
    if len(sys.argv) > 2:
        dump(sys.argv[2], "MMPROJ (vision)")
