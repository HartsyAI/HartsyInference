#!/usr/bin/env bash
# The generation cases every gate under tests/ runs — one list, so a digest recorded by one script is comparable to
# a run made by another. Columns are tab-separated:
#
#   id | tags | checkpoint (relative to the models root) | command|positional|arguments
#
# Seeds are NOT here: each gate sets its own (migration-baseline.sh pins 42; regression-ab.sh runs a seed series).
# Tags pick a subset — `baseline` is what migration-baseline.sh records, `core` is the per-PR regression set.
#
#   regression_cases core      # id, checkpoint, spec — for rows tagged core
#   regression_cases           # every row
regression_cases() {
    local want="${1:-}"
    local id tags ckpt spec
    while IFS=$'\t' read -r id tags ckpt spec; do
        [ -z "$id" ] && continue
        if [ -n "$want" ] && [[ ",$tags," != *",$want,"* ]]; then
            continue
        fi
        printf '%s\t%s\t%s\n' "$id" "$ckpt" "$spec"
    done <<'MATRIX'
sd15	baseline,core	bench-cache/e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916/v1-5-pruned-emaonly-fp16.safetensors	image|a red apple on a table|--steps 8 --width 512 --height 512
krea2	baseline,core	Stable-Diffusion/Krea2/Turbo/krea2_turbo_fp8_scaled.safetensors	image|a red apple on a wooden table|-m krea2 --steps 8 --width 1024 --height 1024
zimage	core	Stable-Diffusion/z-image-turbo.safetensors	image|a lighthouse on a rocky coast at dusk, photograph|-m zimage --steps 8 --width 1024 --height 1024
qwenimage-q4k	core	Stable-Diffusion/QwenImage/Qwen_Image-Q4_K_M.gguf	image|a bowl of ramen on a wooden counter, photograph|-m qwen-image --steps 20 --width 1024 --height 1024
llama32-1b	baseline,core	llm/llama32-1b/llama-3.2-1b-instruct-q8_0.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
MATRIX
}
