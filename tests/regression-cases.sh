#!/usr/bin/env bash
# The generation cases every gate under tests/ runs — one list, so a digest recorded by one script is comparable to
# a run made by another. Columns are tab-separated:
#
#   id | tags | checkpoint (relative to the models root) | command|positional|arguments
#
# Seeds are NOT here: each gate sets its own (migration-baseline.sh pins 42; regression-ab.sh runs a seed series).
# Tags pick a subset — `baseline` is what migration-baseline.sh records, `core` is the per-PR regression set, `rope`
# runs every family on the shared Core RoPE tables (Core/Rope) with a checkpoint on this host.
#
#   regression_cases core         # id, checkpoint, spec — for rows tagged core
#   regression_cases core vulkan  # rows tagged core AND vulkan (a backend tag names the cases that run on it)
#   regression_cases              # every row
regression_cases() {
    local id tags ckpt spec want
    while IFS=$'\t' read -r id tags ckpt spec; do
        [ -z "$id" ] && continue
        local keep=1
        for want in "$@"; do
            [ -z "$want" ] && continue
            if [[ ",$tags," != *",$want,"* ]]; then keep=0; break; fi
        done
        [ "$keep" = 1 ] && printf '%s\t%s\t%s\n' "$id" "$ckpt" "$spec"
    done <<'MATRIX'
sd15	baseline,core,vulkan	bench-cache/e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916/v1-5-pruned-emaonly-fp16.safetensors	image|a red apple on a table|--steps 8 --width 512 --height 512
krea2	baseline,core,vulkan,vulkanbar,rope	Stable-Diffusion/Krea2/Turbo/krea2_turbo_fp8_scaled.safetensors	image|a red apple on a wooden table|-m krea2 --steps 8 --width 1024 --height 1024
zimage	core,vulkanbar,rope	Stable-Diffusion/z-image-turbo.safetensors	image|a lighthouse on a rocky coast at dusk, photograph|-m zimage --steps 8 --width 1024 --height 1024
qwenimage-q4k	core,vulkanbar,rope	Stable-Diffusion/QwenImage/Qwen_Image-Q4_K_M.gguf	image|a bowl of ramen on a wooden counter, photograph|-m qwen-image --steps 20 --width 1024 --height 1024
llama32-1b	baseline,core	llm/llama32-1b/llama-3.2-1b-instruct-q8_0.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
granitemoe-3b	moe	llm/moe-parity/granite-3b-a800m/granite-3.0-3b-a800m-instruct-Q4_K_M.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
qwen3moe-30b-a3b	moe	llm/moe-parity/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0 --no-thinking
llama32-1b-iq4xs	quant	llm/llama32-1b/Llama-3.2-1B-Instruct-IQ4_XS.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
llama32-1b-q3km	quant	llm/llama32-1b/Llama-3.2-1B-Instruct-Q3_K_M.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
llama32-1b-q2k	quant	llm/llama32-1b/Llama-3.2-1B-Instruct-Q2_K.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
llama32-1b-iq3m	quant	llm/llama32-1b/Llama-3.2-1B-Instruct-IQ3_M.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
qwen25-1.5b-iq3xs	quant	llm/qwen25-1.5b/Qwen2.5-1.5B-Instruct-IQ3_XS.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
qwen25-1.5b-iq2m	quant	llm/qwen25-1.5b/Qwen2.5-1.5B-Instruct-IQ2_M.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0
lens-mxfp8	mxfp8,vulkanbar,rope	Stable-Diffusion/Lens/lens_turbo_mxfp8.safetensors	image|a red bicycle leaning on a brick wall, photograph|-m lens --steps 4 --width 1024 --height 1024
sdxl	vulkanbar	Stable-Diffusion/SDXL/sd_xl_base_1.0.safetensors	image|a red apple on a wooden table|--steps 20 --width 1024 --height 1024
sdxl-wide	quality	Stable-Diffusion/SDXL/sd_xl_base_1.0.safetensors	image|a lighthouse on a rocky coast at dusk, dramatic sky, photograph|--steps 20 --width 1280 --height 720
flux1-dev	vulkanbar,rope	Stable-Diffusion/BFL/Flux1/flux1-dev-fp8.safetensors	image|a red apple on a wooden table|-m flux1 --steps 20 --width 1024 --height 1024
flux2-dev-q4ks	vulkanbar,rope	Stable-Diffusion/Flux2/flux2-dev-Q4_K_S.gguf	image|a red apple on a wooden table|-m flux2 --steps 20 --width 1024 --height 1024
boogu-turbo	vulkanbar,rope	Stable-Diffusion/Boogu/boogu_image_turbo_fp8_scaled.safetensors	image|a red apple on a wooden table|-m boogu --steps 4 --width 1024 --height 1024
qwenimage21	vulkanbar,rope	diffusion_models/qwen_image_2.1_bf16.safetensors	image|a red apple on a wooden table|-m qwen-image-2.1 --steps 25 --width 1024 --height 1024
wan5b	video,quality	Stable-Diffusion/Wan/wan2.2_ti2v_5B_fp16.safetensors	video|a red fox trotting through a sunlit snowy forest, cinematic, shallow depth of field|-m wan --width 832 --height 480 --frames 49 --steps 20 --cfg 5
ideogram4	vulkanbar,quality	Stable-Diffusion/Ideogram4/ideogram4_fp8_scaled.safetensors	image|{"high_level_description": "A cinematic wide-angle photorealistic photograph of an astronaut in a white NASA spacesuit riding a chestnut horse across the rust-red Martian desert at golden hour, dramatic warm sunlight, ultra-detailed, sharp focus.", "compositional_deconstruction": {"background": "A vast Martian desert of rolling rust-orange sand dunes under a pale dusty-pink sky, distant eroded mesas and craters on the horizon, low golden-hour sun casting long shadows and warm rim light across the terrain, faint atmospheric haze.", "elements": [{"type": "subject", "desc": "An astronaut in a detailed white pressurized spacesuit with a reflective gold-tinted visor helmet, seated upright riding the horse, one gloved hand holding the reins."}, {"type": "animal", "desc": "A powerful chestnut-brown horse mid-stride galloping across the sand, mane and tail flowing, hooves kicking up red dust."}, {"type": "lighting", "desc": "Warm golden-hour side lighting, long dramatic shadows, high dynamic range, crisp photographic detail."}]}}|-m ideogram4 --steps 20 --width 1024 --height 1024
chroma-hd	vulkanbar,rope	Stable-Diffusion/Chroma/Chroma1-HD-fp8mixed-final.safetensors	image|a red apple on a wooden table|-m chroma --steps 30 --width 1024 --height 1024
ernie-turbo	vulkanbar	Stable-Diffusion/Ernie/ernie-image-turbo-fp8.safetensors	image|a red apple on a wooden table|-m ernie-image --steps 8 --width 1024 --height 1024
hunyuanimage-comfy	hunyuan,rope	Stable-Diffusion/hunyuanimage21-fp8-1_0-hunyuan-image-21-base-fp8.safetensors	image|a red fox sitting in a snowy forest at sunrise, detailed fur, soft golden light|-m hunyuan-image --steps 20 --width 1024 --height 1024 --cfg 4
hunyuanimage-q4	hunyuan,rope	Stable-Diffusion/HunyuanImage/hunyuanimage2.1-q4_0.gguf	image|a red fox sitting in a snowy forest at sunrise, detailed fur, soft golden light|-m hunyuan-image --steps 20 --width 1024 --height 1024 --cfg 4
anima	anima,rope	Stable-Diffusion/Anima/anima-preview3-base.safetensors	image|a bowl of ripe cherries on a wooden table, soft morning light|-m anima --steps 20 --width 512 --height 512 --cfg 4
hidream	rope	Stable-Diffusion/HiDream/hidream_i1_dev_fp8.safetensors	image|a red apple on a wooden table|-m hidream --steps 20 --width 1024 --height 1024
kandinsky5	rope	Stable-Diffusion/kandinsky-5-lite-1_0-kandinsky5-imglite-base-bf16.safetensors	image|a red apple on a wooden table|-m kandinsky5 --steps 20 --width 1024 --height 1024
moss-image	rope	Stable-Diffusion/moss-image-base-1_0-moss-image-base-bf16.safetensors	image|a red apple on a wooden table|-m moss-image --steps 20 --width 512 --height 512
omnigen2	rope	Stable-Diffusion/omnigen-2-1_0-omnigen-2-base-fp16.safetensors	image|a red apple on a wooden table|-m omnigen2 --steps 20 --width 1024 --height 1024
stableaudio	rope	audio/music/StableAudio/stableaudio-open-small_fp32.safetensors	music|lofi hip hop beat with mellow piano and soft drums|-m stableaudio --duration 10 --steps 8
lumina2	rope	Stable-Diffusion/Lumina2/transformer/diffusion_pytorch_model-00001-of-00002.safetensors	image|a red apple on a wooden table|-m lumina2 --steps 20 --width 1024 --height 1024
MATRIX
}
