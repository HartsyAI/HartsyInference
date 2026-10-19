#!/usr/bin/env bash
# The generation cases every gate under tests/ runs — one list, so a digest recorded by one script is comparable to
# a run made by another. Columns are tab-separated:
#
#   id | tags | checkpoint (relative to the models root) | command|positional|arguments
#
# Seeds are NOT here: each gate sets its own (migration-baseline.sh pins 42; regression-ab.sh runs a seed series).
# Tags pick a subset — `baseline` is what migration-baseline.sh records, `core` is the per-PR regression set, `rope`
# runs every family on the shared Core RoPE tables (Core/Rope) with a checkpoint on this host, and `rope-large` runs them
# at 2K, 16:9 and odd-token-grid sizes, where the position range and the table sizes grow.
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
wan5b	video,quality,rope	Stable-Diffusion/Wan/wan2.2_ti2v_5B_fp16.safetensors	video|a red fox trotting through a sunlit snowy forest, cinematic, shallow depth of field|-m wan --width 832 --height 480 --frames 49 --steps 20 --cfg 5
ideogram4	vulkanbar,quality,rope	Stable-Diffusion/Ideogram4/ideogram4_fp8_scaled.safetensors	image|{"high_level_description": "A cinematic wide-angle photorealistic photograph of an astronaut in a white NASA spacesuit riding a chestnut horse across the rust-red Martian desert at golden hour, dramatic warm sunlight, ultra-detailed, sharp focus.", "compositional_deconstruction": {"background": "A vast Martian desert of rolling rust-orange sand dunes under a pale dusty-pink sky, distant eroded mesas and craters on the horizon, low golden-hour sun casting long shadows and warm rim light across the terrain, faint atmospheric haze.", "elements": [{"type": "subject", "desc": "An astronaut in a detailed white pressurized spacesuit with a reflective gold-tinted visor helmet, seated upright riding the horse, one gloved hand holding the reins."}, {"type": "animal", "desc": "A powerful chestnut-brown horse mid-stride galloping across the sand, mane and tail flowing, hooves kicking up red dust."}, {"type": "lighting", "desc": "Warm golden-hour side lighting, long dramatic shadows, high dynamic range, crisp photographic detail."}]}}|-m ideogram4 --steps 20 --width 1024 --height 1024
chroma-hd	vulkanbar,rope	Stable-Diffusion/Chroma/Chroma1-HD-fp8mixed-final.safetensors	image|a red apple on a wooden table|-m chroma --steps 30 --width 1024 --height 1024
ernie-turbo	vulkanbar,rope	Stable-Diffusion/Ernie/ernie-image-turbo-fp8.safetensors	image|a red apple on a wooden table|-m ernie-image --steps 8 --width 1024 --height 1024
hunyuanimage-comfy	hunyuan,rope	Stable-Diffusion/hunyuanimage21-fp8-1_0-hunyuan-image-21-base-fp8.safetensors	image|a red fox sitting in a snowy forest at sunrise, detailed fur, soft golden light|-m hunyuan-image --steps 20 --width 1024 --height 1024 --cfg 4
hunyuanimage-q4	hunyuan,rope	Stable-Diffusion/HunyuanImage/hunyuanimage2.1-q4_0.gguf	image|a red fox sitting in a snowy forest at sunrise, detailed fur, soft golden light|-m hunyuan-image --steps 20 --width 1024 --height 1024 --cfg 4
anima	anima,rope	Stable-Diffusion/Anima/anima-preview3-base.safetensors	image|a bowl of ripe cherries on a wooden table, soft morning light|-m anima --steps 20 --width 512 --height 512 --cfg 4
hidream	rope	Stable-Diffusion/HiDream/hidream_i1_dev_fp8.safetensors	image|a red apple on a wooden table|-m hidream --steps 20 --width 1024 --height 1024
kandinsky5	rope	Stable-Diffusion/kandinsky-5-lite-1_0-kandinsky5-imglite-base-bf16.safetensors	image|a red apple on a wooden table|-m kandinsky5 --steps 20 --width 1024 --height 1024
moss-image	rope	Stable-Diffusion/moss-image-base-1_0-moss-image-base-bf16.safetensors	image|a red apple on a wooden table|-m moss-image --steps 20 --width 512 --height 512
omnigen2	rope	Stable-Diffusion/omnigen-2-1_0-omnigen-2-base-fp16.safetensors	image|a red apple on a wooden table|-m omnigen2 --steps 20 --width 1024 --height 1024
stableaudio	rope	audio/music/StableAudio/stableaudio-open-small_fp32.safetensors	music|lofi hip hop beat with mellow piano and soft drums|-m stableaudio --duration 10 --steps 8
lumina2	rope	Stable-Diffusion/Lumina2/transformer/diffusion_pytorch_model-00001-of-00002.safetensors	image|a red apple on a wooden table|-m lumina2 --steps 20 --width 1024 --height 1024
zimage-2048	rope-large	Stable-Diffusion/z-image-turbo.safetensors	image|a bustling night market in a narrow rainy Tokyo alley, glowing red paper lanterns and neon signs in Japanese, a street vendor grilling skewers under a striped awning, a woman in a yellow raincoat holding a clear umbrella, reflections on wet cobblestones, steam rising from food stalls, two cats asleep on wooden crates, highly detailed, cinematic lighting, 35mm photograph|-m zimage --steps 8 --width 2048 --height 2048
zimage-wide	rope-large	Stable-Diffusion/z-image-turbo.safetensors	image|a panoramic view of a medieval coastal town at golden hour, terracotta rooftops cascading down a cliff to a harbor full of wooden fishing boats, a stone lighthouse at the end of the breakwater, seagulls over the water, laundry hanging between buildings, distant snow-capped mountains, highly detailed wide-angle landscape photograph|-m zimage --steps 8 --width 1920 --height 1088
krea2-2048	rope-large	Stable-Diffusion/Krea2/Turbo/krea2_turbo_fp8_scaled.safetensors	image|an intricate Victorian greenhouse full of exotic plants, a spiral wrought-iron staircase, a stone fountain with orange koi fish in the center, sunlight streaming through hundreds of glass panes casting geometric shadows, a gardener in blue overalls watering orchids, a tabby cat on a wooden bench, ultra detailed, photorealistic|-m krea2 --steps 8 --width 2048 --height 2048
lens-2048	rope-large	Stable-Diffusion/Lens/lens_turbo_mxfp8.safetensors	image|a cozy bookstore window display on a snowy evening, a hand-painted wooden sign that reads 'OPEN LATE', tall stacks of colorful books, a ginger cat sleeping on a velvet cushion, warm string lights, frost on the glass, a red bicycle parked outside in the snow, detailed photograph|-m lens --steps 4 --width 2048 --height 2048
lumina2-2048	rope-large	Stable-Diffusion/Lumina2/transformer/diffusion_pytorch_model-00001-of-00002.safetensors	image|a portrait of an elderly fisherman with a weathered face and a white beard, wearing a knitted navy sweater and a yellow sou'wester hat, holding a coiled rope, a misty harbor with moored boats behind him, dramatic side lighting, sharp focus on the eyes, 85mm photograph|-m lumina2 --steps 20 --width 2048 --height 2048
anima-1024	rope-large	Stable-Diffusion/Anima/anima-preview3-base.safetensors	image|an anime illustration of a girl with long silver hair in a flowing blue dress standing on a cliff at sunset, cherry blossom petals swirling in the wind, a castle with many spires in the distance, a white cat at her feet, detailed background, vibrant colors|-m anima --steps 20 --width 1024 --height 1024 --cfg 4
qwenimage-1328	rope-large	Stable-Diffusion/QwenImage/Qwen_Image-Q4_K_M.gguf	image|a cozy cafe storefront on a rainy evening, a chalkboard menu that reads 'Fresh Ramen Today', a pink neon sign saying 'NOODLE BAR', a cyclist in a red raincoat passing by, people visible at tables through the window, reflections in puddles, warm interior light, highly detailed photograph|-m qwen-image --steps 20 --width 1328 --height 1328
qwenimage-wide	rope-large	Stable-Diffusion/QwenImage/Qwen_Image-Q4_K_M.gguf	image|a panoramic view of a medieval coastal town at golden hour, terracotta rooftops cascading down a cliff to a harbor full of wooden fishing boats, a stone lighthouse at the end of the breakwater, seagulls over the water, laundry hanging between buildings, distant snow-capped mountains, highly detailed wide-angle landscape photograph|-m qwen-image --steps 20 --width 1664 --height 928
hunyuanimage-q4-1536	rope-large	Stable-Diffusion/HunyuanImage/hunyuanimage2.1-q4_0.gguf	image|a red fox and a snowy owl in a moonlit winter forest, frost-covered pine branches, a frozen stream with ice crystals, green northern lights in the sky, intricate fur and feather detail, cinematic wildlife photograph|-m hunyuan-image --steps 20 --width 1536 --height 1536 --cfg 4
omnigen2-1536	rope-large	Stable-Diffusion/omnigen-2-1_0-omnigen-2-base-fp16.safetensors	image|an intricate Victorian greenhouse full of exotic plants, a spiral wrought-iron staircase, a stone fountain with orange koi fish in the center, sunlight streaming through hundreds of glass panes casting geometric shadows, a gardener in blue overalls watering orchids, a tabby cat on a wooden bench, ultra detailed, photorealistic|-m omnigen2 --steps 20 --width 1536 --height 1536
flux1-dev-1536	rope-large	Stable-Diffusion/BFL/Flux1/flux1-dev-fp8.safetensors	image|a bustling night market in a narrow rainy Tokyo alley, glowing red paper lanterns and neon signs in Japanese, a street vendor grilling skewers under a striped awning, a woman in a yellow raincoat holding a clear umbrella, reflections on wet cobblestones, steam rising from food stalls, two cats asleep on wooden crates, highly detailed, cinematic lighting, 35mm photograph|-m flux1 --steps 20 --width 1536 --height 1536
MATRIX
}
