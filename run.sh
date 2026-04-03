#!/bin/bash

#============================命令============================================
PY_FILE=./generate.py
TASK=${TASK:-t2v-1.3B}
CKPT_DIR=${CKPT_DIR:-./Wan2.1-T2V-1.3B}

GPUS_BASE=${GPUS_BASE:-2}
GPUS_FAST=${GPUS_FAST:-8}

SIZE_MULTI=${SIZE_MULTI:-832*480}
SIZE_SINGLE=${SIZE_SINGLE:-480*832}

FRAME_NUM=${FRAME_NUM:-162}
SAMPLE_STEPS=${SAMPLE_STEPS:-20}
SAMPLE_SHIFT=${SAMPLE_SHIFT:-8}
GUIDE_SCALE=${GUIDE_SCALE:-6}

RING_SIZE_FAST=${RING_SIZE_FAST:-8}
ULYSSES_SIZE_FAST=${ULYSSES_SIZE_FAST:-1}
RING_SIZE_MIX=${RING_SIZE_MIX:-4}
ULYSSES_SIZE_MIX=${ULYSSES_SIZE_MIX:-2}

OUT_DIR=${OUT_DIR:-./out_video}
SEGMENT_FRAME_NUM=${SEGMENT_FRAME_NUM:-81}
SEGMENT_COUNT=${SEGMENT_COUNT:-12}
SEGMENT_DIR=${SEGMENT_DIR:-$OUT_DIR/segments}
SEGMENT_PREFIX=${SEGMENT_PREFIX:-seg}
LONG_OUTPUT_FILE=${LONG_OUTPUT_FILE:-$OUT_DIR/long_video_$(date +%Y%m%d_%H%M%S).mp4}
PROMPT_FILE=${PROMPT_FILE:-./prompts.txt}

print_key_params() {
    local mode="$1"
    local nproc="$2"
    local size="$3"
    local frame_num="$4"
    local ring_size="$5"
    local ulysses_size="$6"

    echo "mode=$mode"
    echo "task=$TASK"
    echo "ckpt_dir=$CKPT_DIR"
    echo "python_file=$PY_FILE"
    echo "nproc=$nproc"
    echo "size=$size"
    echo "frame_num=$frame_num"
    echo "sample_steps=$SAMPLE_STEPS"
    echo "sample_shift=$SAMPLE_SHIFT"
    echo "guide_scale=$GUIDE_SCALE"
    if [ -n "$ring_size" ]; then
        echo "ring_size=$ring_size"
    fi
    if [ -n "$ulysses_size" ]; then
        echo "ulysses_size=$ulysses_size"
    fi
    echo "out_dir=$OUT_DIR"
}

run_and_time() {
    local label="$1"
    shift
    local start_ts
    local end_ts
    local elapsed

    echo "============= ${label} ============="
    start_ts=$(date +%s)
    "$@"
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    echo "${label} elapsed: ${elapsed}s"
    LAST_ELAPSED=${elapsed}
}

print_acceleration_percent() {
    local baseline="$1"
    local target="$2"
    local label="$3"
    local accel

    if [ "$baseline" -gt 0 ] && [ "$target" -gt 0 ]; then
        accel=$(( (baseline - target) * 100 / baseline ))
        echo "acceleration(${label}): ${accel}%"
    fi
}

if [ "$1" = "2" ]; then
    echo "-------------run t2v-1.3B multi-GPU (FSDP)-------------"
    print_key_params "2" "$GPUS_BASE" "$SIZE_MULTI" "$FRAME_NUM" "" ""
    torchrun --nproc_per_node=$GPUS_BASE $PY_FILE --task $TASK --ckpt_dir $CKPT_DIR --size $SIZE_MULTI --frame_num $FRAME_NUM --dit_fsdp --t5_fsdp --sample_steps $SAMPLE_STEPS --sample_shift $SAMPLE_SHIFT --sample_guide_scale $GUIDE_SCALE
elif [ "$1" = "3" ]; then
    echo "-------------run t2v-1.3B multi-GPU (Ulysses)-------------"
    print_key_params "3" "$GPUS_BASE" "$SIZE_MULTI" "$FRAME_NUM" "" "$GPUS_BASE"
    torchrun --nproc_per_node=$GPUS_BASE $PY_FILE --task $TASK --ckpt_dir $CKPT_DIR --size $SIZE_MULTI --frame_num $FRAME_NUM --dit_fsdp --t5_fsdp --ulysses_size $GPUS_BASE --sample_steps $SAMPLE_STEPS --sample_shift $SAMPLE_SHIFT --sample_guide_scale $GUIDE_SCALE
elif [ "$1" = "4" ]; then
    echo "-------------run t2v-1.3B fastest 8-GPU mode-------------"
    print_key_params "4" "$GPUS_FAST" "$SIZE_MULTI" "$FRAME_NUM" "$RING_SIZE_FAST" "$ULYSSES_SIZE_FAST"
    torchrun --nproc_per_node=$GPUS_FAST $PY_FILE --task $TASK --ckpt_dir $CKPT_DIR --size $SIZE_MULTI --frame_num $FRAME_NUM --dit_fsdp --t5_fsdp --ring_size $RING_SIZE_FAST --ulysses_size $ULYSSES_SIZE_FAST --sample_steps $SAMPLE_STEPS --sample_shift $SAMPLE_SHIFT --sample_guide_scale $GUIDE_SCALE
elif [ "$1" = "6" ]; then
    echo "-------------run t2v-1.3B long-video direct mode (8-GPU Ring)-------------"
    print_key_params "6" "$GPUS_FAST" "$SIZE_MULTI" "$FRAME_NUM" "$RING_SIZE_FAST" "$ULYSSES_SIZE_FAST"
    mkdir -p "$OUT_DIR"
    output_file="$OUT_DIR/${TASK}_ring_$(date +%Y%m%d_%H%M%S).mp4"
    torchrun --nproc_per_node=$GPUS_FAST $PY_FILE --task $TASK --ckpt_dir $CKPT_DIR --size $SIZE_MULTI --frame_num $FRAME_NUM --save_file "$output_file" --dit_fsdp --t5_fsdp --ring_size $RING_SIZE_FAST --ulysses_size $ULYSSES_SIZE_FAST --sample_steps $SAMPLE_STEPS --sample_shift $SAMPLE_SHIFT --sample_guide_scale $GUIDE_SCALE
elif [ "$1" = "7" ]; then
    # 这个几乎比8-GPU ring快了40%
    echo "-------------run t2v-1.3B long-video direct mode (8-GPU Ring+Ulysses)-------------"
    print_key_params "7" "$GPUS_FAST" "$SIZE_MULTI" "$FRAME_NUM" "$RING_SIZE_MIX" "$ULYSSES_SIZE_MIX"
    mkdir -p "$OUT_DIR"
    output_file="$OUT_DIR/${TASK}_ring_ulysses_$(date +%Y%m%d_%H%M%S).mp4"
    torchrun --nproc_per_node=$GPUS_FAST $PY_FILE --task $TASK --ckpt_dir $CKPT_DIR --size $SIZE_MULTI --frame_num $FRAME_NUM --save_file "$output_file" --dit_fsdp --t5_fsdp --ring_size $RING_SIZE_MIX --ulysses_size $ULYSSES_SIZE_MIX --sample_steps $SAMPLE_STEPS --sample_shift $SAMPLE_SHIFT --sample_guide_scale $GUIDE_SCALE
elif [ "$1" = "8" ]; then
    echo "-------------run t2v-1.3B segmented long-video mode (generate+concat)-------------"
    print_key_params "8" "$GPUS_FAST" "$SIZE_MULTI" "$SEGMENT_FRAME_NUM" "$RING_SIZE_FAST" "$ULYSSES_SIZE_FAST"
    echo "segment_count=$SEGMENT_COUNT"
    mkdir -p "$OUT_DIR" "$SEGMENT_DIR"
    list_file="$SEGMENT_DIR/concat_list.txt"
    : > "$list_file"

    for ((i=0; i<SEGMENT_COUNT; i++)); do
        seg_name=$(printf "%s_%03d.mp4" "$SEGMENT_PREFIX" "$i")
        seg_path="$SEGMENT_DIR/$seg_name"
        echo "[segment $((i + 1))/$SEGMENT_COUNT] generating $seg_path"
        torchrun --nproc_per_node=$GPUS_FAST $PY_FILE \
            --task $TASK \
            --ckpt_dir $CKPT_DIR \
            --size $SIZE_MULTI \
            --frame_num $SEGMENT_FRAME_NUM \
            --save_file "$seg_path" \
            --dit_fsdp \
            --t5_fsdp \
            --ring_size $RING_SIZE_FAST \
            --ulysses_size $ULYSSES_SIZE_FAST \
            --sample_steps $SAMPLE_STEPS \
            --sample_shift $SAMPLE_SHIFT \
            --sample_guide_scale $GUIDE_SCALE

        if [ ! -f "$seg_path" ]; then
            echo "segment generate failed: $seg_path"
            exit 1
        fi
        printf "file '%s'\n" "$seg_name" >> "$list_file"
    done

    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "ffmpeg not found. Segments are ready under $SEGMENT_DIR"
        echo "Install ffmpeg, then run:"
        echo "ffmpeg -y -f concat -safe 0 -i $list_file -c copy $LONG_OUTPUT_FILE"
        exit 1
    fi

    mkdir -p "$(dirname "$LONG_OUTPUT_FILE")"
    ffmpeg -y -f concat -safe 0 -i "$list_file" -c copy "$LONG_OUTPUT_FILE"
    echo "final long video: $LONG_OUTPUT_FILE"
elif [ "$1" = "9" ]; then
    echo "-------------run t2v-1.3B multi-prompt generation (parallel on different GPUs)-------------"
    print_key_params "9" "$GPUS_FAST" "$SIZE_MULTI" "$FRAME_NUM" "" ""
    echo "prompt_file=$PROMPT_FILE"
    mkdir -p "$OUT_DIR"

    if [ ! -f "$PROMPT_FILE" ]; then
        echo "Error: $PROMPT_FILE not found!"
        exit 1
    fi

    prompt_idx=0
    batch_count=0
    total_started=0

    declare -a batch_pids
    declare -a batch_outputs
    declare -a batch_indices

    wait_batch_jobs() {
        local i
        for i in "${!batch_pids[@]}"; do
            wait "${batch_pids[$i]}"
            if [ $? -ne 0 ] || [ ! -f "${batch_outputs[$i]}" ]; then
                echo "Warning: Failed to generate ${batch_outputs[$i]} for prompt ${batch_indices[$i]}"
            else
                echo "Successfully saved: ${batch_outputs[$i]}"
            fi
        done
        batch_pids=()
        batch_outputs=()
        batch_indices=()
        batch_count=0
    }

    while IFS= read -r prompt; do
        # Skip empty lines and comments
        [[ -z "$prompt" ]] && continue
        [[ "$prompt" =~ ^# ]] && continue

        ((prompt_idx++))
        gpu_id=$(( (prompt_idx - 1) % GPUS_FAST ))
        output_file="$OUT_DIR/prompt_${prompt_idx}_$(date +%Y%m%d_%H%M%S).mp4"

        echo ""
        echo "======= [Prompt $prompt_idx / GPU $gpu_id] ======="
        echo "Prompt: $prompt"
        echo "Output: $output_file"
        echo ""

        CUDA_VISIBLE_DEVICES=$gpu_id python $PY_FILE \
            --task $TASK \
            --ckpt_dir $CKPT_DIR \
            --size $SIZE_MULTI \
            --frame_num $FRAME_NUM \
            --save_file "$output_file" \
            --prompt "$prompt" \
            --sample_steps $SAMPLE_STEPS \
            --sample_shift $SAMPLE_SHIFT \
            --sample_guide_scale $GUIDE_SCALE &

        batch_pids+=("$!")
        batch_outputs+=("$output_file")
        batch_indices+=("$prompt_idx")
        batch_count=$((batch_count + 1))
        total_started=$((total_started + 1))

        if [ "$batch_count" -ge "$GPUS_FAST" ]; then
            echo "Waiting for current batch ($batch_count jobs) to finish ..."
            wait_batch_jobs
        fi
    done < "$PROMPT_FILE"

    if [ "$batch_count" -gt 0 ]; then
        echo "Waiting for final batch ($batch_count jobs) to finish ..."
        wait_batch_jobs
    fi

    echo ""
    echo "======= All prompts processed! Total jobs: $total_started ======="
else
    echo "-------------run t2v-1.3B 1-GPU-------------"
    print_key_params "default" "1" "$SIZE_SINGLE" "$FRAME_NUM" "" ""
    mkdir -p "$OUT_DIR"
    output_file="$OUT_DIR/${TASK}_1gpu_$(date +%Y%m%d_%H%M%S).mp4"
    python $PY_FILE --task $TASK --size $SIZE_SINGLE --ckpt_dir $CKPT_DIR --frame_num $FRAME_NUM --save_file "$output_file" --sample_steps $SAMPLE_STEPS --sample_shift $SAMPLE_SHIFT --sample_guide_scale $GUIDE_SCALE
fi

