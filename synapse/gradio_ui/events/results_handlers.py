"""
Synapse Music V12 - Results Handlers Module

Contains event handlers and helper functions related to result display,
quality scoring, subtitle generation, batch management, navigation,
and Synapse Composer metadata handling.
"""

import os
import json
import datetime
import math
import re
import tempfile
import shutil
import zipfile
import time as time_module
from typing import Dict, Any, Optional, List

import gradio as gr
from loguru import logger

from synapse.gradio_ui.i18n import t
from synapse.gradio_ui.events.generation_handlers import parse_and_validate_timesteps
from synapse.inference import generate_music, GenerationParams, GenerationConfig
from synapse.audio_utils import save_audio


# Hugging Face Space environment detection for ZeroGPU support
IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None


def _get_spaces_gpu_decorator(duration=120):
    """
    Get the @spaces.GPU decorator if running in a Hugging Face Space environment.

    Returns an identity decorator if not running in a Space environment.
    """
    if IS_HUGGINGFACE_SPACE:
        try:
            import spaces
            return spaces.GPU(duration=duration)
        except ImportError:
            logger.warning("spaces package not found, GPU decorator disabled")
            return lambda func: func

    return lambda func: func


def parse_lrc_to_subtitles(
    lrc_text: str,
    total_duration: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Parse LRC lyrics text into Gradio subtitle format with smart post-processing.

    Lines that begin very close together, such as section tags followed
    immediately by lyrics, are merged to prevent subtitles from disappearing
    too quickly.

    Args:
        lrc_text:
            LRC-formatted lyrics string.

        total_duration:
            Total audio duration in seconds.

    Returns:
        List of subtitle dictionaries.
    """
    if not lrc_text or not lrc_text.strip():
        return []

    timestamp_pattern = r'\[(\d{2}):(\d{2})\.(\d{2,3})\]'

    raw_entries = []
    lines = lrc_text.strip().split('\n')

    for line in lines:
        line = line.strip()

        if not line:
            continue

        timestamps = re.findall(timestamp_pattern, line)

        if not timestamps:
            continue

        text = re.sub(timestamp_pattern, '', line).strip()

        if not text:
            continue

        start_minutes, start_seconds, start_centiseconds = timestamps[0]

        cs = int(start_centiseconds)

        start_time = (
            int(start_minutes) * 60
            + int(start_seconds)
            + (
                cs / 100.0
                if len(start_centiseconds) == 2
                else cs / 1000.0
            )
        )

        end_time = None

        if len(timestamps) >= 2:
            end_minutes, end_seconds, end_centiseconds = timestamps[1]

            cs_end = int(end_centiseconds)

            end_time = (
                int(end_minutes) * 60
                + int(end_seconds)
                + (
                    cs_end / 100.0
                    if len(end_centiseconds) == 2
                    else cs_end / 1000.0
                )
            )

        raw_entries.append({
            "start": start_time,
            "explicit_end": end_time,
            "text": text,
        })

    raw_entries.sort(key=lambda x: x["start"])

    if not raw_entries:
        return []

    # Merge subtitle entries that would otherwise appear too briefly.
    MIN_DISPLAY_DURATION = 2.0

    merged_entries = []
    i = 0

    while i < len(raw_entries):
        current = raw_entries[i]

        combined_text = current["text"]
        combined_start = current["start"]
        combined_explicit_end = current["explicit_end"]

        next_idx = i + 1

        while next_idx < len(raw_entries):
            next_entry = raw_entries[next_idx]

            gap = next_entry["start"] - combined_start

            if gap < MIN_DISPLAY_DURATION:
                combined_text += "\n" + next_entry["text"]

                if next_entry["explicit_end"]:
                    combined_explicit_end = next_entry["explicit_end"]

                next_idx += 1
            else:
                break

        merged_entries.append({
            "start": combined_start,
            "explicit_end": combined_explicit_end,
            "text": combined_text,
        })

        i = next_idx

    subtitles = []

    for i, entry in enumerate(merged_entries):
        start = entry["start"]
        text = entry["text"]

        if entry["explicit_end"] is not None:
            end = entry["explicit_end"]
        else:
            if i + 1 < len(merged_entries):
                end = merged_entries[i + 1]["start"]
            else:
                if total_duration is not None and total_duration > start:
                    end = total_duration
                else:
                    end = start + 5.0

        if end <= start:
            end = start + 3.0

        subtitles.append({
            "text": text,
            "timestamp": [start, end],
        })

    return subtitles


def _format_vtt_timestamp(seconds: float) -> str:
    """
    Format seconds as a WebVTT timestamp.

    Format:
        HH:MM:SS.mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def lrc_to_vtt_file(
    lrc_text: str,
    total_duration: float = None
) -> Optional[str]:
    """
    Convert LRC lyrics into a WebVTT subtitle file.

    The resulting VTT file can be passed directly to Gradio's Audio
    component and rendered using the browser's native subtitle track.

    Args:
        lrc_text:
            LRC-formatted lyrics.

        total_duration:
            Total audio duration in seconds.

    Returns:
        Path to the generated VTT file, or None if conversion fails.
    """
    if not lrc_text or not lrc_text.strip():
        return None

    subtitles = parse_lrc_to_subtitles(
        lrc_text,
        total_duration=total_duration,
    )

    if not subtitles:
        return None

    vtt_lines = ["WEBVTT", ""]

    for i, subtitle in enumerate(subtitles):
        start_time = subtitle["timestamp"][0]
        end_time = subtitle["timestamp"][1]
        text = subtitle["text"]

        vtt_lines.append(str(i + 1))
        vtt_lines.append(
            f"{_format_vtt_timestamp(start_time)} --> "
            f"{_format_vtt_timestamp(end_time)}"
        )
        vtt_lines.append(text)
        vtt_lines.append("")

    vtt_content = "\n".join(vtt_lines)

    try:
        temp_dir = tempfile.mkdtemp(prefix="synapse_vtt_")

        vtt_path = os.path.join(
            temp_dir,
            "subtitles.vtt",
        )

        with open(vtt_path, "w", encoding="utf-8") as f:
            f.write(vtt_content)

        return vtt_path

    except Exception as e:
        logger.error(
            f"[lrc_to_vtt_file] Failed to create VTT file: {e}"
        )

        return None


def _build_generation_info(
    composer_metadata: Optional[Dict[str, Any]],
    time_costs: Dict[str, float],
    seed_value: str,
    inference_steps: int,
    num_audios: int,
) -> str:
    """
    Build Synapse Music generation information from result data.

    Args:
        composer_metadata:
            Synapse Composer-generated metadata dictionary.

        time_costs:
            Unified generation timing dictionary.

        seed_value:
            Seed value used during generation.

        inference_steps:
            Number of Synapse Music inference steps.

        num_audios:
            Number of generated songs.

    Returns:
        Formatted generation information string.
    """
    info_parts = []

    songs_label = f"({num_audios} songs)"

    if time_costs:
        composer_total = time_costs.get(
            "lm_total_time",
            time_costs.get("composer_total_time", 0.0),
        )

        dit_total = time_costs.get(
            "dit_total_time_cost",
            0.0,
        )

        generation_total = composer_total + dit_total

        if generation_total > 0:
            avg_per_song = (
                generation_total / num_audios
                if num_audios > 0
                else 0
            )

            gen_lines = [
                (
                    f"**🎵 Total generation time {songs_label}: "
                    f"{generation_total:.2f}s**"
                ),
                f"\n**{avg_per_song:.2f}s per song**",
            ]

            if composer_total > 0:
                gen_lines.append(
                    f"- Synapse Composer phase {songs_label}: "
                    f"{composer_total:.2f}s"
                )

            if dit_total > 0:
                gen_lines.append(
                    f"- Synapse Music phase {songs_label}: "
                    f"{dit_total:.2f}s"
                )

            info_parts.append("\n".join(gen_lines))

    if time_costs:
        audio_conversion_time = time_costs.get(
            "audio_conversion_time",
            0.0,
        )

        auto_score_time = time_costs.get(
            "auto_score_time",
            0.0,
        )

        auto_lrc_time = time_costs.get(
            "auto_lrc_time",
            0.0,
        )

        processing_total = (
            audio_conversion_time
            + auto_score_time
            + auto_lrc_time
        )

        if processing_total > 0:
            proc_lines = [
                (
                    f"**🔧 Total processing time {songs_label}: "
                    f"{processing_total:.2f}s**"
                ),
            ]

            if audio_conversion_time > 0:
                info_format = time_costs.get(
                    "audio_format",
                    "mp3",
                )

                proc_lines.append(
                    f"- Audio conversion to {info_format} "
                    f"{songs_label}: {audio_conversion_time:.2f}s"
                )

            if auto_score_time > 0:
                proc_lines.append(
                    f"- Quality scoring {songs_label}: "
                    f"{auto_score_time:.2f}s"
                )

            if auto_lrc_time > 0:
                proc_lines.append(
                    f"- Lyrics timestamp detection {songs_label}: "
                    f"{auto_lrc_time:.2f}s"
                )

            info_parts.append("\n".join(proc_lines))

    return "\n\n".join(info_parts)


def store_batch_in_queue(
    batch_queue,
    batch_index,
    audio_paths,
    generation_info,
    seeds,
    codes=None,
    scores=None,
    allow_lm_batch=False,
    batch_size=2,
    generation_params=None,
    composer_generated_metadata=None,
    extra_outputs=None,
    status="completed",
):
    """
    Store Synapse Music batch results in the queue with all generation
    parameters required for history, restoration, scoring, and LRC generation.

    Args:
        codes:
            Audio codes used for generation.

        scores:
            Quality-score displays for each generated song.

        allow_lm_batch:
            Whether Composer batch generation was enabled.

        batch_size:
            Number of songs in the batch.

        generation_params:
            Complete dictionary of generation parameters.

        composer_generated_metadata:
            Metadata produced by Synapse Composer.

        extra_outputs:
            Additional inference tensors and metadata required for
            quality scoring and lyrics timestamp generation.
    """
    batch_queue[batch_index] = {
        "status": status,
        "audio_paths": audio_paths,
        "generation_info": generation_info,
        "seeds": seeds,
        "codes": codes,
        "scores": scores if scores else [""] * 8,
        "allow_lm_batch": allow_lm_batch,
        "batch_size": batch_size,
        "generation_params": (
            generation_params
            if generation_params
            else {}
        ),
        "composer_generated_metadata": composer_generated_metadata,
        "extra_outputs": (
            extra_outputs
            if extra_outputs
            else {}
        ),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return batch_queue


def update_batch_indicator(current_batch, total_batches):
    """Update the Synapse Music batch indicator."""
    return t(
        "results.batch_indicator",
        current=current_batch + 1,
        total=total_batches,
    )


def update_navigation_buttons(current_batch, total_batches):
    """Determine Synapse Music batch navigation button states."""
    can_go_previous = current_batch > 0
    can_go_next = current_batch < total_batches - 1

    return can_go_previous, can_go_next


def send_audio_to_src_with_metadata(
    audio_file,
    composer_metadata,
):
    """
    Send a generated Synapse Music audio file to the source-audio input
    without modifying any other generation fields.

    All metadata inputs are preserved using gr.skip().

    Args:
        audio_file:
            Generated audio file path.

        composer_metadata:
            Synapse Composer metadata. Retained for API compatibility.

    Returns:
        Tuple containing:
            audio_file,
            bpm,
            caption,
            lyrics,
            duration,
            key_scale,
            language,
            time_signature,
            is_format_caption
    """
    if audio_file is None:
        return (gr.skip(),) * 9

    return (
        audio_file,
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )


def generate_with_progress(
    dit_handler,
    llm_handler,
    captions,
    lyrics,
    bpm,
    key_scale,
    time_signature,
    vocal_language,
    inference_steps,
    guidance_scale,
    random_seed_checkbox,
    seed,
    reference_audio,
    audio_duration,
    batch_size_input,
    src_audio,
    text2music_audio_code_string,
    repainting_start,
    repainting_end,
    instruction_display_gen,
    audio_cover_strength,
    task_type,
    use_adg,
    cfg_interval_start,
    cfg_interval_end,
    shift,
    infer_method,
    custom_timesteps,
    audio_format,
    lm_temperature,
    think_checkbox,
    lm_cfg_scale,
    lm_top_k,
    lm_top_p,
    lm_negative_prompt,
    use_cot_metas,
    use_cot_caption,
    use_cot_language,
    is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    auto_lrc,
    score_scale,
    lm_batch_chunk_size,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate Synapse Music audio with real-time progress tracking.
    """

    # Skip metadata CoT when the sample has already been formatted.
    actual_use_cot_metas = use_cot_metas

    if is_format_caption and use_cot_metas:
        actual_use_cot_metas = False

        logger.info(
            "[generate_with_progress] Skipping metadata CoT: "
            "sample is already formatted "
            "(is_format_caption=True)"
        )

        gr.Info(t("messages.skipping_metas_cot"))

    parsed_timesteps, has_timesteps_warning, _ = (
        parse_and_validate_timesteps(
            custom_timesteps,
            inference_steps,
        )
    )

    actual_inference_steps = inference_steps

    if parsed_timesteps is not None:
        actual_inference_steps = len(parsed_timesteps) - 1

    gen_params = GenerationParams(
        task_type=task_type,
        instruction=instruction_display_gen,
        reference_audio=reference_audio,
        src_audio=src_audio,
        audio_codes=(
            text2music_audio_code_string
            if not think_checkbox
            else ""
        ),
        caption=captions or "",
        lyrics=lyrics or "",
        instrumental=False,
        vocal_language=vocal_language,
        bpm=bpm,
        keyscale=key_scale,
        timesignature=time_signature,
        duration=audio_duration,
        inference_steps=actual_inference_steps,
        guidance_scale=guidance_scale,
        use_adg=use_adg,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_end=cfg_interval_end,
        shift=shift,
        infer_method=infer_method,
        timesteps=parsed_timesteps,
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        audio_cover_strength=audio_cover_strength,
        thinking=think_checkbox,
        lm_temperature=lm_temperature,
        lm_cfg_scale=lm_cfg_scale,
        lm_top_k=lm_top_k,
        lm_top_p=lm_top_p,
        lm_negative_prompt=lm_negative_prompt,
        use_cot_metas=actual_use_cot_metas,
        use_cot_caption=use_cot_caption,
        use_cot_language=use_cot_language,
        use_constrained_decoding=True,
    )

    if isinstance(seed, str) and seed.strip():
        if "," in seed:
            seed_list = [
                int(s.strip())
                for s in seed.split(",")
            ]
        else:
            seed_list = [int(seed.strip())]
    else:
        seed_list = None

    gen_config = GenerationConfig(
        batch_size=batch_size_input,
        allow_lm_batch=allow_lm_batch,
        use_random_seed=random_seed_checkbox,
        seeds=seed_list,
        lm_batch_chunk_size=lm_batch_chunk_size,
        constrained_decoding_debug=constrained_decoding_debug,
        audio_format=audio_format,
    )

    result = generate_music(
        dit_handler,
        llm_handler,
        params=gen_params,
        config=gen_config,
        progress=progress,
    )

    audio_outputs = [None] * 8
    all_audio_paths = []

    final_codes_list = [""] * 8
    final_scores_list = [""] * 8

    status_message = result.status_message

    seed_value_for_ui = result.extra_outputs.get(
        "seed_value",
        "",
    )

    composer_generated_metadata = result.extra_outputs.get(
        "composer_metadata",
        result.extra_outputs.get("lm_metadata", {}),
    )

    time_costs = result.extra_outputs.get(
        "time_costs",
        {},
    ).copy()

    audio_conversion_start_time = time_module.time()

    total_auto_score_time = 0.0
    total_auto_lrc_time = 0.0

    final_lrcs_list = [""] * 8
    final_subtitles_list = [None] * 8

    updated_audio_codes = (
        text2music_audio_code_string
        if not think_checkbox
        else ""
    )

    generation_info = _build_generation_info(
        composer_metadata=composer_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=(
            len(result.audios)
            if result.success
            else 0
        ),
    )

    if not result.success:
        yield (
            (None,) * 8
            + (
                None,
                generation_info,
                result.status_message,
                gr.skip(),
            )
            + (gr.skip(),) * 8
            + (gr.skip(),) * 8
            + (gr.skip(),) * 8
            + (gr.skip(),) * 8
            + (
                None,
                is_format_caption,
                None,
                None,
            )
        )

        return

    audios = result.audios

    progress(
        0.99,
        f"Converting audio to {audio_format}...",
    )

    clear_scores = [
        gr.update(
            value="",
            visible=True,
        )
        for _ in range(8)
    ]

    clear_codes = [
        gr.update(
            value="",
            visible=True,
        )
        for _ in range(8)
    ]

    clear_lrcs = [
        gr.update(
            value="",
            visible=True,
        )
        for _ in range(8)
    ]

    clear_accordions = [
        gr.skip()
        for _ in range(8)
    ]

    dump_audio = [
        gr.update(
            value=None,
            subtitles=None,
        )
        for _ in range(8)
    ]

    yield (
        dump_audio[0],
        dump_audio[1],
        dump_audio[2],
        dump_audio[3],
        dump_audio[4],
        dump_audio[5],
        dump_audio[6],
        dump_audio[7],

        None,
        generation_info,
        "Clearing previous Synapse Music results...",
        gr.skip(),

        clear_scores[0],
        clear_scores[1],
        clear_scores[2],
        clear_scores[3],
        clear_scores[4],
        clear_scores[5],
        clear_scores[6],
        clear_scores[7],

        clear_codes[0],
        clear_codes[1],
        clear_codes[2],
        clear_codes[3],
        clear_codes[4],
        clear_codes[5],
        clear_codes[6],
        clear_codes[7],

        clear_accordions[0],
        clear_accordions[1],
        clear_accordions[2],
        clear_accordions[3],
        clear_accordions[4],
        clear_accordions[5],
        clear_accordions[6],
        clear_accordions[7],

        clear_lrcs[0],
        clear_lrcs[1],
        clear_lrcs[2],
        clear_lrcs[3],
        clear_lrcs[4],
        clear_lrcs[5],
        clear_lrcs[6],
        clear_lrcs[7],

        composer_generated_metadata,
        is_format_caption,
        None,
        None,
    )

    time_module.sleep(0.1)

    for i in range(8):
        if i < len(audios):
            key = audios[i]["key"]
            audio_tensor = audios[i]["tensor"]
            sample_rate = audios[i]["sample_rate"]
            audio_params = audios[i]["params"]

            temp_dir = tempfile.mkdtemp(
                prefix="synapse_gradio_results_"
            )

            os.makedirs(
                temp_dir,
                exist_ok=True,
            )

            json_path = os.path.join(
                temp_dir,
                f"{key}.json",
            )

            audio_path = os.path.join(
                temp_dir,
                f"{key}.{audio_format}",
            )

            save_audio(
                audio_data=audio_tensor,
                output_path=audio_path,
                sample_rate=sample_rate,
                format=audio_format,
                channels_first=True,
            )

            with open(
                json_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    audio_params,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            audio_outputs[i] = audio_path

            all_audio_paths.append(audio_path)
            all_audio_paths.append(json_path)

            code_str = audio_params.get(
                "audio_codes",
                "",
            )

            final_codes_list[i] = code_str

            scores_ui_updates = [
                gr.skip()
                for _ in range(8)
            ]

            score_str = "Done!"

            if auto_score:
                auto_score_start = time_module.time()

                sample_tensor_data = None

                try:
                    full_pred = result.extra_outputs.get(
                        "pred_latents"
                    )

                    if (
                        full_pred is not None
                        and i < full_pred.shape[0]
                    ):
                        sample_tensor_data = {
                            "pred_latent":
                                full_pred[i:i + 1],

                            "encoder_hidden_states":
                                result.extra_outputs.get(
                                    "encoder_hidden_states"
                                )[i:i + 1]
                                if result.extra_outputs.get(
                                    "encoder_hidden_states"
                                ) is not None
                                else None,

                            "encoder_attention_mask":
                                result.extra_outputs.get(
                                    "encoder_attention_mask"
                                )[i:i + 1]
                                if result.extra_outputs.get(
                                    "encoder_attention_mask"
                                ) is not None
                                else None,

                            "context_latents":
                                result.extra_outputs.get(
                                    "context_latents"
                                )[i:i + 1]
                                if result.extra_outputs.get(
                                    "context_latents"
                                ) is not None
                                else None,

                            "lyric_token_ids":
                                result.extra_outputs.get(
                                    "lyric_token_idss"
                                )[i:i + 1]
                                if result.extra_outputs.get(
                                    "lyric_token_idss"
                                ) is not None
                                else None,
                        }

                        if any(
                            v is None
                            for v in sample_tensor_data.values()
                        ):
                            sample_tensor_data = None

                except Exception as e:
                    logger.warning(
                        "[Synapse Auto Score] Failed to prepare "
                        f"tensor data for sample {i}: {e}"
                    )

                    sample_tensor_data = None

                score_str = calculate_score_handler(
                    llm_handler,
                    code_str,
                    captions,
                    lyrics,
                    composer_generated_metadata,
                    bpm,
                    key_scale,
                    time_signature,
                    audio_duration,
                    vocal_language,
                    score_scale,
                    dit_handler,
                    sample_tensor_data,
                    inference_steps,
                )

                auto_score_end = time_module.time()

                total_auto_score_time += (
                    auto_score_end
                    - auto_score_start
                )

            scores_ui_updates[i] = score_str
            final_scores_list[i] = score_str

            if auto_lrc:
                auto_lrc_start = time_module.time()

                logger.info(
                    "[synapse_auto_lrc] Starting LRC generation "
                    f"for sample {i + 1}"
                )

                try:
                    pred_latents = result.extra_outputs.get(
                        "pred_latents"
                    )

                    encoder_hidden_states = (
                        result.extra_outputs.get(
                            "encoder_hidden_states"
                        )
                    )

                    encoder_attention_mask = (
                        result.extra_outputs.get(
                            "encoder_attention_mask"
                        )
                    )

                    context_latents = result.extra_outputs.get(
                        "context_latents"
                    )

                    lyric_token_idss = result.extra_outputs.get(
                        "lyric_token_idss"
                    )

                    logger.info(
                        "[synapse_auto_lrc] "
                        f"pred_latents: {pred_latents is not None}, "
                        f"encoder_hidden_states: "
                        f"{encoder_hidden_states is not None}, "
                        f"encoder_attention_mask: "
                        f"{encoder_attention_mask is not None}, "
                        f"context_latents: "
                        f"{context_latents is not None}, "
                        f"lyric_token_idss: "
                        f"{lyric_token_idss is not None}"
                    )

                    if all(
                        x is not None
                        for x in [
                            pred_latents,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            context_latents,
                            lyric_token_idss,
                        ]
                    ):
                        sample_pred_latent = (
                            pred_latents[i:i + 1]
                        )

                        sample_encoder_hidden_states = (
                            encoder_hidden_states[i:i + 1]
                        )

                        sample_encoder_attention_mask = (
                            encoder_attention_mask[i:i + 1]
                        )

                        sample_context_latents = (
                            context_latents[i:i + 1]
                        )

                        sample_lyric_token_ids = (
                            lyric_token_idss[i:i + 1]
                        )

                        actual_duration = audio_duration

                        if (
                            actual_duration is None
                            or actual_duration <= 0
                        ):
                            latent_length = (
                                pred_latents.shape[1]
                            )

                            actual_duration = (
                                latent_length / 25.0
                            )

                        lrc_result = (
                            dit_handler.get_lyric_timestamp(
                                pred_latent=sample_pred_latent,
                                encoder_hidden_states=(
                                    sample_encoder_hidden_states
                                ),
                                encoder_attention_mask=(
                                    sample_encoder_attention_mask
                                ),
                                context_latents=(
                                    sample_context_latents
                                ),
                                lyric_token_ids=(
                                    sample_lyric_token_ids
                                ),
                                total_duration_seconds=float(
                                    actual_duration
                                ),
                                vocal_language=(
                                    vocal_language or "en"
                                ),
                                inference_steps=int(
                                    inference_steps
                                ),
                                seed=42,
                            )
                        )

                        logger.info(
                            "[synapse_auto_lrc] "
                            f"LRC result for sample {i + 1}: "
                            f"success="
                            f"{lrc_result.get('success')}"
                        )

                        if lrc_result.get("success"):
                            lrc_text = lrc_result.get(
                                "lrc_text",
                                "",
                            )

                            final_lrcs_list[i] = lrc_text

                            logger.info(
                                "[synapse_auto_lrc] "
                                f"LRC text length for sample "
                                f"{i + 1}: {len(lrc_text)}"
                            )

                            vtt_path = lrc_to_vtt_file(
                                lrc_text,
                                total_duration=float(
                                    actual_duration
                                ),
                            )

                            final_subtitles_list[i] = (
                                vtt_path
                            )

                    else:
                        logger.warning(
                            "[synapse_auto_lrc] Missing required "
                            f"extra outputs for sample {i + 1}"
                        )

                except Exception as e:
                    logger.warning(
                        "[synapse_auto_lrc] Failed to generate "
                        f"LRC for sample {i + 1}: {e}"
                    )

                auto_lrc_end = time_module.time()

                total_auto_lrc_time += (
                    auto_lrc_end
                    - auto_lrc_start
                )

            status_message = (
                f"Synapse Music Ready: "
                f"{i + 1}/{len(audios)}"
            )

            has_lrc = bool(
                final_lrcs_list[i]
            )

            has_score = (
                bool(score_str)
                and score_str != "Done!"
            )

            has_content = (
                bool(code_str)
                or has_lrc
                or has_score
            )

            current_audio_updates = [
                gr.skip()
                for _ in range(8)
            ]

            current_audio_updates[i] = audio_path

            codes_display_updates = [
                gr.skip()
                for _ in range(8)
            ]

            codes_display_updates[i] = gr.update(
                value=code_str,
                visible=True,
            )

            details_accordion_updates = [
                gr.skip()
                for _ in range(8)
            ]

            if auto_score or auto_lrc:
                details_accordion_updates[i] = (
                    gr.Accordion(open=True)
                )

            lrc_clear_updates = [
                gr.skip()
                for _ in range(8)
            ]

            lrc_clear_updates[i] = gr.update(
                value="",
                visible=True,
            )

            yield (
                current_audio_updates[0],
                current_audio_updates[1],
                current_audio_updates[2],
                current_audio_updates[3],
                current_audio_updates[4],
                current_audio_updates[5],
                current_audio_updates[6],
                current_audio_updates[7],

                all_audio_paths,
                generation_info,
                status_message,
                seed_value_for_ui,

                scores_ui_updates[0],
                scores_ui_updates[1],
                scores_ui_updates[2],
                scores_ui_updates[3],
                scores_ui_updates[4],
                scores_ui_updates[5],
                scores_ui_updates[6],
                scores_ui_updates[7],

                codes_display_updates[0],
                codes_display_updates[1],
                codes_display_updates[2],
                codes_display_updates[3],
                codes_display_updates[4],
                codes_display_updates[5],
                codes_display_updates[6],
                codes_display_updates[7],

                details_accordion_updates[0],
                details_accordion_updates[1],
                details_accordion_updates[2],
                details_accordion_updates[3],
                details_accordion_updates[4],
                details_accordion_updates[5],
                details_accordion_updates[6],
                details_accordion_updates[7],

                lrc_clear_updates[0],
                lrc_clear_updates[1],
                lrc_clear_updates[2],
                lrc_clear_updates[3],
                lrc_clear_updates[4],
                lrc_clear_updates[5],
                lrc_clear_updates[6],
                lrc_clear_updates[7],

                composer_generated_metadata,
                is_format_caption,
                None,
                None,
            )

            time_module.sleep(0.05)

            if has_lrc:
                skip_audio = [
                    gr.skip()
                    for _ in range(8)
                ]

                skip_scores = [
                    gr.skip()
                    for _ in range(8)
                ]

                skip_codes = [
                    gr.skip()
                    for _ in range(8)
                ]

                skip_accordions = [
                    gr.skip()
                    for _ in range(8)
                ]

                lrc_actual_updates = [
                    gr.skip()
                    for _ in range(8)
                ]

                lrc_actual_updates[i] = gr.update(
                    value=final_lrcs_list[i],
                    visible=True,
                )

                yield (
                    skip_audio[0],
                    skip_audio[1],
                    skip_audio[2],
                    skip_audio[3],
                    skip_audio[4],
                    skip_audio[5],
                    skip_audio[6],
                    skip_audio[7],

                    gr.skip(),
                    gr.skip(),
                    gr.skip(),
                    gr.skip(),

                    skip_scores[0],
                    skip_scores[1],
                    skip_scores[2],
                    skip_scores[3],
                    skip_scores[4],
                    skip_scores[5],
                    skip_scores[6],
                    skip_scores[7],

                    skip_codes[0],
                    skip_codes[1],
                    skip_codes[2],
                    skip_codes[3],
                    skip_codes[4],
                    skip_codes[5],
                    skip_codes[6],
                    skip_codes[7],

                    skip_accordions[0],
                    skip_accordions[1],
                    skip_accordions[2],
                    skip_accordions[3],
                    skip_accordions[4],
                    skip_accordions[5],
                    skip_accordions[6],
                    skip_accordions[7],

                    lrc_actual_updates[0],
                    lrc_actual_updates[1],
                    lrc_actual_updates[2],
                    lrc_actual_updates[3],
                    lrc_actual_updates[4],
                    lrc_actual_updates[5],
                    lrc_actual_updates[6],
                    lrc_actual_updates[7],

                    gr.skip(),
                    gr.skip(),
                    None,
                    None,
                )

        time_module.sleep(0.05)

    audio_conversion_end_time = (
        time_module.time()
    )

    audio_conversion_time = (
        audio_conversion_end_time
        - audio_conversion_start_time
    )

    if audio_conversion_time > 0:
        time_costs["audio_conversion_time"] = (
            audio_conversion_time
        )

    if total_auto_score_time > 0:
        time_costs["auto_score_time"] = (
            total_auto_score_time
        )

    if total_auto_lrc_time > 0:
        time_costs["auto_lrc_time"] = (
            total_auto_lrc_time
        )

    if "pipeline_total_time" in time_costs:
        time_costs["pipeline_total_time"] += (
            audio_conversion_time
            + total_auto_score_time
            + total_auto_lrc_time
        )

    generation_info = _build_generation_info(
        composer_metadata=composer_generated_metadata,
        time_costs=time_costs,
        seed_value=seed_value_for_ui,
        inference_steps=inference_steps,
        num_audios=len(result.audios),
    )

    final_codes_display_updates = [
        gr.skip()
        for _ in range(8)
    ]

    final_accordion_updates = [
        gr.skip()
        for _ in range(8)
    ]

    audio_playback_updates = [
        gr.update(playback_position=0)
        for _ in range(8)
    ]

    yield (
        audio_playback_updates[0],
        audio_playback_updates[1],
        audio_playback_updates[2],
        audio_playback_updates[3],
        audio_playback_updates[4],
        audio_playback_updates[5],
        audio_playback_updates[6],
        audio_playback_updates[7],

        all_audio_paths,
        generation_info,
        "Synapse Music Generation Complete",
        seed_value_for_ui,

        final_scores_list[0],
        final_scores_list[1],
        final_scores_list[2],
        final_scores_list[3],
        final_scores_list[4],
        final_scores_list[5],
        final_scores_list[6],
        final_scores_list[7],

        final_codes_display_updates[0],
        final_codes_display_updates[1],
        final_codes_display_updates[2],
        final_codes_display_updates[3],
        final_codes_display_updates[4],
        final_codes_display_updates[5],
        final_codes_display_updates[6],
        final_codes_display_updates[7],

        final_accordion_updates[0],
        final_accordion_updates[1],
        final_accordion_updates[2],
        final_accordion_updates[3],
        final_accordion_updates[4],
        final_accordion_updates[5],
        final_accordion_updates[6],
        final_accordion_updates[7],

        final_lrcs_list[0],
        final_lrcs_list[1],
        final_lrcs_list[2],
        final_lrcs_list[3],
        final_lrcs_list[4],
        final_lrcs_list[5],
        final_lrcs_list[6],
        final_lrcs_list[7],

        composer_generated_metadata,
        is_format_caption,

        {
            **result.extra_outputs,
            "composer_metadata": composer_generated_metadata,
            "lrcs": final_lrcs_list,
            "subtitles": final_subtitles_list,
        },

        final_codes_list,
    )


def calculate_score_handler(
    llm_handler,
    audio_codes_str,
    caption,
    lyrics,
    composer_metadata,
    bpm,
    key_scale,
    time_signature,
    audio_duration,
    vocal_language,
    score_scale,
    dit_handler,
    extra_tensor_data,
    inference_steps,
):
    """
    Calculate Synapse Music quality scores.

    PMI removes condition bias using:

        score = log P(condition | codes) - log P(condition)

    Cover and Repaint modes may not provide audio codes. In that case,
    Synapse falls back to DiT lyrics-alignment scoring when the required
    inference tensors are available.
    """
    from synapse.test_time_scaling import (
        calculate_pmi_score_per_condition,
    )

    has_audio_codes = (
        audio_codes_str
        and audio_codes_str.strip()
    )

    has_dit_alignment_data = (
        dit_handler
        and extra_tensor_data
        and lyrics
        and lyrics.strip()
    )

    if (
        not has_audio_codes
        and not has_dit_alignment_data
    ):
        return t("messages.no_codes")

    try:
        scores_per_condition = {}
        global_score = 0.0
        alignment_report = ""

        if has_audio_codes:
            if not llm_handler.llm_initialized:
                if not has_dit_alignment_data:
                    return t(
                        "messages.lm_not_initialized"
                    )

            else:
                metadata = {}

                if (
                    composer_metadata
                    and isinstance(
                        composer_metadata,
                        dict,
                    )
                ):
                    metadata.update(
                        composer_metadata
                    )

                if (
                    bpm is not None
                    and "bpm" not in metadata
                ):
                    try:
                        metadata["bpm"] = int(bpm)
                    except (TypeError, ValueError):
                        pass

                if (
                    caption
                    and "caption" not in metadata
                ):
                    metadata["caption"] = caption

                if (
                    audio_duration is not None
                    and audio_duration > 0
                    and "duration" not in metadata
                ):
                    try:
                        metadata["duration"] = int(
                            audio_duration
                        )
                    except (TypeError, ValueError):
                        pass

                if (
                    key_scale
                    and key_scale.strip()
                    and "keyscale" not in metadata
                ):
                    metadata["keyscale"] = (
                        key_scale.strip()
                    )

                if (
                    vocal_language
                    and vocal_language.strip()
                    and "language" not in metadata
                ):
                    metadata["language"] = (
                        vocal_language.strip()
                    )

                if (
                    time_signature
                    and time_signature.strip()
                    and "timesignature" not in metadata
                ):
                    metadata["timesignature"] = (
                        time_signature.strip()
                    )

                (
                    scores_per_condition,
                    global_score,
                    status,
                ) = calculate_pmi_score_per_condition(
                    llm_handler=llm_handler,
                    audio_codes=audio_codes_str,
                    caption=caption or "",
                    lyrics=lyrics or "",
                    metadata=(
                        metadata
                        if metadata
                        else None
                    ),
                    temperature=1.0,
                    topk=10,
                    score_scale=score_scale,
                )

        if has_dit_alignment_data:
            try:
                align_result = (
                    dit_handler.get_lyric_score(
                        pred_latent=(
                            extra_tensor_data.get(
                                "pred_latent"
                            )
                        ),
                        encoder_hidden_states=(
                            extra_tensor_data.get(
                                "encoder_hidden_states"
                            )
                        ),
                        encoder_attention_mask=(
                            extra_tensor_data.get(
                                "encoder_attention_mask"
                            )
                        ),
                        context_latents=(
                            extra_tensor_data.get(
                                "context_latents"
                            )
                        ),
                        lyric_token_ids=(
                            extra_tensor_data.get(
                                "lyric_token_ids"
                            )
                        ),
                        vocal_language=(
                            vocal_language or "en"
                        ),
                        inference_steps=int(
                            inference_steps
                        ),
                        seed=42,
                    )
                )

                if align_result.get("success"):
                    composer_align_score = (
                        align_result.get(
                            "lm_score",
                            0.0,
                        )
                    )

                    dit_align_score = (
                        align_result.get(
                            "dit_score",
                            0.0,
                        )
                    )

                    alignment_report = (
                        f"  • Synapse Composer lyrics alignment "
                        f"score: {composer_align_score:.4f}\n"
                        f"  • Synapse Music lyrics alignment "
                        f"score: {dit_align_score:.4f}\n"
                        "\n(Measures how well lyrics timestamps "
                        "match audio energy using Cross-Attention)"
                    )

                else:
                    align_err = align_result.get(
                        "error",
                        "Unknown error",
                    )

                    alignment_report = (
                        "\n⚠️ Alignment Score Failed: "
                        f"{align_err}"
                    )

            except Exception as e:
                alignment_report = (
                    "\n⚠️ Alignment Score Error: "
                    f"{str(e)}"
                )

        if (
            has_audio_codes
            and llm_handler.llm_initialized
        ):
            if (
                global_score == 0.0
                and not scores_per_condition
            ):
                if (
                    alignment_report
                    and not alignment_report.startswith(
                        "\n⚠️"
                    )
                ):
                    final_output = (
                        "📊 Synapse Music Alignment Scores "
                        "(Composer codes unavailable):\n"
                    )

                    final_output += alignment_report

                    return final_output

                return t(
                    "messages.score_failed",
                    error=(
                        "PMI scoring returned no results"
                    ),
                )

            condition_lines = []

            for (
                condition_name,
                score_value,
            ) in sorted(
                scores_per_condition.items()
            ):
                condition_lines.append(
                    f"  • {condition_name}: "
                    f"{score_value:.4f}"
                )

            conditions_display = (
                "\n".join(condition_lines)
                if condition_lines
                else "  (no conditions)"
            )

            final_output = (
                f"✅ Synapse Quality Score: "
                f"{global_score:.4f} "
                f"(0-1, higher=better)\n\n"
                f"📊 Per-Condition Scores (0-1):\n"
                f"{conditions_display}\n"
            )

            if alignment_report:
                final_output += (
                    alignment_report + "\n"
                )

            final_output += (
                "Note: Metadata uses Top-k Recall; "
                "caption and lyrics use PMI."
            )

            return final_output

        if (
            alignment_report
            and not alignment_report.startswith(
                "\n⚠️"
            )
        ):
            final_output = (
                "📊 Synapse Music Alignment Scores "
                "(Composer codes unavailable for "
                "Cover/Repaint mode):\n"
            )

            final_output += alignment_report

            return final_output

        if alignment_report:
            return alignment_report

        return "⚠️ No Synapse scoring data available"

    except Exception as e:
        import traceback

        error_msg = (
            t(
                "messages.score_error",
                error=str(e),
            )
            + f"\n{traceback.format_exc()}"
        )

        return error_msg


def calculate_score_handler_with_selection(
    dit_handler,
    llm_handler,
    sample_idx,
    score_scale,
    current_batch_index,
    batch_queue,
):
    """
    Calculate Synapse Music quality score from historical batch data.

    All generation parameters are read directly from batch_queue so
    historical songs are scored using the settings that actually created
    them rather than the current UI state.
    """
    if current_batch_index not in batch_queue:
        return (
            gr.skip(),
            gr.skip(),
            batch_queue,
        )

    batch_data = batch_queue[
        current_batch_index
    ]

    params = batch_data.get(
        "generation_params",
        {},
    )

    caption = params.get(
        "captions",
        "",
    )

    lyrics = params.get(
        "lyrics",
        "",
    )

    bpm = params.get("bpm")

    key_scale = params.get(
        "key_scale",
        "",
    )

    time_signature = params.get(
        "time_signature",
        "",
    )

    audio_duration = params.get(
        "audio_duration",
        -1,
    )

    vocal_language = params.get(
        "vocal_language",
        "",
    )

    inference_steps = params.get(
        "inference_steps",
        8,
    )

    composer_metadata = batch_data.get(
        "composer_generated_metadata",
        batch_data.get(
            "lm_generated_metadata",
            None,
        ),
    )

    stored_codes = batch_data.get(
        "codes",
        "",
    )

    stored_allow_lm_batch = batch_data.get(
        "allow_lm_batch",
        False,
    )

    audio_codes_str = ""

    if (
        stored_allow_lm_batch
        and isinstance(stored_codes, list)
    ):
        if (
            0
            <= sample_idx - 1
            < len(stored_codes)
        ):
            code_item = stored_codes[
                sample_idx - 1
            ]

            audio_codes_str = (
                code_item
                if isinstance(code_item, str)
                else ""
            )

    else:
        audio_codes_str = (
            stored_codes
            if isinstance(stored_codes, str)
            else ""
        )

    extra_tensor_data = None

    extra_outputs = batch_data.get(
        "extra_outputs",
        {},
    )

    if extra_outputs and dit_handler:
        pred_latents = extra_outputs.get(
            "pred_latents"
        )

        if pred_latents is not None:
            sample_idx_0based = (
                sample_idx - 1
            )

            batch_size = (
                pred_latents.shape[0]
            )

            if (
                0
                <= sample_idx_0based
                < batch_size
            ):
                try:
                    encoder_hidden_states = (
                        extra_outputs.get(
                            "encoder_hidden_states"
                        )
                    )

                    encoder_attention_mask = (
                        extra_outputs.get(
                            "encoder_attention_mask"
                        )
                    )

                    context_latents = (
                        extra_outputs.get(
                            "context_latents"
                        )
                    )

                    lyric_token_idss = (
                        extra_outputs.get(
                            "lyric_token_idss"
                        )
                    )

                    extra_tensor_data = {
                        "pred_latent":
                            pred_latents[
                                sample_idx_0based:
                                sample_idx_0based + 1
                            ],

                        "encoder_hidden_states":
                            encoder_hidden_states[
                                sample_idx_0based:
                                sample_idx_0based + 1
                            ]
                            if encoder_hidden_states
                            is not None
                            else None,

                        "encoder_attention_mask":
                            encoder_attention_mask[
                                sample_idx_0based:
                                sample_idx_0based + 1
                            ]
                            if encoder_attention_mask
                            is not None
                            else None,

                        "context_latents":
                            context_latents[
                                sample_idx_0based:
                                sample_idx_0based + 1
                            ]
                            if context_latents
                            is not None
                            else None,

                        "lyric_token_ids":
                            lyric_token_idss[
                                sample_idx_0based:
                                sample_idx_0based + 1
                            ]
                            if lyric_token_idss
                            is not None
                            else None,
                    }

                    if any(
                        v is None
                        for v in extra_tensor_data.values()
                    ):
                        extra_tensor_data = None

                except Exception as e:
                    logger.warning(
                        "Error slicing Synapse tensor "
                        f"data for score: {e}"
                    )

                    extra_tensor_data = None

    score_display = calculate_score_handler(
        llm_handler,
        audio_codes_str,
        caption,
        lyrics,
        composer_metadata,
        bpm,
        key_scale,
        time_signature,
        audio_duration,
        vocal_language,
        score_scale,
        dit_handler,
        extra_tensor_data,
        inference_steps,
    )

    if current_batch_index in batch_queue:
        if (
            "scores"
            not in batch_queue[
                current_batch_index
            ]
        ):
            batch_queue[
                current_batch_index
            ]["scores"] = [""] * 8

        batch_queue[
            current_batch_index
        ]["scores"][sample_idx - 1] = (
            score_display
        )

    return (
        gr.update(
            value=score_display,
            visible=True,
        ),
        gr.skip(),
        batch_queue,
    )


def generate_lrc_handler(
    dit_handler,
    sample_idx,
    current_batch_index,
    batch_queue,
    vocal_language,
    inference_steps,
):
    """
    Generate LRC timestamps for a specific Synapse Music result.

    Cached inference data is loaded from the batch queue and passed to
    the Synapse Music lyrics-timestamp system.

    Only the LRC display is updated directly. The Audio component receives
    its subtitle file through the lrc_display.change() event.
    """
    import torch

    if current_batch_index not in batch_queue:
        return (
            gr.skip(),
            gr.skip(),
            batch_queue,
        )

    batch_data = batch_queue[
        current_batch_index
    ]

    extra_outputs = batch_data.get(
        "extra_outputs",
        {},
    )

    if not extra_outputs:
        return (
            gr.update(
                value=t(
                    "messages.lrc_no_extra_outputs"
                ),
                visible=True,
            ),
            gr.skip(),
            batch_queue,
        )

    pred_latents = extra_outputs.get(
        "pred_latents"
    )

    encoder_hidden_states = (
        extra_outputs.get(
            "encoder_hidden_states"
        )
    )

    encoder_attention_mask = (
        extra_outputs.get(
            "encoder_attention_mask"
        )
    )

    context_latents = extra_outputs.get(
        "context_latents"
    )

    lyric_token_idss = extra_outputs.get(
        "lyric_token_idss"
    )

    if any(
        x is None
        for x in [
            pred_latents,
            encoder_hidden_states,
            encoder_attention_mask,
            context_latents,
            lyric_token_idss,
        ]
    ):
        return (
            gr.update(
                value=t(
                    "messages.lrc_missing_tensors"
                ),
                visible=True,
            ),
            gr.skip(),
            batch_queue,
        )

    sample_idx_0based = sample_idx - 1

    batch_size = pred_latents.shape[0]

    if sample_idx_0based >= batch_size:
        return (
            gr.update(
                value=t(
                    "messages.lrc_sample_not_exist"
                ),
                visible=True,
            ),
            gr.skip(),
            batch_queue,
        )

    try:
        params = batch_data.get(
            "generation_params",
            {},
        )

        audio_duration = params.get(
            "audio_duration",
            -1,
        )

        if (
            audio_duration is None
            or audio_duration <= 0
        ):
            latent_length = (
                pred_latents.shape[1]
            )

            audio_duration = (
                latent_length / 25.0
            )

        sample_pred_latent = (
            pred_latents[
                sample_idx_0based:
                sample_idx_0based + 1
            ]
        )

        sample_encoder_hidden_states = (
            encoder_hidden_states[
                sample_idx_0based:
                sample_idx_0based + 1
            ]
        )

        sample_encoder_attention_mask = (
            encoder_attention_mask[
                sample_idx_0based:
                sample_idx_0based + 1
            ]
        )

        sample_context_latents = (
            context_latents[
                sample_idx_0based:
                sample_idx_0based + 1
            ]
        )

        sample_lyric_token_ids = (
            lyric_token_idss[
                sample_idx_0based:
                sample_idx_0based + 1
            ]
        )

        result = dit_handler.get_lyric_timestamp(
            pred_latent=sample_pred_latent,
            encoder_hidden_states=(
                sample_encoder_hidden_states
            ),
            encoder_attention_mask=(
                sample_encoder_attention_mask
            ),
            context_latents=(
                sample_context_latents
            ),
            lyric_token_ids=(
                sample_lyric_token_ids
            ),
            total_duration_seconds=float(
                audio_duration
            ),
            vocal_language=(
                vocal_language or "en"
            ),
            inference_steps=int(
                inference_steps
            ),
            seed=42,
        )

        if result.get("success"):
            lrc_text = result.get(
                "lrc_text",
                "",
            )

            if not lrc_text:
                return (
                    gr.update(
                        value=t(
                            "messages.lrc_empty_result"
                        ),
                        visible=True,
                    ),
                    gr.skip(),
                    batch_queue,
                )

            if (
                "lrcs"
                not in batch_queue[
                    current_batch_index
                ]
            ):
                batch_queue[
                    current_batch_index
                ]["lrcs"] = [""] * 8

            batch_queue[
                current_batch_index
            ]["lrcs"][
                sample_idx_0based
            ] = lrc_text

            vtt_path = lrc_to_vtt_file(
                lrc_text,
                total_duration=float(
                    audio_duration
                ),
            )

            if (
                "subtitles"
                not in batch_queue[
                    current_batch_index
                ]
            ):
                batch_queue[
                    current_batch_index
                ]["subtitles"] = [None] * 8

            batch_queue[
                current_batch_index
            ]["subtitles"][
                sample_idx_0based
            ] = vtt_path

            return (
                gr.update(
                    value=lrc_text,
                    visible=True,
                ),
                gr.skip(),
                batch_queue,
            )

        error_msg = result.get(
            "error",
            "Unknown error",
        )

        return (
            gr.update(
                value=f"❌ {error_msg}",
                visible=True,
            ),
            gr.skip(),
            batch_queue,
        )

    except Exception as e:
        logger.exception(
            "[generate_lrc_handler] "
            "Error generating Synapse Music LRC"
        )

        return (
            gr.update(
                value=f"❌ Error: {str(e)}",
                visible=True,
            ),
            gr.skip(),
            batch_queue,
        )


def update_audio_subtitles_from_lrc(
    lrc_text: str,
    audio_duration: float = None,
):
    """
    Update a Synapse Music Audio component's subtitles from LRC text.

    The LRC data is converted into a WebVTT file and passed to Gradio
    for native browser subtitle rendering.
    """
    if (
        not lrc_text
        or not lrc_text.strip()
    ):
        return gr.update(
            subtitles=None
        )

    vtt_path = lrc_to_vtt_file(
        lrc_text,
        total_duration=audio_duration,
    )

    return gr.update(
        subtitles=vtt_path
    )


def capture_current_params(
    captions,
    lyrics,
    bpm,
    key_scale,
    time_signature,
    vocal_language,
    inference_steps,
    guidance_scale,
    random_seed_checkbox,
    seed,
    reference_audio,
    audio_duration,
    batch_size_input,
    src_audio,
    text2music_audio_code_string,
    repainting_start,
    repainting_end,
    instruction_display_gen,
    audio_cover_strength,
    task_type,
    use_adg,
    cfg_interval_start,
    cfg_interval_end,
    shift,
    infer_method,
    custom_timesteps,
    audio_format,
    lm_temperature,
    think_checkbox,
    lm_cfg_scale,
    lm_top_k,
    lm_top_p,
    lm_negative_prompt,
    use_cot_metas,
    use_cot_caption,
    use_cot_language,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    auto_lrc,
    score_scale,
    lm_batch_chunk_size,
    track_name,
    complete_track_classes,
):
    """
    Capture current Synapse Music UI parameters for AutoGen.

    Audio codes are intentionally cleared so each new batch receives
    fresh Synapse Composer output or fresh Synapse Music seeds.
    """
    return {
        "captions": captions,
        "lyrics": lyrics,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "random_seed_checkbox": True,
        "seed": seed,
        "reference_audio": reference_audio,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,
        "src_audio": src_audio,
        "text2music_audio_code_string": "",
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "instruction_display_gen": instruction_display_gen,
        "audio_cover_strength": audio_cover_strength,
        "task_type": task_type,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "infer_method": infer_method,
        "custom_timesteps": custom_timesteps,
        "audio_format": audio_format,
        "lm_temperature": lm_temperature,
        "think_checkbox": think_checkbox,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,
        "constrained_decoding_debug": (
            constrained_decoding_debug
        ),
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "auto_lrc": auto_lrc,
        "score_scale": score_scale,
        "lm_batch_chunk_size": (
            lm_batch_chunk_size
        ),
        "track_name": track_name,
        "complete_track_classes": (
            complete_track_classes
        ),
    }


def generate_with_batch_management(
    dit_handler,
    llm_handler,
    captions,
    lyrics,
    bpm,
    key_scale,
    time_signature,
    vocal_language,
    inference_steps,
    guidance_scale,
    random_seed_checkbox,
    seed,
    reference_audio,
    audio_duration,
    batch_size_input,
    src_audio,
    text2music_audio_code_string,
    repainting_start,
    repainting_end,
    instruction_display_gen,
    audio_cover_strength,
    task_type,
    use_adg,
    cfg_interval_start,
    cfg_interval_end,
    shift,
    infer_method,
    custom_timesteps,
    audio_format,
    lm_temperature,
    think_checkbox,
    lm_cfg_scale,
    lm_top_k,
    lm_top_p,
    lm_negative_prompt,
    use_cot_metas,
    use_cot_caption,
    use_cot_language,
    is_format_caption,
    constrained_decoding_debug,
    allow_lm_batch,
    auto_score,
    auto_lrc,
    score_scale,
    lm_batch_chunk_size,
    track_name,
    complete_track_classes,
    autogen_checkbox,
    current_batch_index,
    total_batches,
    batch_queue,
    generation_params_state,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate Synapse Music audio with batch queue management.
    """
    generator = generate_with_progress(
        dit_handler,
        llm_handler,
        captions,
        lyrics,
        bpm,
        key_scale,
        time_signature,
        vocal_language,
        inference_steps,
        guidance_scale,
        random_seed_checkbox,
        seed,
        reference_audio,
        audio_duration,
        batch_size_input,
        src_audio,
        text2music_audio_code_string,
        repainting_start,
        repainting_end,
        instruction_display_gen,
        audio_cover_strength,
        task_type,
        use_adg,
        cfg_interval_start,
        cfg_interval_end,
        shift,
        infer_method,
        custom_timesteps,
        audio_format,
        lm_temperature,
        think_checkbox,
        lm_cfg_scale,
        lm_top_k,
        lm_top_p,
        lm_negative_prompt,
        use_cot_metas,
        use_cot_caption,
        use_cot_language,
        is_format_caption,
        constrained_decoding_debug,
        allow_lm_batch,
        auto_score,
        auto_lrc,
        score_scale,
        lm_batch_chunk_size,
        progress,
    )

    final_result_from_inner = None

    for partial_result in generator:
        final_result_from_inner = (
            partial_result
        )

        ui_result = (
            partial_result[:-2]
            if len(partial_result) > 47
            else (
                partial_result[:-1]
                if len(partial_result) > 46
                else partial_result
            )
        )

        yield ui_result + (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    result = final_result_from_inner

    if result is None:
        return

    all_audio_paths = result[8]

    if all_audio_paths is None:
        ui_result = (
            result[:-2]
            if len(result) > 47
            else (
                result[:-1]
                if len(result) > 46
                else result
            )
        )

        yield ui_result + (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

        return

    generation_info = result[9]
    seed_value_for_ui = result[11]

    composer_generated_metadata = (
        result[44]
    )

    raw_codes_list = (
        result[47]
        if len(result) > 47
        else [""] * 8
    )

    generated_codes_batch = (
        raw_codes_list
        if isinstance(raw_codes_list, list)
        else [""] * 8
    )

    generated_codes_single = (
        generated_codes_batch[0]
        if generated_codes_batch
        else ""
    )

    if (
        allow_lm_batch
        and batch_size_input >= 2
    ):
        codes_to_store = (
            generated_codes_batch[
                :int(batch_size_input)
            ]
        )
    else:
        codes_to_store = (
            generated_codes_single
        )

    saved_params = {
        "captions": captions,
        "lyrics": lyrics,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "vocal_language": vocal_language,
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "random_seed_checkbox": (
            random_seed_checkbox
        ),
        "seed": seed,
        "reference_audio": reference_audio,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,
        "src_audio": src_audio,
        "text2music_audio_code_string": (
            text2music_audio_code_string
        ),
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,
        "instruction_display_gen": (
            instruction_display_gen
        ),
        "audio_cover_strength": (
            audio_cover_strength
        ),
        "task_type": task_type,
        "use_adg": use_adg,
        "cfg_interval_start": (
            cfg_interval_start
        ),
        "cfg_interval_end": (
            cfg_interval_end
        ),
        "shift": shift,
        "infer_method": infer_method,
        "custom_timesteps": custom_timesteps,
        "audio_format": audio_format,
        "lm_temperature": lm_temperature,
        "think_checkbox": think_checkbox,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": (
            lm_negative_prompt
        ),
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": (
            use_cot_language
        ),
        "constrained_decoding_debug": (
            constrained_decoding_debug
        ),
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "auto_lrc": auto_lrc,
        "score_scale": score_scale,
        "lm_batch_chunk_size": (
            lm_batch_chunk_size
        ),
        "track_name": track_name,
        "complete_track_classes": (
            complete_track_classes
        ),
    }

    next_params = saved_params.copy()

    next_params[
        "text2music_audio_code_string"
    ] = ""

    next_params[
        "random_seed_checkbox"
    ] = True

    extra_outputs_from_result = (
        result[46]
        if (
            len(result) > 46
            and result[46] is not None
        )
        else {}
    )

    batch_queue = store_batch_in_queue(
        batch_queue,
        current_batch_index,
        all_audio_paths,
        generation_info,
        seed_value_for_ui,
        codes=codes_to_store,
        allow_lm_batch=allow_lm_batch,
        batch_size=int(batch_size_input),
        generation_params=saved_params,
        composer_generated_metadata=(
            composer_generated_metadata
        ),
        extra_outputs=(
            extra_outputs_from_result
        ),
        status="completed",
    )

    if (
        auto_lrc
        and extra_outputs_from_result
    ):
        lrcs_from_extra = (
            extra_outputs_from_result.get(
                "lrcs",
                [""] * 8,
            )
        )

        subtitles_from_extra = (
            extra_outputs_from_result.get(
                "subtitles",
                [None] * 8,
            )
        )

        batch_queue[
            current_batch_index
        ]["lrcs"] = lrcs_from_extra

        batch_queue[
            current_batch_index
        ]["subtitles"] = (
            subtitles_from_extra
        )

    total_batches = max(
        total_batches,
        current_batch_index + 1,
    )

    batch_indicator_text = (
        update_batch_indicator(
            current_batch_index,
            total_batches,
        )
    )

    (
        can_go_previous,
        can_go_next,
    ) = update_navigation_buttons(
        current_batch_index,
        total_batches,
    )

    next_batch_status_text = ""

    if autogen_checkbox:
        next_batch_status_text = t(
            "messages.autogen_enabled"
        )

    ui_result = (
        result[:-2]
        if len(result) > 47
        else (
            result[:-1]
            if len(result) > 46
            else result
        )
    )

    ui_result_list = list(ui_result)

    for i in range(8):
        ui_result_list[i] = gr.skip()

    ui_result = tuple(
        ui_result_list
    )

    yield ui_result + (
        current_batch_index,
        total_batches,
        batch_queue,
        next_params,
        batch_indicator_text,
        gr.update(
            interactive=can_go_previous
        ),
        gr.update(
            interactive=can_go_next
        ),
        next_batch_status_text,
        gr.update(interactive=True),
    )


def generate_next_batch_background(
    dit_handler,
    llm_handler,
    autogen_enabled,
    generation_params,
    current_batch_index,
    total_batches,
    batch_queue,
    is_format_caption,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate the next Synapse Music batch in the background when
    AutoGen is enabled.
    """
    if not autogen_enabled:
        return (
            batch_queue,
            total_batches,
            "",
            gr.update(interactive=False),
        )

    next_batch_idx = (
        current_batch_index + 1
    )

    if (
        next_batch_idx in batch_queue
        and batch_queue[
            next_batch_idx
        ].get("status") == "completed"
    ):
        return (
            batch_queue,
            total_batches,
            t(
                "messages.batch_ready",
                n=next_batch_idx + 1,
            ),
            gr.update(interactive=True),
        )

    total_batches = (
        next_batch_idx + 1
    )

    gr.Info(
        t(
            "messages.batch_generating",
            n=next_batch_idx + 1,
        )
    )

    params = generation_params.copy()

    logger.info(
        "========== SYNAPSE MUSIC BACKGROUND "
        f"GENERATION BATCH {next_batch_idx + 1} "
        "=========="
    )

    logger.info(
        "Parameters used for Synapse Music "
        "background generation:"
    )

    logger.info(
        f"  - captions: "
        f"{params.get('captions', 'N/A')}"
    )

    logger.info(
        (
            f"  - lyrics: "
            f"{params.get('lyrics', 'N/A')[:50]}..."
        )
        if params.get("lyrics")
        else "  - lyrics: N/A"
    )

    logger.info(
        f"  - bpm: {params.get('bpm')}"
    )

    logger.info(
        f"  - batch_size_input: "
        f"{params.get('batch_size_input')}"
    )

    logger.info(
        f"  - allow_lm_batch: "
        f"{params.get('allow_lm_batch')}"
    )

    logger.info(
        f"  - think_checkbox: "
        f"{params.get('think_checkbox')}"
    )

    logger.info(
        f"  - lm_temperature: "
        f"{params.get('lm_temperature')}"
    )

    logger.info(
        f"  - track_name: "
        f"{params.get('track_name')}"
    )

    logger.info(
        f"  - complete_track_classes: "
        f"{params.get('complete_track_classes')}"
    )

    logger.info(
        "  - text2music_audio_code_string: "
        + (
            "<CLEARED>"
            if params.get(
                "text2music_audio_code_string"
            ) == ""
            else "HAS_VALUE"
        )
    )

    logger.info(
        "==========================================="
    )

    try:
        params.setdefault(
            "captions",
            "",
        )

        params.setdefault(
            "lyrics",
            "",
        )

        params.setdefault(
            "bpm",
            None,
        )

        params.setdefault(
            "key_scale",
            "",
        )

        params.setdefault(
            "time_signature",
            "",
        )

        params.setdefault(
            "vocal_language",
            "unknown",
        )

        params.setdefault(
            "inference_steps",
            8,
        )

        params.setdefault(
            "guidance_scale",
            7.0,
        )

        params.setdefault(
            "random_seed_checkbox",
            True,
        )

        params.setdefault(
            "seed",
            "-1",
        )

        params.setdefault(
            "reference_audio",
            None,
        )

        params.setdefault(
            "audio_duration",
            -1,
        )

        params.setdefault(
            "batch_size_input",
            2,
        )

        params.setdefault(
            "src_audio",
            None,
        )

        params.setdefault(
            "text2music_audio_code_string",
            "",
        )

        params.setdefault(
            "repainting_start",
            0.0,
        )

        params.setdefault(
            "repainting_end",
            -1,
        )

        params.setdefault(
            "instruction_display_gen",
            "",
        )

        params.setdefault(
            "audio_cover_strength",
            1.0,
        )

        params.setdefault(
            "task_type",
            "text2music",
        )

        params.setdefault(
            "use_adg",
            False,
        )

        params.setdefault(
            "cfg_interval_start",
            0.0,
        )

        params.setdefault(
            "cfg_interval_end",
            1.0,
        )

        params.setdefault(
            "shift",
            1.0,
        )

        params.setdefault(
            "infer_method",
            "ode",
        )

        params.setdefault(
            "custom_timesteps",
            "",
        )

        params.setdefault(
            "audio_format",
            "mp3",
        )

        params.setdefault(
            "lm_temperature",
            0.85,
        )

        params.setdefault(
            "think_checkbox",
            True,
        )

        params.setdefault(
            "lm_cfg_scale",
            2.0,
        )

        params.setdefault(
            "lm_top_k",
            0,
        )

        params.setdefault(
            "lm_top_p",
            0.9,
        )

        params.setdefault(
            "lm_negative_prompt",
            "NO USER INPUT",
        )

        params.setdefault(
            "use_cot_metas",
            True,
        )

        params.setdefault(
            "use_cot_caption",
            True,
        )

        params.setdefault(
            "use_cot_language",
            True,
        )

        params.setdefault(
            "constrained_decoding_debug",
            False,
        )

        params.setdefault(
            "allow_lm_batch",
            True,
        )

        params.setdefault(
            "auto_score",
            False,
        )

        params.setdefault(
            "auto_lrc",
            False,
        )

        params.setdefault(
            "score_scale",
            0.5,
        )

        params.setdefault(
            "lm_batch_chunk_size",
            8,
        )

        params.setdefault(
            "track_name",
            None,
        )

        params.setdefault(
            "complete_track_classes",
            [],
        )

        generator = generate_with_progress(
            dit_handler,
            llm_handler,

            captions=params.get(
                "captions"
            ),

            lyrics=params.get(
                "lyrics"
            ),

            bpm=params.get(
                "bpm"
            ),

            key_scale=params.get(
                "key_scale"
            ),

            time_signature=params.get(
                "time_signature"
            ),

            vocal_language=params.get(
                "vocal_language"
            ),

            inference_steps=params.get(
                "inference_steps"
            ),

            guidance_scale=params.get(
                "guidance_scale"
            ),

            random_seed_checkbox=params.get(
                "random_seed_checkbox"
            ),

            seed=params.get(
                "seed"
            ),

            reference_audio=params.get(
                "reference_audio"
            ),

            audio_duration=params.get(
                "audio_duration"
            ),

            batch_size_input=params.get(
                "batch_size_input"
            ),

            src_audio=params.get(
                "src_audio"
            ),

            text2music_audio_code_string=params.get(
                "text2music_audio_code_string"
            ),

            repainting_start=params.get(
                "repainting_start"
            ),

            repainting_end=params.get(
                "repainting_end"
            ),

            instruction_display_gen=params.get(
                "instruction_display_gen"
            ),

            audio_cover_strength=params.get(
                "audio_cover_strength"
            ),

            task_type=params.get(
                "task_type"
            ),

            use_adg=params.get(
                "use_adg"
            ),

            cfg_interval_start=params.get(
                "cfg_interval_start"
            ),

            cfg_interval_end=params.get(
                "cfg_interval_end"
            ),

            shift=params.get(
                "shift"
            ),

            infer_method=params.get(
                "infer_method"
            ),

            custom_timesteps=params.get(
                "custom_timesteps"
            ),

            audio_format=params.get(
                "audio_format"
            ),

            lm_temperature=params.get(
                "lm_temperature"
            ),

            think_checkbox=params.get(
                "think_checkbox"
            ),

            lm_cfg_scale=params.get(
                "lm_cfg_scale"
            ),

            lm_top_k=params.get(
                "lm_top_k"
            ),

            lm_top_p=params.get(
                "lm_top_p"
            ),

            lm_negative_prompt=params.get(
                "lm_negative_prompt"
            ),

            use_cot_metas=params.get(
                "use_cot_metas"
            ),

            use_cot_caption=params.get(
                "use_cot_caption"
            ),

            use_cot_language=params.get(
                "use_cot_language"
            ),

            is_format_caption=is_format_caption,

            constrained_decoding_debug=params.get(
                "constrained_decoding_debug"
            ),

            allow_lm_batch=params.get(
                "allow_lm_batch"
            ),

            auto_score=params.get(
                "auto_score"
            ),

            auto_lrc=params.get(
                "auto_lrc"
            ),

            score_scale=params.get(
                "score_scale"
            ),

            lm_batch_chunk_size=params.get(
                "lm_batch_chunk_size"
            ),

            progress=progress,
        )

        final_result = None

        for partial_result in generator:
            final_result = partial_result

        if final_result is None:
            raise RuntimeError(
                "Synapse Music generation returned "
                "no result."
            )

        all_audio_paths = (
            final_result[8]
        )

        generation_info = (
            final_result[9]
        )

        seed_value_for_ui = (
            final_result[11]
        )

        composer_generated_metadata = (
            final_result[44]
        )

        raw_codes_list = (
            final_result[47]
            if len(final_result) > 47
            else [""] * 8
        )

        generated_codes_batch = (
            raw_codes_list
            if isinstance(
                raw_codes_list,
                list,
            )
            else [""] * 8
        )

        generated_codes_single = (
            generated_codes_batch[0]
            if generated_codes_batch
            else ""
        )

        extra_outputs_from_bg = (
            final_result[46]
            if (
                len(final_result) > 46
                and final_result[46]
                is not None
            )
            else {}
        )

        scores_from_bg = []

        for score_idx in range(
            12,
            20,
        ):
            if score_idx < len(
                final_result
            ):
                score_val = (
                    final_result[
                        score_idx
                    ]
                )

                if hasattr(
                    score_val,
                    "value",
                ):
                    scores_from_bg.append(
                        score_val.value
                        if score_val.value
                        else ""
                    )

                elif isinstance(
                    score_val,
                    str,
                ):
                    scores_from_bg.append(
                        score_val
                    )

                else:
                    scores_from_bg.append(
                        ""
                    )

            else:
                scores_from_bg.append(
                    ""
                )

        batch_size = params.get(
            "batch_size_input",
            2,
        )

        allow_lm_batch = params.get(
            "allow_lm_batch",
            False,
        )

        if (
            allow_lm_batch
            and batch_size >= 2
        ):
            codes_to_store = (
                generated_codes_batch[
                    :int(batch_size)
                ]
            )
        else:
            codes_to_store = (
                generated_codes_single
            )

        logger.info(
            "Synapse code extraction for "
            f"Batch {next_batch_idx + 1}:"
        )

        logger.info(
            f"  - allow_lm_batch: "
            f"{allow_lm_batch}"
        )

        logger.info(
            f"  - batch_size: "
            f"{batch_size}"
        )

        logger.info(
            "  - generated_codes_single exists: "
            f"{bool(generated_codes_single)}"
        )

        logger.info(
            "  - extra_outputs_from_bg exists: "
            f"{bool(extra_outputs_from_bg)}"
        )

        logger.info(
            f"  - scores_from_bg: "
            f"{[bool(s) for s in scores_from_bg]}"
        )

        if isinstance(
            codes_to_store,
            list,
        ):
            logger.info(
                "  - codes_to_store: LIST with "
                f"{len(codes_to_store)} items"
            )

            for idx, code in enumerate(
                codes_to_store
            ):
                logger.info(
                    f"    * Sample {idx + 1}: "
                    f"{len(code) if code else 0} chars"
                )

        else:
            logger.info(
                "  - codes_to_store: STRING with "
                f"{len(codes_to_store) if codes_to_store else 0} chars"
            )

        batch_queue = store_batch_in_queue(
            batch_queue,
            next_batch_idx,
            all_audio_paths,
            generation_info,
            seed_value_for_ui,
            codes=codes_to_store,
            scores=scores_from_bg,
            allow_lm_batch=allow_lm_batch,
            batch_size=int(batch_size),
            generation_params=params,
            composer_generated_metadata=(
                composer_generated_metadata
            ),
            extra_outputs=(
                extra_outputs_from_bg
            ),
            status="completed",
        )

        auto_lrc = params.get(
            "auto_lrc",
            False,
        )

        if (
            auto_lrc
            and extra_outputs_from_bg
        ):
            lrcs_from_extra = (
                extra_outputs_from_bg.get(
                    "lrcs",
                    [""] * 8,
                )
            )

            subtitles_from_extra = (
                extra_outputs_from_bg.get(
                    "subtitles",
                    [None] * 8,
                )
            )

            batch_queue[
                next_batch_idx
            ]["lrcs"] = lrcs_from_extra

            batch_queue[
                next_batch_idx
            ]["subtitles"] = (
                subtitles_from_extra
            )

            logger.info(
                "  - Synapse auto-LRC results stored: "
                f"{[bool(l) for l in lrcs_from_extra]}"
            )

        logger.info(
            f"Synapse Music Batch "
            f"{next_batch_idx + 1} "
            "stored successfully."
        )

        next_batch_status = t(
            "messages.batch_ready",
            n=next_batch_idx + 1,
        )

        return (
            batch_queue,
            total_batches,
            next_batch_status,
            gr.update(interactive=True),
        )

    except Exception as e:
        import traceback

        error_msg = t(
            "messages.batch_failed",
            error=str(e),
        )

        gr.Warning(error_msg)

        batch_queue[
            next_batch_idx
        ] = {
            "status": "error",
            "error": str(e),
            "traceback": (
                traceback.format_exc()
            ),
        }

        return (
            batch_queue,
            total_batches,
            error_msg,
            gr.update(interactive=False),
        )


def navigate_to_previous_batch(
    current_batch_index,
    batch_queue,
):
    """
    Navigate to the previous Synapse Music batch.

    Result view only; this does not modify the generation input UI.

    Uses a two-step update to prevent subtitle flickering.
    """
    if current_batch_index <= 0:
        gr.Warning(
            t("messages.at_first_batch")
        )

        yield tuple(
            [gr.update()] * 48
        )

        return

    new_batch_index = (
        current_batch_index - 1
    )

    if (
        new_batch_index
        not in batch_queue
    ):
        gr.Warning(
            t(
                "messages.batch_not_found",
                n=new_batch_index + 1,
            )
        )

        yield tuple(
            [gr.update()] * 48
        )

        return

    batch_data = batch_queue[
        new_batch_index
    ]

    audio_paths = batch_data.get(
        "audio_paths",
        [],
    )

    generation_info_text = (
        batch_data.get(
            "generation_info",
            "",
        )
    )

    real_audio_paths = [
        p
        for p in audio_paths
        if not p.lower().endswith(
            ".json"
        )
    ]

    audio_updates = []

    for idx in range(8):
        if idx < len(real_audio_paths):
            audio_path = (
                real_audio_paths[idx]
            )

            audio_updates.append(
                gr.update(
                    value=audio_path
                )
            )

        else:
            audio_updates.append(
                gr.update(
                    value=None
                )
            )

    total_batches = len(
        batch_queue
    )

    batch_indicator_text = (
        update_batch_indicator(
            new_batch_index,
            total_batches,
        )
    )

    (
        can_go_previous,
        can_go_next,
    ) = update_navigation_buttons(
        new_batch_index,
        total_batches,
    )

    stored_scores = batch_data.get(
        "scores",
        [""] * 8,
    )

    score_displays = (
        stored_scores
        if stored_scores
        else [""] * 8
    )

    stored_lrcs = batch_data.get(
        "lrcs",
        [""] * 8,
    )

    lrc_displays = (
        stored_lrcs
        if stored_lrcs
        else [""] * 8
    )

    stored_codes = batch_data.get(
        "codes",
        "",
    )

    stored_allow_lm_batch = (
        batch_data.get(
            "allow_lm_batch",
            False,
        )
    )

    batch_size = batch_data.get(
        "batch_size",
        2,
    )

    codes_display_updates = []
    lrc_display_updates = []
    lrc_clear_updates = []
    details_accordion_updates = []

    for i in range(8):
        if (
            stored_allow_lm_batch
            and isinstance(
                stored_codes,
                list,
            )
        ):
            code_str = (
                stored_codes[i]
                if i < len(stored_codes)
                else ""
            )

        else:
            code_str = (
                stored_codes
                if (
                    isinstance(
                        stored_codes,
                        str,
                    )
                    and i == 0
                )
                else ""
            )

        lrc_str = (
            lrc_displays[i]
            if i < len(lrc_displays)
            else ""
        )

        score_str = (
            score_displays[i]
            if i < len(score_displays)
            else ""
        )

        codes_display_updates.append(
            gr.update(
                value=code_str,
                visible=True,
            )
        )

        lrc_display_updates.append(
            gr.update(
                value=lrc_str,
                visible=True,
            )
        )

        lrc_clear_updates.append(
            gr.update(
                value="",
                visible=True,
            )
        )

        details_accordion_updates.append(
            gr.skip()
        )

    yield (
        audio_updates[0],
        audio_updates[1],
        audio_updates[2],
        audio_updates[3],
        audio_updates[4],
        audio_updates[5],
        audio_updates[6],
        audio_updates[7],

        audio_paths,
        generation_info_text,
        new_batch_index,
        batch_indicator_text,

        gr.update(
            interactive=can_go_previous
        ),

        gr.update(
            interactive=can_go_next
        ),

        t(
            "messages.viewing_batch",
            n=new_batch_index + 1,
        ),

        score_displays[0],
        score_displays[1],
        score_displays[2],
        score_displays[3],
        score_displays[4],
        score_displays[5],
        score_displays[6],
        score_displays[7],

        codes_display_updates[0],
        codes_display_updates[1],
        codes_display_updates[2],
        codes_display_updates[3],
        codes_display_updates[4],
        codes_display_updates[5],
        codes_display_updates[6],
        codes_display_updates[7],

        lrc_clear_updates[0],
        lrc_clear_updates[1],
        lrc_clear_updates[2],
        lrc_clear_updates[3],
        lrc_clear_updates[4],
        lrc_clear_updates[5],
        lrc_clear_updates[6],
        lrc_clear_updates[7],

        details_accordion_updates[0],
        details_accordion_updates[1],
        details_accordion_updates[2],
        details_accordion_updates[3],
        details_accordion_updates[4],
        details_accordion_updates[5],
        details_accordion_updates[6],
        details_accordion_updates[7],

        gr.update(interactive=True),
    )

    time_module.sleep(0.05)

    skip_audio = [
        gr.skip()
        for _ in range(8)
    ]

    skip_scores = [
        gr.skip()
        for _ in range(8)
    ]

    skip_codes = [
        gr.skip()
        for _ in range(8)
    ]

    skip_accordions = [
        gr.skip()
        for _ in range(8)
    ]

    yield (
        skip_audio[0],
        skip_audio[1],
        skip_audio[2],
        skip_audio[3],
        skip_audio[4],
        skip_audio[5],
        skip_audio[6],
        skip_audio[7],

        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),

        gr.skip(),
        gr.skip(),

        gr.skip(),

        skip_scores[0],
        skip_scores[1],
        skip_scores[2],
        skip_scores[3],
        skip_scores[4],
        skip_scores[5],
        skip_scores[6],
        skip_scores[7],

        skip_codes[0],
        skip_codes[1],
        skip_codes[2],
        skip_codes[3],
        skip_codes[4],
        skip_codes[5],
        skip_codes[6],
        skip_codes[7],

        lrc_display_updates[0],
        lrc_display_updates[1],
        lrc_display_updates[2],
        lrc_display_updates[3],
        lrc_display_updates[4],
        lrc_display_updates[5],
        lrc_display_updates[6],
        lrc_display_updates[7],

        skip_accordions[0],
        skip_accordions[1],
        skip_accordions[2],
        skip_accordions[3],
        skip_accordions[4],
        skip_accordions[5],
        skip_accordions[6],
        skip_accordions[7],

        gr.skip(),
    )


def navigate_to_next_batch(
    autogen_enabled,
    current_batch_index,
    total_batches,
    batch_queue,
):
    """
    Navigate to the next Synapse Music batch.

    Result view only; this does not modify the generation input UI.

    Uses a two-step update to prevent subtitle flickering.
    """
    if (
        current_batch_index
        >= total_batches - 1
    ):
        gr.Warning(
            t("messages.at_last_batch")
        )

        yield tuple(
            [gr.update()] * 49
        )

        return

    new_batch_index = (
        current_batch_index + 1
    )

    if (
        new_batch_index
        not in batch_queue
    ):
        gr.Warning(
            t(
                "messages.batch_not_found",
                n=new_batch_index + 1,
            )
        )

        yield tuple(
            [gr.update()] * 49
        )

        return

    batch_data = batch_queue[
        new_batch_index
    ]

    audio_paths = batch_data.get(
        "audio_paths",
        [],
    )

    generation_info_text = (
        batch_data.get(
            "generation_info",
            "",
        )
    )

    real_audio_paths = [
        p
        for p in audio_paths
        if not p.lower().endswith(
            ".json"
        )
    ]

    audio_updates = []

    for idx in range(8):
        if idx < len(real_audio_paths):
            audio_path = (
                real_audio_paths[idx]
            )

            audio_updates.append(
                gr.update(
                    value=audio_path
                )
            )

        else:
            audio_updates.append(
                gr.update(
                    value=None
                )
            )

    batch_indicator_text = (
        update_batch_indicator(
            new_batch_index,
            total_batches,
        )
    )

    (
        can_go_previous,
        can_go_next,
    ) = update_navigation_buttons(
        new_batch_index,
        total_batches,
    )

    next_batch_status_text = ""

    is_latest_view = (
        new_batch_index
        == total_batches - 1
    )

    if (
        autogen_enabled
        and is_latest_view
    ):
        next_batch_status_text = (
            "🔄 Synapse AutoGen will generate "
            "the next batch in the background..."
        )

    stored_scores = batch_data.get(
        "scores",
        [""] * 8,
    )

    score_displays = (
        stored_scores
        if stored_scores
        else [""] * 8
    )

    stored_lrcs = batch_data.get(
        "lrcs",
        [""] * 8,
    )

    lrc_displays = (
        stored_lrcs
        if stored_lrcs
        else [""] * 8
    )

    stored_codes = batch_data.get(
        "codes",
        "",
    )

    stored_allow_lm_batch = (
        batch_data.get(
            "allow_lm_batch",
            False,
        )
    )

    batch_size = batch_data.get(
        "batch_size",
        2,
    )

    codes_display_updates = []
    lrc_display_updates = []
    lrc_clear_updates = []
    details_accordion_updates = []

    for i in range(8):
        if (
            stored_allow_lm_batch
            and isinstance(
                stored_codes,
                list,
            )
        ):
            code_str = (
                stored_codes[i]
                if i < len(stored_codes)
                else ""
            )

        else:
            code_str = (
                stored_codes
                if (
                    isinstance(
                        stored_codes,
                        str,
                    )
                    and i == 0
                )
                else ""
            )

        lrc_str = (
            lrc_displays[i]
            if i < len(lrc_displays)
            else ""
        )

        codes_display_updates.append(
            gr.update(
                value=code_str,
                visible=True,
            )
        )

        lrc_display_updates.append(
            gr.update(
                value=lrc_str,
                visible=True,
            )
        )

        lrc_clear_updates.append(
            gr.update(
                value="",
                visible=True,
            )
        )

        details_accordion_updates.append(
            gr.skip()
        )

    yield (
        audio_updates[0],
        audio_updates[1],
        audio_updates[2],
        audio_updates[3],
        audio_updates[4],
        audio_updates[5],
        audio_updates[6],
        audio_updates[7],

        audio_paths,
        generation_info_text,
        new_batch_index,
        batch_indicator_text,

        gr.update(
            interactive=can_go_previous
        ),

        gr.update(
            interactive=can_go_next
        ),

        t(
            "messages.viewing_batch",
            n=new_batch_index + 1,
        ),

        next_batch_status_text,

        score_displays[0],
        score_displays[1],
        score_displays[2],
        score_displays[3],
        score_displays[4],
        score_displays[5],
        score_displays[6],
        score_displays[7],

        codes_display_updates[0],
        codes_display_updates[1],
        codes_display_updates[2],
        codes_display_updates[3],
        codes_display_updates[4],
        codes_display_updates[5],
        codes_display_updates[6],
        codes_display_updates[7],

        lrc_clear_updates[0],
        lrc_clear_updates[1],
        lrc_clear_updates[2],
        lrc_clear_updates[3],
        lrc_clear_updates[4],
        lrc_clear_updates[5],
        lrc_clear_updates[6],
        lrc_clear_updates[7],

        details_accordion_updates[0],
        details_accordion_updates[1],
        details_accordion_updates[2],
        details_accordion_updates[3],
        details_accordion_updates[4],
        details_accordion_updates[5],
        details_accordion_updates[6],
        details_accordion_updates[7],

        gr.update(interactive=True),
    )

    time_module.sleep(0.05)

    skip_audio = [
        gr.skip()
        for _ in range(8)
    ]

    skip_scores = [
        gr.skip()
        for _ in range(8)
    ]

    skip_codes = [
        gr.skip()
        for _ in range(8)
    ]

    skip_accordions = [
        gr.skip()
        for _ in range(8)
    ]

    yield (
        skip_audio[0],
        skip_audio[1],
        skip_audio[2],
        skip_audio[3],
        skip_audio[4],
        skip_audio[5],
        skip_audio[6],
        skip_audio[7],

        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),

        gr.skip(),
        gr.skip(),

        gr.skip(),
        gr.skip(),

        skip_scores[0],
        skip_scores[1],
        skip_scores[2],
        skip_scores[3],
        skip_scores[4],
        skip_scores[5],
        skip_scores[6],
        skip_scores[7],

        skip_codes[0],
        skip_codes[1],
        skip_codes[2],
        skip_codes[3],
        skip_codes[4],
        skip_codes[5],
        skip_codes[6],
        skip_codes[7],

        lrc_display_updates[0],
        lrc_display_updates[1],
        lrc_display_updates[2],
        lrc_display_updates[3],
        lrc_display_updates[4],
        lrc_display_updates[5],
        lrc_display_updates[6],
        lrc_display_updates[7],

        skip_accordions[0],
        skip_accordions[1],
        skip_accordions[2],
        skip_accordions[3],
        skip_accordions[4],
        skip_accordions[5],
        skip_accordions[6],
        skip_accordions[7],

        gr.skip(),
    )


def restore_batch_parameters(
    current_batch_index,
    batch_queue,
):
    """
    Restore parameters from the currently viewed Synapse Music batch
    back into the generation UI.

    This allows historical songs to be reused as starting points for
    new generations.
    """
    if (
        current_batch_index
        not in batch_queue
    ):
        gr.Warning(
            t("messages.no_batch_data")
        )

        return [
            gr.update()
        ] * 20

    batch_data = batch_queue[
        current_batch_index
    ]

    params = batch_data.get(
        "generation_params",
        {},
    )

    captions = params.get(
        "captions",
        "",
    )

    lyrics = params.get(
        "lyrics",
        "",
    )

    bpm = params.get(
        "bpm",
        None,
    )

    key_scale = params.get(
        "key_scale",
        "",
    )

    time_signature = params.get(
        "time_signature",
        "",
    )

    vocal_language = params.get(
        "vocal_language",
        "unknown",
    )

    audio_duration = params.get(
        "audio_duration",
        -1,
    )

    batch_size_input = params.get(
        "batch_size_input",
        2,
    )

    inference_steps = params.get(
        "inference_steps",
        8,
    )

    lm_temperature = params.get(
        "lm_temperature",
        0.85,
    )

    lm_cfg_scale = params.get(
        "lm_cfg_scale",
        2.0,
    )

    lm_top_k = params.get(
        "lm_top_k",
        0,
    )

    lm_top_p = params.get(
        "lm_top_p",
        0.9,
    )

    think_checkbox = params.get(
        "think_checkbox",
        True,
    )

    use_cot_caption = params.get(
        "use_cot_caption",
        True,
    )

    use_cot_language = params.get(
        "use_cot_language",
        True,
    )

    allow_lm_batch = params.get(
        "allow_lm_batch",
        True,
    )

    track_name = params.get(
        "track_name",
        None,
    )

    complete_track_classes = params.get(
        "complete_track_classes",
        [],
    )

    stored_codes = batch_data.get(
        "codes",
        "",
    )

    if stored_codes:
        if isinstance(
            stored_codes,
            list,
        ):
            codes_main = (
                stored_codes[0]
                if stored_codes
                else ""
            )

        else:
            codes_main = stored_codes

    else:
        codes_main = ""

    gr.Info(
        t(
            "messages.params_restored",
            n=current_batch_index + 1,
        )
    )

    return (
        codes_main,
        captions,
        lyrics,
        bpm,
        key_scale,
        time_signature,
        vocal_language,
        audio_duration,
        batch_size_input,
        inference_steps,
        lm_temperature,
        lm_cfg_scale,
        lm_top_k,
        lm_top_p,
        think_checkbox,
        use_cot_caption,
        use_cot_language,
        allow_lm_batch,
        track_name,
        complete_track_classes,
    )