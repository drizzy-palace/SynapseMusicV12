"""
Synapse Music V12 - Generation Input Handlers

Contains event handlers and helper functions related to Synapse Music V12
generation inputs, model configuration, Synapse Composer metadata generation,
audio-code processing, and generation mode management.
"""

import os
import json
import random
import glob
from typing import Optional, List, Tuple

import gradio as gr
from loguru import logger

from synapse.constants import (
    TASK_TYPES_TURBO,
    TASK_TYPES_BASE,
)
from synapse.gradio_ui.i18n import t
from synapse.inference import (
    understand_music,
    create_sample,
    format_sample,
)


# Hugging Face Space environment detection for ZeroGPU support
IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None


def _get_spaces_gpu_decorator(duration=120):
    """
    Get the @spaces.GPU decorator when running in a Hugging Face Space.

    Returns an identity decorator when not running inside a Space or when
    the optional spaces package is unavailable.
    """
    if IS_HUGGINGFACE_SPACE:
        try:
            import spaces

            return spaces.GPU(duration=duration)
        except ImportError:
            logger.warning(
                "spaces package not found, GPU decorator disabled"
            )
            return lambda func: func

    return lambda func: func


def parse_and_validate_timesteps(
    timesteps_str: str,
    inference_steps: int,
) -> Tuple[Optional[List[float]], bool, str]:
    """
    Parse and validate a custom timestep sequence.

    Args:
        timesteps_str:
            Comma-separated timestep values, for example:
            "0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0"

        inference_steps:
            Expected number of inference steps.

    Returns:
        Tuple of:
        - parsed_timesteps: List of float timesteps, or None if invalid/empty
        - has_warning: Whether a warning was produced
        - warning_message: Description of the warning
    """
    if not timesteps_str or not timesteps_str.strip():
        return None, False, ""

    # Parse comma-separated values.
    values = [
        value.strip()
        for value in timesteps_str.split(",")
        if value.strip()
    ]

    if not values:
        return None, False, ""

    # Ensure the schedule has a trailing zero.
    if values[-1] != "0":
        values.append("0")

    try:
        timesteps = [
            float(value)
            for value in values
        ]
    except ValueError:
        gr.Warning(
            t("messages.invalid_timesteps_format")
        )
        return None, True, "Invalid format"

    # Validate range [0, 1].
    if any(
        timestep < 0 or timestep > 1
        for timestep in timesteps
    ):
        gr.Warning(
            t("messages.timesteps_out_of_range")
        )
        return None, True, "Out of range"

    # Check whether the schedule matches the requested inference step count.
    actual_steps = len(timesteps) - 1

    if actual_steps != inference_steps:
        gr.Warning(
            t(
                "messages.timesteps_count_mismatch",
                actual=actual_steps,
                expected=inference_steps,
            )
        )

        return (
            timesteps,
            True,
            f"Using {actual_steps} steps from timesteps",
        )

    return timesteps, False, ""


def load_metadata(file_obj):
    """
    Load Synapse Music V12 generation parameters from a JSON metadata file.
    """
    if file_obj is None:
        gr.Warning(
            t("messages.no_file_selected")
        )

        # Return None for all fields and False for is_format_caption.
        return [None] * 36 + [False]

    try:
        # Resolve uploaded file path.
        if hasattr(file_obj, "name"):
            filepath = file_obj.name
        else:
            filepath = file_obj

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        # ---------------------------------------------------------------------
        # Core generation metadata
        # ---------------------------------------------------------------------

        task_type = metadata.get(
            "task_type",
            "text2music",
        )

        captions = metadata.get(
            "caption",
            "",
        )

        lyrics = metadata.get(
            "lyrics",
            "",
        )

        vocal_language = metadata.get(
            "vocal_language",
            "unknown",
        )

        # ---------------------------------------------------------------------
        # BPM
        # ---------------------------------------------------------------------

        bpm_value = metadata.get("bpm")

        if (
            bpm_value is not None
            and bpm_value != "N/A"
        ):
            try:
                bpm = (
                    int(bpm_value)
                    if bpm_value
                    else None
                )
            except (ValueError, TypeError):
                bpm = None
        else:
            bpm = None

        key_scale = metadata.get(
            "keyscale",
            "",
        )

        time_signature = metadata.get(
            "timesignature",
            "",
        )

        # ---------------------------------------------------------------------
        # Duration
        # ---------------------------------------------------------------------

        duration_value = metadata.get(
            "duration",
            -1,
        )

        if (
            duration_value is not None
            and duration_value != "N/A"
        ):
            try:
                audio_duration = float(
                    duration_value
                )
            except (ValueError, TypeError):
                audio_duration = -1
        else:
            audio_duration = -1

        # ---------------------------------------------------------------------
        # Generation configuration
        # ---------------------------------------------------------------------

        batch_size = metadata.get(
            "batch_size",
            2,
        )

        inference_steps = metadata.get(
            "inference_steps",
            8,
        )

        guidance_scale = metadata.get(
            "guidance_scale",
            7.0,
        )

        seed = metadata.get(
            "seed",
            "-1",
        )

        # Always disable random seed when loading metadata so the saved seed
        # can be reproduced.
        random_seed = False

        use_adg = metadata.get(
            "use_adg",
            False,
        )

        cfg_interval_start = metadata.get(
            "cfg_interval_start",
            0.0,
        )

        cfg_interval_end = metadata.get(
            "cfg_interval_end",
            1.0,
        )

        shift = metadata.get(
            "shift",
            3.0,
        )

        infer_method = metadata.get(
            "infer_method",
            "ode",
        )

        custom_timesteps = metadata.get(
            "timesteps",
            "",
        )

        if custom_timesteps is None:
            custom_timesteps = ""

        audio_format = metadata.get(
            "audio_format",
            "mp3",
        )

        # ---------------------------------------------------------------------
        # Synapse Composer configuration
        # ---------------------------------------------------------------------

        lm_temperature = metadata.get(
            "lm_temperature",
            0.85,
        )

        lm_cfg_scale = metadata.get(
            "lm_cfg_scale",
            2.0,
        )

        lm_top_k = metadata.get(
            "lm_top_k",
            0,
        )

        lm_top_p = metadata.get(
            "lm_top_p",
            0.9,
        )

        lm_negative_prompt = metadata.get(
            "lm_negative_prompt",
            "NO USER INPUT",
        )

        use_cot_metas = metadata.get(
            "use_cot_metas",
            True,
        )

        use_cot_caption = metadata.get(
            "use_cot_caption",
            True,
        )

        use_cot_language = metadata.get(
            "use_cot_language",
            True,
        )

        # ---------------------------------------------------------------------
        # Audio transformation settings
        # ---------------------------------------------------------------------

        audio_cover_strength = metadata.get(
            "audio_cover_strength",
            1.0,
        )

        think = metadata.get(
            "thinking",
            True,
        )

        audio_codes = metadata.get(
            "audio_codes",
            "",
        )

        repainting_start = metadata.get(
            "repainting_start",
            0.0,
        )

        repainting_end = metadata.get(
            "repainting_end",
            -1,
        )

        track_name = metadata.get(
            "track_name"
        )

        complete_track_classes = metadata.get(
            "complete_track_classes",
            [],
        )

        instrumental = metadata.get(
            "instrumental",
            False,
        )

        gr.Info(
            t(
                "messages.params_loaded",
                filename=os.path.basename(filepath),
            )
        )

        return (
            task_type,
            captions,
            lyrics,
            vocal_language,
            bpm,
            key_scale,
            time_signature,
            audio_duration,
            batch_size,
            inference_steps,
            guidance_scale,
            seed,
            random_seed,
            use_adg,
            cfg_interval_start,
            cfg_interval_end,
            shift,
            infer_method,
            custom_timesteps,
            audio_format,
            lm_temperature,
            lm_cfg_scale,
            lm_top_k,
            lm_top_p,
            lm_negative_prompt,
            use_cot_metas,
            use_cot_caption,
            use_cot_language,
            audio_cover_strength,
            think,
            audio_codes,
            repainting_start,
            repainting_end,
            track_name,
            complete_track_classes,
            instrumental,
            True,
        )

    except json.JSONDecodeError as error:
        gr.Warning(
            t(
                "messages.invalid_json",
                error=str(error),
            )
        )

        return [None] * 36 + [False]

    except Exception as error:
        gr.Warning(
            t(
                "messages.load_error",
                error=str(error),
            )
        )

        return [None] * 36 + [False]


def load_random_example(task_type: str):
    """
    Load a random Synapse generation example from the task-specific
    examples directory.

    Args:
        task_type:
            The generation task type, for example "text2music".

    Returns:
        Tuple containing:
        caption,
        lyrics,
        think,
        bpm,
        duration,
        keyscale,
        language,
        timesignature
    """
    try:
        current_file = os.path.abspath(
            __file__
        )

        # This module lives in:
        # synapse/gradio_ui/events/
        #
        # Move four levels upward to reach the project root.
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        current_file
                    )
                )
            )
        )

        examples_dir = os.path.join(
            project_root,
            "examples",
            task_type,
        )

        if not os.path.exists(
            examples_dir
        ):
            gr.Warning(
                f"Examples directory not found: "
                f"examples/{task_type}/"
            )

            return (
                "",
                "",
                True,
                None,
                None,
                "",
                "",
                "",
            )

        json_files = glob.glob(
            os.path.join(
                examples_dir,
                "*.json",
            )
        )

        if not json_files:
            gr.Warning(
                f"No JSON files found in "
                f"examples/{task_type}/"
            )

            return (
                "",
                "",
                True,
                None,
                None,
                "",
                "",
                "",
            )

        selected_file = random.choice(
            json_files
        )

        try:
            with open(
                selected_file,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            # Prefer caption and fall back to prompt.
            caption_value = data.get(
                "caption",
                data.get(
                    "prompt",
                    "",
                ),
            )

            if not isinstance(
                caption_value,
                str,
            ):
                caption_value = (
                    str(caption_value)
                    if caption_value
                    else ""
                )

            lyrics_value = data.get(
                "lyrics",
                "",
            )

            if not isinstance(
                lyrics_value,
                str,
            ):
                lyrics_value = (
                    str(lyrics_value)
                    if lyrics_value
                    else ""
                )

            # Thinking defaults to enabled.
            think_value = data.get(
                "think",
                True,
            )

            if not isinstance(
                think_value,
                bool,
            ):
                think_value = True

            # -----------------------------------------------------------------
            # Optional metadata
            # -----------------------------------------------------------------

            bpm_value = None

            if (
                "bpm" in data
                and data["bpm"] not in [
                    None,
                    "N/A",
                    "",
                ]
            ):
                try:
                    bpm_value = int(
                        data["bpm"]
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    pass

            duration_value = None

            if (
                "duration" in data
                and data["duration"] not in [
                    None,
                    "N/A",
                    "",
                ]
            ):
                try:
                    duration_value = float(
                        data["duration"]
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    pass

            keyscale_value = data.get(
                "keyscale",
                "",
            )

            if keyscale_value in [
                None,
                "N/A",
            ]:
                keyscale_value = ""

            language_value = data.get(
                "language",
                "",
            )

            if language_value in [
                None,
                "N/A",
            ]:
                language_value = ""

            timesignature_value = data.get(
                "timesignature",
                "",
            )

            if timesignature_value in [
                None,
                "N/A",
            ]:
                timesignature_value = ""

            gr.Info(
                t(
                    "messages.example_loaded",
                    filename=os.path.basename(
                        selected_file
                    ),
                )
            )

            return (
                caption_value,
                lyrics_value,
                think_value,
                bpm_value,
                duration_value,
                keyscale_value,
                language_value,
                timesignature_value,
            )

        except json.JSONDecodeError as error:
            gr.Warning(
                t(
                    "messages.example_failed",
                    filename=os.path.basename(
                        selected_file
                    ),
                    error=str(error),
                )
            )

            return (
                "",
                "",
                True,
                None,
                None,
                "",
                "",
                "",
            )

        except Exception as error:
            gr.Warning(
                t(
                    "messages.example_error",
                    error=str(error),
                )
            )

            return (
                "",
                "",
                True,
                None,
                None,
                "",
                "",
                "",
            )

    except Exception as error:
        gr.Warning(
            t(
                "messages.example_error",
                error=str(error),
            )
        )

        return (
            "",
            "",
            True,
            None,
            None,
            "",
            "",
            "",
        )


def sample_example_smart(
    llm_handler,
    task_type: str,
    constrained_decoding_debug: bool = False,
):
    """
    Smart Synapse example generator.

    Uses Synapse Composer when initialized. If Synapse Composer is unavailable
    or generation fails, falls back to examples stored in the local examples
    directory.

    This is a Gradio wrapper around the understand_music API from
    synapse.inference.

    Args:
        llm_handler:
            Synapse Composer / language-model handler instance.

        task_type:
            Generation task type, for example "text2music".

        constrained_decoding_debug:
            Whether constrained-decoding debug logging should be enabled.

    Returns:
        Tuple containing:
        caption,
        lyrics,
        think,
        bpm,
        duration,
        keyscale,
        language,
        timesignature
    """

    if llm_handler.llm_initialized:
        try:
            result = understand_music(
                llm_handler=llm_handler,
                audio_codes="NO USER INPUT",
                temperature=0.85,
                use_constrained_decoding=True,
                constrained_decoding_debug=(
                    constrained_decoding_debug
                ),
            )

            if result.success:
                gr.Info(
                    t("messages.lm_generated")
                )

                return (
                    result.caption,
                    result.lyrics,
                    True,
                    result.bpm,
                    result.duration,
                    result.keyscale,
                    result.language,
                    result.timesignature,
                )

            gr.Warning(
                t("messages.lm_fallback")
            )

            return load_random_example(
                task_type
            )

        except Exception:
            gr.Warning(
                t("messages.lm_fallback")
            )

            return load_random_example(
                task_type
            )

    return load_random_example(
        task_type
    )


def load_random_simple_description():
    """
    Load a random music description from the Simple Mode examples directory.

    Returns:
        Tuple of:
        - description
        - instrumental
        - vocal_language
    """
    try:
        current_file = os.path.abspath(
            __file__
        )

        # This module lives in:
        # synapse/gradio_ui/events/
        #
        # Move four levels upward to reach the project root.
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        current_file
                    )
                )
            )
        )

        examples_dir = os.path.join(
            project_root,
            "examples",
            "simple_mode",
        )

        if not os.path.exists(
            examples_dir
        ):
            gr.Warning(
                t(
                    "messages.simple_examples_not_found"
                )
            )

            return (
                gr.update(),
                gr.update(),
                gr.update(),
            )

        json_files = glob.glob(
            os.path.join(
                examples_dir,
                "*.json",
            )
        )

        if not json_files:
            gr.Warning(
                t(
                    "messages.simple_examples_empty"
                )
            )

            return (
                gr.update(),
                gr.update(),
                gr.update(),
            )

        selected_file = random.choice(
            json_files
        )

        try:
            with open(
                selected_file,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            description = data.get(
                "description",
                "",
            )

            instrumental = data.get(
                "instrumental",
                False,
            )

            vocal_language = data.get(
                "vocal_language",
                "unknown",
            )

            if isinstance(
                vocal_language,
                list,
            ):
                vocal_language = (
                    vocal_language[0]
                    if vocal_language
                    else "unknown"
                )

            gr.Info(
                t(
                    "messages.simple_example_loaded",
                    filename=os.path.basename(
                        selected_file
                    ),
                )
            )

            return (
                description,
                instrumental,
                vocal_language,
            )

        except json.JSONDecodeError as error:
            gr.Warning(
                t(
                    "messages.example_failed",
                    filename=os.path.basename(
                        selected_file
                    ),
                    error=str(error),
                )
            )

            return (
                gr.update(),
                gr.update(),
                gr.update(),
            )

        except Exception as error:
            gr.Warning(
                t(
                    "messages.example_error",
                    error=str(error),
                )
            )

            return (
                gr.update(),
                gr.update(),
                gr.update(),
            )

    except Exception as error:
        gr.Warning(
            t(
                "messages.example_error",
                error=str(error),
            )
        )

        return (
            gr.update(),
            gr.update(),
            gr.update(),
        )


def refresh_checkpoints(dit_handler):
    """
    Refresh available Synapse Music checkpoints.
    """
    choices = (
        dit_handler.get_available_checkpoints()
    )

    return gr.update(
        choices=choices
    )


def update_model_type_settings(
    config_path,
):
    """
    Update Synapse generation settings based on model type.

    This acts as a fallback before the model handler has been initialized.
    Once initialized, the actual model type is determined by the handler's
    is_turbo_model() method.
    """
    if config_path is None:
        config_path = ""

    config_path_lower = (
        config_path.lower()
    )

    # Heuristic fallback based on model configuration name.
    if "turbo" in config_path_lower:
        is_turbo = True

    elif "base" in config_path_lower:
        is_turbo = False

    else:
        # Synapse defaults to Turbo behavior for unknown model variants.
        is_turbo = True

    return get_model_type_ui_settings(
        is_turbo
    )


def init_service_wrapper(
    dit_handler,
    llm_handler,
    checkpoint,
    config_path,
    device,
    init_llm,
    lm_model_path,
    backend,
    use_flash_attention,
    offload_to_cpu,
    offload_dit_to_cpu,
):
    """
    Initialize the Synapse Music generation service and optionally initialize
    Synapse Composer.

    Returns:
        - initialization status
        - Generate button state
        - service configuration accordion state
        - model-specific UI settings
    """

    # -------------------------------------------------------------------------
    # Initialize Synapse audio generation model
    # -------------------------------------------------------------------------

    status, enable = (
        dit_handler.initialize_service(
            checkpoint,
            config_path,
            device,
            use_flash_attention=(
                use_flash_attention
            ),
            compile_model=False,
            offload_to_cpu=(
                offload_to_cpu
            ),
            offload_dit_to_cpu=(
                offload_dit_to_cpu
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Initialize Synapse Composer
    # -------------------------------------------------------------------------

    if init_llm:
        current_file = os.path.abspath(
            __file__
        )

        # This module lives in:
        # synapse/gradio_ui/events/
        #
        # Move four levels upward to reach the project root.
        project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        current_file
                    )
                )
            )
        )

        checkpoint_dir = os.path.join(
            project_root,
            "checkpoints",
        )

        lm_status, lm_success = (
            llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend=backend,
                device=device,
                offload_to_cpu=(
                    offload_to_cpu
                ),
                dtype=dit_handler.dtype,
            )
        )

        status += f"\n{lm_status}"

        if not lm_success:
            logger.warning(
                "Synapse Composer initialization failed. "
                "Audio generation model remains available if its "
                "initialization succeeded."
            )

    # Collapse service configuration after successful model initialization.
    is_model_initialized = (
        dit_handler.model is not None
    )

    accordion_state = gr.Accordion(
        open=not is_model_initialized
    )

    # Determine UI configuration from the actual loaded Synapse model.
    is_turbo = (
        dit_handler.is_turbo_model()
    )

    model_type_settings = (
        get_model_type_ui_settings(
            is_turbo
        )
    )

    return (
        status,
        gr.update(
            interactive=enable
        ),
        accordion_state,
        *model_type_settings,
    )


def get_model_type_ui_settings(
    is_turbo: bool,
):
    """
    Return generation UI settings for Synapse Turbo or Base models.
    """

    if is_turbo:
        # Synapse Turbo:
        # - maximum 20 steps
        # - default 8 steps
        # - shift enabled
        # - CFG/ADG controls hidden
        # - Turbo-compatible task set

        return (
            gr.update(
                value=8,
                maximum=20,
                minimum=1,
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                value=3.0,
                visible=True,
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                visible=False
            ),
            gr.update(
                choices=TASK_TYPES_TURBO
            ),
        )

    # Synapse Base:
    # - maximum 200 steps
    # - default 32 steps
    # - CFG/ADG/shift available
    # - complete task set

    return (
        gr.update(
            value=32,
            maximum=200,
            minimum=1,
        ),
        gr.update(
            visible=True
        ),
        gr.update(
            visible=True
        ),
        gr.update(
            value=3.0,
            visible=True,
        ),
        gr.update(
            visible=True
        ),
        gr.update(
            visible=True
        ),
        gr.update(
            choices=TASK_TYPES_BASE
        ),
    )


def update_negative_prompt_visibility(
    init_llm_checked,
):
    """
    Show the Synapse Composer negative-prompt control when
    Synapse Composer is enabled.
    """
    return gr.update(
        visible=init_llm_checked
    )


def update_audio_cover_strength_visibility(
    task_type_value,
    init_llm_checked,
):
    """
    Update audio-code / cover strength visibility and context-sensitive labels.
    """

    # Repaint never displays this control.
    is_repaint = (
        task_type_value == "repaint"
    )

    is_cover = (
        task_type_value == "cover"
    )

    is_visible = (
        is_cover
        or (
            init_llm_checked
            and not is_repaint
        )
    )

    if (
        init_llm_checked
        and not is_cover
    ):
        label = "Synapse Composer Code Strength"

        info = (
            "Control how many denoising steps use "
            "Synapse Composer-generated audio codes"
        )

    else:
        label = "Audio Cover Strength"

        info = (
            "Control how many denoising steps use "
            "the source audio in Cover mode"
        )

    return gr.update(
        visible=is_visible,
        label=label,
        info=info,
    )


def convert_src_audio_to_codes_wrapper(
    dit_handler,
    src_audio,
):
    """
    Convert source audio into Synapse audio codes.
    """
    codes_string = (
        dit_handler.convert_src_audio_to_codes(
            src_audio
        )
    )

    return codes_string


def update_instruction_ui(
    dit_handler,
    task_type_value: str,
    track_name_value: Optional[str],
    complete_track_classes_value: list,
    audio_codes_content: str = "",
    init_llm_checked: bool = False,
) -> tuple:
    """
    Update Synapse generation instructions and component visibility
    based on the selected task.
    """

    instruction = (
        dit_handler.generate_instruction(
            task_type=task_type_value,
            track_name=track_name_value,
            complete_track_classes=(
                complete_track_classes_value
            ),
        )
    )

    # Track selection is required for LEGO and extraction tasks.
    track_name_visible = (
        task_type_value
        in [
            "lego",
            "extract",
        ]
    )

    # Track classes are required for completion tasks.
    complete_visible = (
        task_type_value == "complete"
    )

    is_repaint = (
        task_type_value == "repaint"
    )

    is_cover = (
        task_type_value == "cover"
    )

    audio_cover_strength_visible = (
        is_cover
        or (
            init_llm_checked
            and not is_repaint
        )
    )

    if (
        init_llm_checked
        and not is_cover
    ):
        audio_cover_strength_label = (
            "Synapse Composer Code Strength"
        )

        audio_cover_strength_info = (
            "Control how many denoising steps use "
            "Synapse Composer-generated audio codes"
        )

    else:
        audio_cover_strength_label = (
            "Audio Cover Strength"
        )

        audio_cover_strength_info = (
            "Control how many denoising steps use "
            "the source audio in Cover mode"
        )

    # Repaint timing controls are used for Repaint and LEGO tasks.
    repainting_visible = (
        task_type_value
        in [
            "repaint",
            "lego",
        ]
    )

    # Keep audio-code controls visible when codes already exist, even if the
    # user switches away from text-to-music.
    has_audio_codes = bool(
        audio_codes_content
        and str(
            audio_codes_content
        ).strip()
    )

    text2music_audio_codes_visible = (
        task_type_value == "text2music"
        or has_audio_codes
    )

    return (
        instruction,
        gr.update(
            visible=track_name_visible
        ),
        gr.update(
            visible=complete_visible
        ),
        gr.update(
            visible=(
                audio_cover_strength_visible
            ),
            label=(
                audio_cover_strength_label
            ),
            info=(
                audio_cover_strength_info
            ),
        ),
        gr.update(
            visible=repainting_visible
        ),
        gr.update(
            visible=(
                text2music_audio_codes_visible
            )
        ),
    )


def transcribe_audio_codes(
    llm_handler,
    audio_code_string,
    constrained_decoding_debug,
):
    """
    Convert Synapse audio codes into music metadata using Synapse Composer.

    If audio_code_string is empty, Synapse Composer can generate an example
    composition instead.

    This is a Gradio wrapper around the understand_music API in
    synapse.inference.

    Args:
        llm_handler:
            Synapse Composer / language-model handler.

        audio_code_string:
            String containing Synapse audio codes, or an empty value for
            example generation.

        constrained_decoding_debug:
            Whether constrained-decoding debug logging is enabled.

    Returns:
        Tuple containing:
        status_message,
        caption,
        lyrics,
        bpm,
        duration,
        keyscale,
        language,
        timesignature,
        is_format_caption
    """

    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=audio_code_string,
        use_constrained_decoding=True,
        constrained_decoding_debug=(
            constrained_decoding_debug
        ),
    )

    if not result.success:
        if result.error == "LLM not initialized":
            return (
                t(
                    "messages.lm_not_initialized"
                ),
                "",
                "",
                None,
                None,
                "",
                "",
                "",
                False,
            )

        return (
            result.status_message,
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            False,
        )

    return (
        result.status_message,
        result.caption,
        result.lyrics,
        result.bpm,
        result.duration,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
    )


def update_transcribe_button_text(
    audio_code_string,
):
    """
    Update the Synapse Composer analysis button text.

    Empty input:
        Generate Example

    Existing audio codes:
        Analyze Audio
    """
    if (
        not audio_code_string
        or not audio_code_string.strip()
    ):
        return gr.update(
            value="Generate Example"
        )

    return gr.update(
        value="Analyze Audio"
    )


def reset_format_caption_flag():
    """
    Reset the formatted-caption state when the user manually edits
    generation metadata.
    """
    return False


def update_audio_uploads_accordion(
    reference_audio,
    src_audio,
):
    """
    Show the Audio Uploads section when reference or source audio exists.
    """
    has_audio = (
        reference_audio is not None
        or src_audio is not None
    )

    return gr.update(
        visible=has_audio
    )


def handle_instrumental_checkbox(
    instrumental_checked,
    current_lyrics,
):
    """
    Handle changes to the Instrumental checkbox.

    When enabled:
        If lyrics are empty, insert [Instrumental].

    When disabled:
        If lyrics contain only [Instrumental], clear them.
    """

    if instrumental_checked:
        if (
            not current_lyrics
            or not current_lyrics.strip()
        ):
            return "[Instrumental]"

        return current_lyrics

    if (
        current_lyrics
        and current_lyrics.strip()
        == "[Instrumental]"
    ):
        return ""

    return current_lyrics


def handle_simple_instrumental_change(
    is_instrumental: bool,
):
    """
    Handle Simple Mode instrumental state changes.

    Instrumental:
        Set vocal language to "unknown" and disable editing.

    Vocal:
        Re-enable vocal-language selection.
    """

    if is_instrumental:
        return gr.update(
            value="unknown",
            interactive=False,
        )

    return gr.update(
        interactive=True
    )


def update_audio_components_visibility(
    batch_size,
):
    """
    Show or hide generated-audio components based on batch size.

    Row 1:
        Outputs 1-4

    Row 2:
        Outputs 5-8
    """

    batch_size = min(
        max(
            int(batch_size),
            1,
        ),
        8,
    )

    updates_row1 = (
        gr.update(
            visible=True
        ),
        gr.update(
            visible=batch_size >= 2
        ),
        gr.update(
            visible=batch_size >= 3
        ),
        gr.update(
            visible=batch_size >= 4
        ),
    )

    show_row_5_8 = (
        batch_size >= 5
    )

    updates_row2 = (
        gr.update(
            visible=show_row_5_8
        ),
        gr.update(
            visible=batch_size >= 5
        ),
        gr.update(
            visible=batch_size >= 6
        ),
        gr.update(
            visible=batch_size >= 7
        ),
        gr.update(
            visible=batch_size >= 8
        ),
    )

    return (
        updates_row1
        + updates_row2
    )


def handle_generation_mode_change(
    mode: str,
):
    """
    Handle Synapse Music generation mode changes.

    Modes:
        Simple:
            Natural-language prompt powered by Synapse Composer.

        Custom:
            Direct control over prompt, lyrics, metadata, and generation
            parameters.

        Cover:
            Generate a new interpretation from source audio.

        Repaint:
            Regenerate a selected region of source audio.

    Returns updates for:
        - simple_mode_group
        - custom_mode_content
        - cover_mode_group
        - repainting_group
        - task_type
        - generate_btn
        - simple_sample_created
        - src_audio_group
        - audio_cover_strength
        - think_checkbox
    """

    is_simple = (
        mode == "simple"
    )

    is_cover = (
        mode == "cover"
    )

    is_repaint = (
        mode == "repaint"
    )

    task_type_map = {
        "simple": "text2music",
        "custom": "text2music",
        "cover": "cover",
        "repaint": "repaint",
    }

    task_type_value = (
        task_type_map.get(
            mode,
            "text2music",
        )
    )

    # Cover and Repaint use source-audio conditioning rather than
    # Synapse Composer thinking.
    if is_cover or is_repaint:
        think_checkbox_update = (
            gr.update(
                value=False,
                interactive=False,
            )
        )
    else:
        think_checkbox_update = (
            gr.update(
                value=True,
                interactive=True,
            )
        )

    return (
        gr.update(
            visible=is_simple
        ),
        gr.update(
            visible=not is_simple
        ),
        gr.update(
            visible=False
        ),
        gr.update(
            visible=is_repaint
        ),
        gr.update(
            value=task_type_value
        ),
        gr.update(
            interactive=True
        ),
        False,
        gr.update(
            visible=(
                is_cover
                or is_repaint
            )
        ),
        gr.update(
            visible=is_cover
        ),
        think_checkbox_update,
    )


def process_source_audio(
    dit_handler,
    llm_handler,
    src_audio,
    constrained_decoding_debug,
):
    """
    Process source audio through the Synapse Music analysis pipeline.

    The operation:
    1. Converts source audio into Synapse audio codes.
    2. Uses Synapse Composer to analyze those codes.
    3. Returns structured music metadata for the UI.

    Args:
        dit_handler:
            Synapse audio-model handler.

        llm_handler:
            Synapse Composer / language-model handler.

        src_audio:
            Path to source audio.

        constrained_decoding_debug:
            Whether constrained-decoding debug logging is enabled.

    Returns:
        Tuple containing:
        audio_codes,
        status_message,
        caption,
        lyrics,
        bpm,
        duration,
        keyscale,
        language,
        timesignature,
        is_format_caption
    """

    if src_audio is None:
        return (
            "",
            "No audio file provided",
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            False,
        )

    # -------------------------------------------------------------------------
    # Step 1 - Convert source audio into Synapse audio codes
    # -------------------------------------------------------------------------

    try:
        codes_string = (
            dit_handler.convert_src_audio_to_codes(
                src_audio
            )
        )

        if not codes_string:
            return (
                "",
                "Failed to convert audio to Synapse codes",
                "",
                "",
                None,
                None,
                "",
                "",
                "",
                False,
            )

    except Exception as error:
        return (
            "",
            f"Error converting audio: {str(error)}",
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            False,
        )

    # -------------------------------------------------------------------------
    # Step 2 - Analyze codes with Synapse Composer
    # -------------------------------------------------------------------------

    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=codes_string,
        use_constrained_decoding=True,
        constrained_decoding_debug=(
            constrained_decoding_debug
        ),
    )

    if not result.success:
        if result.error == "LLM not initialized":
            return (
                codes_string,
                t(
                    "messages.lm_not_initialized"
                ),
                "",
                "",
                None,
                None,
                "",
                "",
                "",
                False,
            )

        return (
            codes_string,
            result.status_message,
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            False,
        )

    return (
        codes_string,
        result.status_message,
        result.caption,
        result.lyrics,
        result.bpm,
        result.duration,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
    )


def handle_create_sample(
    llm_handler,
    query: str,
    instrumental: bool,
    vocal_language: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
):
    """
    Handle Synapse Music Simple Mode composition creation.

    Synapse Composer converts the user's natural-language music request into
    structured generation metadata including caption, lyrics, BPM, duration,
    key, language, and time signature.

    cfg_scale and negative_prompt are not used by this operation.

    Args:
        llm_handler:
            Synapse Composer / language-model handler.

        query:
            User's natural-language music description.

        instrumental:
            Whether the requested composition is instrumental.

        vocal_language:
            Preferred vocal language.

        lm_temperature:
            Composer sampling temperature.

        lm_top_k:
            Composer top-k sampling value.

        lm_top_p:
            Composer top-p sampling value.

        constrained_decoding_debug:
            Whether constrained-decoding debug logging is enabled.

    Returns:
        UI updates for the generated composition metadata.
    """

    # -------------------------------------------------------------------------
    # Verify Synapse Composer
    # -------------------------------------------------------------------------

    if not llm_handler.llm_initialized:
        gr.Warning(
            t(
                "messages.lm_not_initialized"
            )
        )

        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(
                interactive=False
            ),
            False,
            gr.update(),
            gr.update(),
            t(
                "messages.lm_not_initialized"
            ),
        )

    # -------------------------------------------------------------------------
    # Convert Composer sampling parameters
    # -------------------------------------------------------------------------

    top_k_value = (
        None
        if (
            not lm_top_k
            or lm_top_k == 0
        )
        else int(lm_top_k)
    )

    top_p_value = (
        None
        if (
            not lm_top_p
            or lm_top_p >= 1.0
        )
        else lm_top_p
    )

    # -------------------------------------------------------------------------
    # Generate structured composition
    # -------------------------------------------------------------------------

    result = create_sample(
        llm_handler=llm_handler,
        query=query,
        instrumental=instrumental,
        vocal_language=vocal_language,
        temperature=lm_temperature,
        top_k=top_k_value,
        top_p=top_p_value,
        use_constrained_decoding=True,
        constrained_decoding_debug=(
            constrained_decoding_debug
        ),
    )

    # -------------------------------------------------------------------------
    # Handle failure
    # -------------------------------------------------------------------------

    if not result.success:
        gr.Warning(
            result.status_message
            or t(
                "messages.sample_creation_failed"
            )
        )

        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(
                interactive=False
            ),
            False,
            gr.update(),
            gr.update(),
            (
                result.status_message
                or t(
                    "messages.sample_creation_failed"
                )
            ),
        )

    # -------------------------------------------------------------------------
    # Success
    # -------------------------------------------------------------------------

    gr.Info(
        t("messages.sample_created")
    )

    return (
        result.caption,
        result.lyrics,
        result.bpm,
        (
            result.duration
            if (
                result.duration
                and result.duration > 0
            )
            else -1
        ),
        result.keyscale,
        result.language,
        result.language,
        result.timesignature,
        result.instrumental,
        gr.Accordion(
            open=True
        ),
        gr.Accordion(
            open=True
        ),
        gr.update(
            interactive=True
        ),
        True,
        True,
        True,
        result.status_message,
    )


def handle_format_sample(
    llm_handler,
    caption: str,
    lyrics: str,
    bpm,
    audio_duration,
    key_scale: str,
    time_signature: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
):
    """
    Format user-provided music information with Synapse Composer.

    Synapse Composer uses the supplied caption, lyrics, and optional metadata
    to produce a normalized and enhanced composition description.

    cfg_scale and negative_prompt are not used by this operation.

    Args:
        llm_handler:
            Synapse Composer / language-model handler.

        caption:
            User's music description.

        lyrics:
            User-provided lyrics.

        bpm:
            Optional BPM constraint.

        audio_duration:
            Optional duration constraint.

        key_scale:
            Optional musical key constraint.

        time_signature:
            Optional time-signature constraint.

        lm_temperature:
            Composer sampling temperature.

        lm_top_k:
            Composer top-k sampling value.

        lm_top_p:
            Composer top-p sampling value.

        constrained_decoding_debug:
            Whether constrained-decoding debug logging is enabled.

    Returns:
        UI updates containing the formatted composition metadata.
    """

    # -------------------------------------------------------------------------
    # Verify Synapse Composer
    # -------------------------------------------------------------------------

    if not llm_handler.llm_initialized:
        gr.Warning(
            t(
                "messages.lm_not_initialized"
            )
        )

        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            t(
                "messages.lm_not_initialized"
            ),
        )

    # -------------------------------------------------------------------------
    # Build constrained metadata
    # -------------------------------------------------------------------------

    user_metadata = {}

    if (
        bpm is not None
        and bpm > 0
    ):
        user_metadata["bpm"] = int(
            bpm
        )

    if (
        audio_duration is not None
        and audio_duration > 0
    ):
        user_metadata["duration"] = int(
            audio_duration
        )

    if (
        key_scale
        and key_scale.strip()
    ):
        user_metadata["keyscale"] = (
            key_scale.strip()
        )

    if (
        time_signature
        and time_signature.strip()
    ):
        user_metadata["timesignature"] = (
            time_signature.strip()
        )

    user_metadata_to_pass = (
        user_metadata
        if user_metadata
        else None
    )

    # -------------------------------------------------------------------------
    # Convert Composer sampling parameters
    # -------------------------------------------------------------------------

    top_k_value = (
        None
        if (
            not lm_top_k
            or lm_top_k == 0
        )
        else int(lm_top_k)
    )

    top_p_value = (
        None
        if (
            not lm_top_p
            or lm_top_p >= 1.0
        )
        else lm_top_p
    )

    # -------------------------------------------------------------------------
    # Format composition
    # -------------------------------------------------------------------------

    result = format_sample(
        llm_handler=llm_handler,
        caption=caption,
        lyrics=lyrics,
        user_metadata=(
            user_metadata_to_pass
        ),
        temperature=lm_temperature,
        top_k=top_k_value,
        top_p=top_p_value,
        use_constrained_decoding=True,
        constrained_decoding_debug=(
            constrained_decoding_debug
        ),
    )

    # -------------------------------------------------------------------------
    # Handle failure
    # -------------------------------------------------------------------------

    if not result.success:
        gr.Warning(
            result.status_message
            or t(
                "messages.format_failed"
            )
        )

        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            (
                result.status_message
                or t(
                    "messages.format_failed"
                )
            ),
        )

    # -------------------------------------------------------------------------
    # Success
    # -------------------------------------------------------------------------

    gr.Info(
        t("messages.format_success")
    )

    return (
        result.caption,
        result.lyrics,
        result.bpm,
        (
            result.duration
            if (
                result.duration
                and result.duration > 0
            )
            else -1
        ),
        result.keyscale,
        result.language,
        result.timesignature,
        True,
        result.status_message,
    )