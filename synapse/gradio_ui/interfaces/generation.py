"""
Synapse Music V12 - Gradio UI Generation Section

Contains the generation section component definitions for the
simplified Synapse Music Studio interface.
"""

import gradio as gr

from synapse.constants import (
    VALID_LANGUAGES,
    TRACK_NAMES,
    TASK_TYPES_TURBO,
    TASK_TYPES_BASE,
    DEFAULT_DIT_INSTRUCTION,
)
from synapse.gradio_ui.i18n import t


def create_generation_section(
    dit_handler,
    llm_handler,
    init_params=None,
    language="en",
) -> dict:
    """
    Create the Synapse Music V12 generation section.

    Args:
        dit_handler:
            Synapse Music generation model handler instance.

        llm_handler:
            Synapse Composer model handler instance.

        init_params:
            Dictionary containing initialization parameters and state.
            If None, the service will not be treated as pre-initialized.

        language:
            UI language code ('en', 'zh', 'ja').

    Returns:
        dict:
            Dictionary containing all Gradio components required by the
            Synapse Music event system.
    """

    # Check if service is pre-initialized.
    service_pre_initialized = (
        init_params is not None
        and init_params.get("pre_initialized", False)
    )

    # Check if running in service mode (restricted UI).
    service_mode = (
        init_params is not None
        and init_params.get("service_mode", False)
    )

    # Get current language from init_params if available.
    current_language = (
        init_params.get("language", language)
        if init_params
        else language
    )

    # Get available Synapse Music generation models.
    available_dit_models = (
        init_params.get("available_dit_models", [])
        if init_params
        else []
    )

    current_model_value = (
        init_params.get("config_path", "")
        if init_params
        else ""
    )

    show_model_selector = len(available_dit_models) > 1

    with gr.Group():

        # =====================================================================
        # Synapse Service Configuration
        # Hidden when the service has already been initialized.
        # =====================================================================

        accordion_open = not service_pre_initialized
        accordion_visible = not service_pre_initialized

        with gr.Accordion(
            t("service.title"),
            open=accordion_open,
            visible=accordion_visible,
        ) as service_config_accordion:

            # -----------------------------------------------------------------
            # Language
            # -----------------------------------------------------------------

            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=[
                        ("English", "en"),
                        ("中文", "zh"),
                        ("日本語", "ja"),
                    ],
                    value=current_language,
                    label=t("service.language_label"),
                    info=t("service.language_info"),
                    scale=1,
                )

            # -----------------------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------------------

            with gr.Row(equal_height=True):
                with gr.Column(scale=4):

                    checkpoint_value = (
                        init_params.get("checkpoint")
                        if service_pre_initialized
                        else None
                    )

                    checkpoint_dropdown = gr.Dropdown(
                        label=t("service.checkpoint_label"),
                        choices=dit_handler.get_available_checkpoints(),
                        value=checkpoint_value,
                        info=t("service.checkpoint_info"),
                    )

                with gr.Column(scale=1, min_width=90):
                    refresh_btn = gr.Button(
                        t("service.refresh_btn"),
                        size="sm",
                    )

            # -----------------------------------------------------------------
            # Synapse Music Model + Device
            # -----------------------------------------------------------------

            with gr.Row():

                available_models = (
                    dit_handler.get_available_synapse_v12_models()
                )

                default_model = (
                    "synapse-v12-turbo"
                    if "synapse-v12-turbo" in available_models
                    else (
                        available_models[0]
                        if available_models
                        else None
                    )
                )

                config_path_value = (
                    init_params.get(
                        "config_path",
                        default_model,
                    )
                    if service_pre_initialized
                    else default_model
                )

                config_path = gr.Dropdown(
                    label=t("service.model_path_label"),
                    choices=available_models,
                    value=config_path_value,
                    info=t("service.model_path_info"),
                )

                device_value = (
                    init_params.get("device", "auto")
                    if service_pre_initialized
                    else "auto"
                )

                device = gr.Dropdown(
                    choices=[
                        "auto",
                        "cuda",
                        "cpu",
                    ],
                    value=device_value,
                    label=t("service.device_label"),
                    info=t("service.device_info"),
                )

            # -----------------------------------------------------------------
            # Synapse Composer + Backend
            # -----------------------------------------------------------------

            with gr.Row():

                available_lm_models = (
                    llm_handler.get_available_synapse_composer_models()
                )

                default_lm_model = (
                    "synapse-composer-0.6B"
                    if "synapse-composer-0.6B"
                    in available_lm_models
                    else (
                        available_lm_models[0]
                        if available_lm_models
                        else None
                    )
                )

                lm_model_path_value = (
                    init_params.get(
                        "lm_model_path",
                        default_lm_model,
                    )
                    if service_pre_initialized
                    else default_lm_model
                )

                lm_model_path = gr.Dropdown(
                    label=t("service.lm_model_path_label"),
                    choices=available_lm_models,
                    value=lm_model_path_value,
                    info=t("service.lm_model_path_info"),
                )

                backend_value = (
                    init_params.get(
                        "backend",
                        "vllm",
                    )
                    if service_pre_initialized
                    else "vllm"
                )

                backend_dropdown = gr.Dropdown(
                    choices=[
                        "vllm",
                        "pt",
                    ],
                    value=backend_value,
                    label=t("service.backend_label"),
                    info=t("service.backend_info"),
                )

            # -----------------------------------------------------------------
            # Runtime / Memory Settings
            # -----------------------------------------------------------------

            with gr.Row():

                init_llm_value = (
                    init_params.get(
                        "init_llm",
                        True,
                    )
                    if service_pre_initialized
                    else True
                )

                init_llm_checkbox = gr.Checkbox(
                    label=t("service.init_llm_label"),
                    value=init_llm_value,
                    info=t("service.init_llm_info"),
                )

                flash_attn_available = (
                    dit_handler.is_flash_attention_available()
                )

                use_flash_attention_value = (
                    init_params.get(
                        "use_flash_attention",
                        flash_attn_available,
                    )
                    if service_pre_initialized
                    else flash_attn_available
                )

                use_flash_attention_checkbox = gr.Checkbox(
                    label=t("service.flash_attention_label"),
                    value=use_flash_attention_value,
                    interactive=flash_attn_available,
                    info=(
                        t("service.flash_attention_info_enabled")
                        if flash_attn_available
                        else t(
                            "service.flash_attention_info_disabled"
                        )
                    ),
                )

                offload_to_cpu_value = (
                    init_params.get(
                        "offload_to_cpu",
                        False,
                    )
                    if service_pre_initialized
                    else False
                )

                offload_to_cpu_checkbox = gr.Checkbox(
                    label=t("service.offload_cpu_label"),
                    value=offload_to_cpu_value,
                    info=t("service.offload_cpu_info"),
                )

                offload_dit_to_cpu_value = (
                    init_params.get(
                        "offload_dit_to_cpu",
                        False,
                    )
                    if service_pre_initialized
                    else False
                )

                offload_dit_to_cpu_checkbox = gr.Checkbox(
                    label=t("service.offload_dit_cpu_label"),
                    value=offload_dit_to_cpu_value,
                    info=t("service.offload_dit_cpu_info"),
                )

            init_btn = gr.Button(
                t("service.init_btn"),
                variant="primary",
                size="lg",
            )

            init_status_value = (
                init_params.get(
                    "init_status",
                    "",
                )
                if service_pre_initialized
                else ""
            )

            init_status = gr.Textbox(
                label=t("service.status_label"),
                interactive=False,
                lines=3,
                value=init_status_value,
            )

            # -----------------------------------------------------------------
            # Synapse LoRA
            # -----------------------------------------------------------------

            gr.HTML(
                "<hr><h4>🧬 Synapse LoRA Adapter</h4>"
            )

            with gr.Row():

                lora_path = gr.Textbox(
                    label="Synapse LoRA Path",
                    placeholder="./lora_output/final/adapter",
                    info=(
                        "Path to a trained Synapse Music "
                        "LoRA adapter directory"
                    ),
                    scale=3,
                )

                load_lora_btn = gr.Button(
                    "📥 Load LoRA",
                    variant="secondary",
                    scale=1,
                )

                unload_lora_btn = gr.Button(
                    "🗑️ Unload",
                    variant="secondary",
                    scale=1,
                )

            with gr.Row():

                use_lora_checkbox = gr.Checkbox(
                    label="Use Synapse LoRA",
                    value=False,
                    info=(
                        "Enable the loaded Synapse LoRA "
                        "adapter for generation"
                    ),
                    scale=1,
                )

                lora_status = gr.Textbox(
                    label="LoRA Status",
                    value="No Synapse LoRA loaded",
                    interactive=False,
                    scale=2,
                )

        # =====================================================================
        # Model Selector
        # =====================================================================

        with gr.Row(
            visible=show_model_selector
        ):

            dit_model_selector = gr.Dropdown(
                choices=available_dit_models,
                value=current_model_value,
                label="Synapse Music Model",
                scale=1,
            )

        # Hidden dropdown when only one model.
        if not show_model_selector:

            dit_model_selector = gr.Dropdown(
                choices=(
                    available_dit_models
                    if available_dit_models
                    else [current_model_value]
                ),
                value=current_model_value,
                visible=False,
            )

        # =====================================================================
        # Generation Mode
        # =====================================================================

        gr.HTML(
            "<div style='"
            "background: #4a5568;"
            "color: white;"
            "padding: 8px 16px;"
            "border-radius: 4px;"
            "font-weight: bold;"
            "'>"
            "Synapse Generation Mode"
            "</div>"
        )

        with gr.Row():

            generation_mode = gr.Radio(
                choices=[
                    ("Simple", "simple"),
                    ("Custom", "custom"),
                    ("Cover", "cover"),
                    ("Repaint", "repaint"),
                ],
                value="custom",
                label="",
                show_label=False,
            )

        # =====================================================================
        # Simple Mode
        # =====================================================================

        with gr.Column(
            visible=False
        ) as simple_mode_group:

            with gr.Row(
                equal_height=True
            ):

                simple_query_input = gr.Textbox(
                    label=t(
                        "generation.simple_query_label"
                    ),
                    placeholder=t(
                        "generation.simple_query_placeholder"
                    ),
                    lines=2,
                    info=t(
                        "generation.simple_query_info"
                    ),
                    scale=10,
                )

                simple_vocal_language = gr.Dropdown(
                    choices=VALID_LANGUAGES,
                    value="unknown",
                    allow_custom_value=True,
                    label=t(
                        "generation.simple_vocal_language_label"
                    ),
                    interactive=True,
                    info=(
                        "Use unknown for automatic language "
                        "selection or instrumental generation"
                    ),
                    scale=2,
                )

                with gr.Column(
                    scale=1,
                    min_width=60,
                ):

                    random_desc_btn = gr.Button(
                        "🎲",
                        variant="primary",
                        size="lg",
                    )

            # Hidden compatibility components.
            simple_instrumental_checkbox = gr.Checkbox(
                label=t(
                    "generation.instrumental_label"
                ),
                value=False,
                visible=False,
            )

            create_sample_btn = gr.Button(
                t(
                    "generation.create_sample_btn"
                ),
                variant="primary",
                size="lg",
                visible=False,
            )

        # Tracks whether Synapse Composer has created a Simple-mode sample.
        simple_sample_created = gr.State(
            value=False
        )

        # =====================================================================
        # Source Audio
        # Cover / Repaint
        # =====================================================================

        with gr.Column(
            visible=False
        ) as src_audio_group:

            with gr.Row(
                equal_height=True
            ):

                src_audio = gr.Audio(
                    label="Source Audio",
                    type="filepath",
                    scale=10,
                )

                with gr.Column(
                    scale=1,
                    min_width=80,
                ):

                    process_src_btn = gr.Button(
                        "🧠 Analyze",
                        variant="secondary",
                        size="lg",
                    )

        # Hidden Composer audio-code storage.
        text2music_audio_code_string = gr.Textbox(
            label="Synapse Composer Audio Codes",
            visible=False,
        )

        # =====================================================================
        # Custom / Cover / Repaint Content
        # =====================================================================

        with gr.Column() as custom_mode_content:

            with gr.Row(
                equal_height=True
            ):

                # -------------------------------------------------------------
                # Reference Audio
                # -------------------------------------------------------------

                with gr.Column(
                    scale=2,
                    min_width=200,
                ):

                    reference_audio = gr.Audio(
                        label="Reference Audio (optional)",
                        type="filepath",
                        show_label=True,
                    )

                # -------------------------------------------------------------
                # Prompt + Lyrics
                # -------------------------------------------------------------

                with gr.Column(
                    scale=8
                ):

                    with gr.Row(
                        equal_height=True
                    ):

                        captions = gr.Textbox(
                            label="Music Style / Prompt",
                            placeholder=(
                                "Describe genre, style, mood, "
                                "instruments, vocals, production, "
                                "tempo and energy..."
                            ),
                            lines=12,
                            max_lines=12,
                            scale=1,
                        )

                        lyrics = gr.Textbox(
                            label="Lyrics",
                            placeholder=(
                                "Enter lyrics here...\n\n"
                                "Use [Intro], [Verse], [Chorus], "
                                "[Drop], [Bridge], [Outro], etc."
                            ),
                            lines=12,
                            max_lines=12,
                            scale=1,
                        )

                    format_btn = gr.Button(
                        "✨ Enhance with Synapse Composer",
                        variant="secondary",
                    )

                # -------------------------------------------------------------
                # Random Concept
                # -------------------------------------------------------------

                with gr.Column(
                    scale=1,
                    min_width=60,
                ):

                    sample_btn = gr.Button(
                        "🎲",
                        variant="primary",
                        size="lg",
                    )

        # Compatibility placeholder.
        audio_uploads_accordion = gr.Column(
            visible=False
        )

        # Legacy compatibility components.
        cover_mode_group = gr.Column(
            visible=False
        )

        convert_src_to_codes_btn = gr.Button(
            "Convert to Synapse Codes",
            visible=False,
        )

        # =====================================================================
        # Repaint Controls
        # =====================================================================

        with gr.Column(
            visible=False
        ) as repainting_group:

            with gr.Row():

                repainting_start = gr.Number(
                    label="Start (seconds)",
                    value=0.0,
                    step=0.1,
                    scale=1,
                )

                repainting_end = gr.Number(
                    label="End (seconds, -1 for end)",
                    value=-1,
                    minimum=-1,
                    step=0.1,
                    scale=1,
                )

        # =====================================================================
        # Optional Parameters
        # =====================================================================

        with gr.Accordion(
            "⚙️ Optional Parameters",
            open=False,
            visible=False,
        ) as optional_params_accordion:
            pass

        # =====================================================================
        # Advanced Settings
        # =====================================================================

        with gr.Accordion(
            "🔧 Advanced Synapse Settings",
            open=False,
        ) as advanced_options_accordion:

            with gr.Row():

                bpm = gr.Number(
                    label="BPM (optional)",
                    value=0,
                    step=1,
                    info="Leave empty or 0 for automatic BPM",
                    scale=1,
                )

                key_scale = gr.Textbox(
                    label="Key / Scale (optional)",
                    placeholder="Automatic",
                    value="",
                    info="Examples: C major, F minor, G# minor",
                    scale=1,
                )

                time_signature = gr.Dropdown(
                    choices=[
                        "",
                        "2",
                        "3",
                        "4",
                    ],
                    value="",
                    label="Time Signature (optional)",
                    allow_custom_value=True,
                    info="Examples: 2/4, 3/4, 4/4",
                    scale=1,
                )

                audio_duration = gr.Number(
                    label="Song Duration (seconds)",
                    value=-1,
                    minimum=-1,
                    maximum=600.0,
                    step=1,
                    info=(
                        "Use -1 for automatic duration, "
                        "or select 10–600 seconds"
                    ),
                    scale=1,
                )

                vocal_language = gr.Dropdown(
                    choices=VALID_LANGUAGES,
                    value="unknown",
                    label="Vocal Language",
                    allow_custom_value=True,
                    info=(
                        "Use unknown for automatic language "
                        "selection or instrumental generation"
                    ),
                    scale=1,
                )

                batch_size_input = gr.Number(
                    label="Number of Songs",
                    info="Generate up to 8 variations",
                    value=2,
                    minimum=1,
                    maximum=8,
                    step=1,
                    scale=1,
                    interactive=False,
                )

            # -----------------------------------------------------------------
            # Generation Steps / Seed / Format
            # -----------------------------------------------------------------

            with gr.Row():

                inference_steps = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=8,
                    step=1,
                    label="Synapse Generation Steps",
                    info=(
                        "Turbo: up to 8 steps. "
                        "Base: up to 200 steps."
                    ),
                )

                seed = gr.Textbox(
                    label="Seed",
                    value="-1",
                    info=(
                        "Use comma-separated values "
                        "for multiple songs"
                    ),
                )

                audio_format = gr.Dropdown(
                    choices=[
                        "mp3",
                        "flac",
                    ],
                    value="mp3",
                    label="Audio Format",
                    info=(
                        "Audio format for generated files"
                    ),
                )

            # -----------------------------------------------------------------
            # Shift / Random Seed / Inference
            # -----------------------------------------------------------------

            with gr.Row():

                shift = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,
                    step=0.1,
                    label="Timestep Shift",
                    info=(
                        "Timestep shift for Base models "
                        "(1.0–5.0, default 3.0). "
                        "Does not affect Turbo models."
                    ),
                )

                random_seed_checkbox = gr.Checkbox(
                    label="Random Seed",
                    value=True,
                    info=(
                        "Automatically generate new seeds"
                    ),
                )

                infer_method = gr.Dropdown(
                    choices=[
                        "ode",
                        "sde",
                    ],
                    value="ode",
                    label="Inference Method",
                    info=(
                        "ODE (Euler) is faster. "
                        "SDE introduces stochastic sampling."
                    ),
                )

            # -----------------------------------------------------------------
            # Custom Timesteps
            # -----------------------------------------------------------------

            custom_timesteps = gr.Textbox(
                label="Custom Timesteps",
                placeholder=(
                    "0.97,0.76,0.615,0.5,"
                    "0.395,0.28,0.18,0.085,0"
                ),
                value="",
                info=(
                    "Optional comma-separated values from "
                    "1.0 to 0.0. Overrides generation steps "
                    "and timestep shift."
                ),
            )

            # =================================================================
            # Synapse Composer
            # =================================================================

            gr.HTML(
                "<h4>🧠 Synapse Composer Parameters</h4>"
            )

            with gr.Row():

                lm_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.85,
                    step=0.05,
                    label="Composer Creativity",
                    info=(
                        "Higher values produce more varied "
                        "composition decisions"
                    ),
                )

                lm_cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="Composer Guidance Scale",
                    info=(
                        "Synapse Composer CFG strength "
                        "(1.0 disables CFG)"
                    ),
                )

                lm_top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Composer Top-K",
                    info="Top-K sampling (0 disables Top-K)",
                )

                lm_top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.9,
                    step=0.01,
                    label="Composer Top-P",
                    info=(
                        "Nucleus sampling threshold "
                        "(1.0 disables filtering)"
                    ),
                )

            lm_negative_prompt = gr.Textbox(
                label="Composer Negative Prompt",
                value="NO USER INPUT",
                placeholder=(
                    "Describe musical characteristics "
                    "to avoid..."
                ),
                lines=2,
                info=(
                    "Negative Composer conditioning used "
                    "when guidance is greater than 1.0"
                ),
            )

            # Hidden until advanced Cover controls are exposed.
            audio_cover_strength = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                visible=False,
            )

        # =====================================================================
        # Generate Button
        # =====================================================================

        generate_btn_interactive = (
            init_params.get(
                "enable_generate",
                False,
            )
            if service_pre_initialized
            else False
        )

        with gr.Row(
            equal_height=True
        ):

            # -----------------------------------------------------------------
            # Composition controls
            # -----------------------------------------------------------------

            with gr.Column(
                scale=1,
                min_width=120,
            ):

                think_checkbox = gr.Checkbox(
                    label="Composer Thinking",
                    value=True,
                )

                instrumental_checkbox = gr.Checkbox(
                    label="Instrumental",
                    value=False,
                )

            # -----------------------------------------------------------------
            # Generate
            # -----------------------------------------------------------------

            with gr.Column(
                scale=4
            ):

                generate_btn = gr.Button(
                    "🎵 Generate with Synapse",
                    variant="primary",
                    size="lg",
                    interactive=generate_btn_interactive,
                )

            # -----------------------------------------------------------------
            # Analysis
            # -----------------------------------------------------------------

            with gr.Column(
                scale=1,
                min_width=120,
            ):

                auto_score = gr.Checkbox(
                    label="Quality Score",
                    value=False,
                )

                auto_lrc = gr.Checkbox(
                    label="Lyrics Timing",
                    value=False,
                )

        # =====================================================================
        # Hidden Internal Components
        # =====================================================================

        # ---------------------------------------------------------------------
        # Task Type
        # ---------------------------------------------------------------------

        actual_model = (
            init_params.get(
                "config_path",
                "synapse-v12-turbo",
            )
            if service_pre_initialized
            else "synapse-v12-turbo"
        )

        actual_model_lower = (
            actual_model or ""
        ).lower()

        if "turbo" in actual_model_lower:
            initial_task_choices = TASK_TYPES_TURBO
        else:
            initial_task_choices = TASK_TYPES_BASE

        task_type = gr.Dropdown(
            choices=initial_task_choices,
            value="text2music",
            visible=False,
        )

        instruction_display_gen = gr.Textbox(
            value=DEFAULT_DIT_INSTRUCTION,
            visible=False,
        )

        track_name = gr.Dropdown(
            choices=TRACK_NAMES,
            value=None,
            visible=False,
        )

        complete_track_classes = gr.CheckboxGroup(
            choices=TRACK_NAMES,
            visible=False,
        )

        # ---------------------------------------------------------------------
        # Hidden generation settings
        # ---------------------------------------------------------------------

        guidance_scale = gr.Slider(
            value=7.0,
            visible=False,
        )

        use_adg = gr.Checkbox(
            value=False,
            visible=False,
        )

        cfg_interval_start = gr.Slider(
            value=0.0,
            visible=False,
        )

        cfg_interval_end = gr.Slider(
            value=1.0,
            visible=False,
        )

        # ---------------------------------------------------------------------
        # Hidden Synapse Composer settings
        # ---------------------------------------------------------------------

        use_cot_metas = gr.Checkbox(
            value=True,
            visible=False,
        )

        use_cot_caption = gr.Checkbox(
            value=True,
            visible=False,
        )

        use_cot_language = gr.Checkbox(
            value=True,
            visible=False,
        )

        constrained_decoding_debug = gr.Checkbox(
            value=False,
            visible=False,
        )

        allow_lm_batch = gr.Checkbox(
            value=True,
            visible=False,
        )

        lm_batch_chunk_size = gr.Number(
            value=8,
            visible=False,
        )

        score_scale = gr.Slider(
            minimum=0.01,
            maximum=1.0,
            value=0.5,
            visible=False,
        )

        autogen_checkbox = gr.Checkbox(
            value=False,
            visible=False,
        )

        # ---------------------------------------------------------------------
        # Hidden transcription / audio code controls
        # ---------------------------------------------------------------------

        transcribe_btn = gr.Button(
            value="Transcribe",
            visible=False,
        )

        text2music_audio_codes_group = gr.Group(
            visible=False
        )

        # ---------------------------------------------------------------------
        # Hidden load control
        # ---------------------------------------------------------------------

        load_file = gr.UploadButton(
            label="Load",
            file_types=[".json"],
            file_count="single",
            visible=False,
        )

        # ---------------------------------------------------------------------
        # Hidden compatibility accordions
        # ---------------------------------------------------------------------

        caption_accordion = gr.Accordion(
            "Music Style",
            visible=False,
        )

        lyrics_accordion = gr.Accordion(
            "Lyrics",
            visible=False,
        )

    # =========================================================================
    # Component Registry
    # =========================================================================

    return {
        "service_config_accordion": service_config_accordion,
        "language_dropdown": language_dropdown,
        "checkpoint_dropdown": checkpoint_dropdown,
        "refresh_btn": refresh_btn,
        "config_path": config_path,
        "device": device,
        "init_btn": init_btn,
        "init_status": init_status,

        # Synapse Composer
        "lm_model_path": lm_model_path,
        "init_llm_checkbox": init_llm_checkbox,
        "backend_dropdown": backend_dropdown,

        # Runtime
        "use_flash_attention_checkbox": use_flash_attention_checkbox,
        "offload_to_cpu_checkbox": offload_to_cpu_checkbox,
        "offload_dit_to_cpu_checkbox": offload_dit_to_cpu_checkbox,

        # LoRA
        "lora_path": lora_path,
        "load_lora_btn": load_lora_btn,
        "unload_lora_btn": unload_lora_btn,
        "use_lora_checkbox": use_lora_checkbox,
        "lora_status": lora_status,

        # Model selection
        "dit_model_selector": dit_model_selector,

        # Task configuration
        "task_type": task_type,
        "instruction_display_gen": instruction_display_gen,
        "track_name": track_name,
        "complete_track_classes": complete_track_classes,

        # Audio
        "audio_uploads_accordion": audio_uploads_accordion,
        "reference_audio": reference_audio,
        "src_audio": src_audio,
        "convert_src_to_codes_btn": convert_src_to_codes_btn,
        "text2music_audio_code_string": text2music_audio_code_string,
        "transcribe_btn": transcribe_btn,
        "text2music_audio_codes_group": text2music_audio_codes_group,

        # Composer parameters
        "lm_temperature": lm_temperature,
        "lm_cfg_scale": lm_cfg_scale,
        "lm_top_k": lm_top_k,
        "lm_top_p": lm_top_p,
        "lm_negative_prompt": lm_negative_prompt,
        "use_cot_metas": use_cot_metas,
        "use_cot_caption": use_cot_caption,
        "use_cot_language": use_cot_language,

        # Repaint
        "repainting_group": repainting_group,
        "repainting_start": repainting_start,
        "repainting_end": repainting_end,

        # Cover
        "audio_cover_strength": audio_cover_strength,

        # Generation modes
        "generation_mode": generation_mode,
        "simple_mode_group": simple_mode_group,
        "simple_query_input": simple_query_input,
        "random_desc_btn": random_desc_btn,
        "simple_instrumental_checkbox": simple_instrumental_checkbox,
        "simple_vocal_language": simple_vocal_language,
        "create_sample_btn": create_sample_btn,
        "simple_sample_created": simple_sample_created,

        # Compatibility accordions
        "caption_accordion": caption_accordion,
        "lyrics_accordion": lyrics_accordion,
        "optional_params_accordion": optional_params_accordion,

        # Custom mode
        "custom_mode_content": custom_mode_content,
        "cover_mode_group": cover_mode_group,

        # Source audio
        "src_audio_group": src_audio_group,
        "process_src_btn": process_src_btn,

        # Advanced options
        "advanced_options_accordion": advanced_options_accordion,

        # Main generation inputs
        "captions": captions,
        "sample_btn": sample_btn,
        "load_file": load_file,
        "lyrics": lyrics,
        "vocal_language": vocal_language,
        "bpm": bpm,
        "key_scale": key_scale,
        "time_signature": time_signature,
        "audio_duration": audio_duration,
        "batch_size_input": batch_size_input,

        # Generation settings
        "inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "random_seed_checkbox": random_seed_checkbox,
        "use_adg": use_adg,
        "cfg_interval_start": cfg_interval_start,
        "cfg_interval_end": cfg_interval_end,
        "shift": shift,
        "infer_method": infer_method,
        "custom_timesteps": custom_timesteps,
        "audio_format": audio_format,

        # Composer / generation behaviour
        "think_checkbox": think_checkbox,
        "autogen_checkbox": autogen_checkbox,

        # Main action
        "generate_btn": generate_btn,

        # Song mode
        "instrumental_checkbox": instrumental_checkbox,

        # Composer enhancement
        "format_btn": format_btn,

        # Diagnostics
        "constrained_decoding_debug": constrained_decoding_debug,
        "score_scale": score_scale,
        "allow_lm_batch": allow_lm_batch,
        "auto_score": auto_score,
        "auto_lrc": auto_lrc,
        "lm_batch_chunk_size": lm_batch_chunk_size,
    }