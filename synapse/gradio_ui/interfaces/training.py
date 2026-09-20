"""
Synapse Music V12 - Gradio UI Training Tab Module

Contains the Synapse Music V12 dataset builder, preprocessing pipeline,
AI-assisted metadata generation, and LoRA training interface components.
"""

import os
import gradio as gr

from synapse.gradio_ui.i18n import t


def create_training_section(dit_handler, llm_handler, init_params=None) -> dict:
    """
    Create the Synapse Music V12 training section with dataset builder,
    preprocessing, metadata generation, and LoRA training controls.

    Args:
        dit_handler:
            Synapse Music DiT generation handler instance.

        llm_handler:
            Synapse Music Composer / language-model handler instance.

        init_params:
            Dictionary containing initialization parameters and state.
            If None, the service is treated as not pre-initialized.

    Returns:
        Dictionary of Gradio components used by the Synapse training
        event-handling system.
    """

    # Check if running in service mode.
    # Training is hidden in restricted/service deployments.
    service_mode = (
        init_params is not None
        and init_params.get("service_mode", False)
    )

    with gr.Tab(
        "🎓 Synapse LoRA Training",
        visible=not service_mode,
    ):
        gr.HTML(
            """
            <div style="
                text-align: center;
                padding: 10px;
                margin-bottom: 15px;
            ">
                <h2>🎵 Synapse Music V12 LoRA Training</h2>
                <p>
                    Build music datasets, generate metadata,
                    preprocess audio, and train custom Synapse Music
                    LoRA adapters.
                </p>
            </div>
            """
        )

        with gr.Tabs():

            # =================================================================
            # Dataset Builder
            # =================================================================

            with gr.Tab("📁 Dataset Builder"):

                # =============================================================
                # Step 1 - Load Existing Dataset or Scan New Audio
                # =============================================================

                gr.HTML(
                    """
                    <div style="
                        padding: 10px;
                        margin-bottom: 10px;
                        border: 1px solid #4a4a6a;
                        border-radius: 8px;
                        background:
                            linear-gradient(
                                135deg,
                                #2a2a4a 0%,
                                #1a1a3a 100%
                            );
                    ">
                        <h3 style="margin: 0 0 5px 0;">
                            🚀 Quick Start
                        </h3>

                        <p style="margin: 0; color: #aaa;">
                            Choose one:
                            <b>Load an existing Synapse dataset</b>
                            or
                            <b>scan a new audio directory</b>.
                        </p>
                    </div>
                    """
                )

                with gr.Row():

                    # ---------------------------------------------------------
                    # Existing Dataset
                    # ---------------------------------------------------------

                    with gr.Column(scale=1):
                        gr.HTML(
                            "<h4>📂 Load Existing Dataset</h4>"
                        )

                        with gr.Row():
                            load_json_path = gr.Textbox(
                                label="Dataset JSON Path",
                                placeholder=(
                                    "./datasets/"
                                    "my_synapse_dataset.json"
                                ),
                                info=(
                                    "Load a previously saved "
                                    "Synapse Music dataset"
                                ),
                                scale=3,
                            )

                            load_json_btn = gr.Button(
                                "📂 Load",
                                variant="primary",
                                scale=1,
                            )

                        load_json_status = gr.Textbox(
                            label="Load Status",
                            interactive=False,
                        )

                    # ---------------------------------------------------------
                    # Scan Audio Directory
                    # ---------------------------------------------------------

                    with gr.Column(scale=1):
                        gr.HTML(
                            "<h4>🔍 Scan New Directory</h4>"
                        )

                        with gr.Row():
                            audio_directory = gr.Textbox(
                                label="Audio Directory Path",
                                placeholder=(
                                    "/path/to/your/audio/folder"
                                ),
                                info=(
                                    "Scan for audio files "
                                    "(wav, mp3, flac, ogg, opus)"
                                ),
                                scale=3,
                            )

                            scan_btn = gr.Button(
                                "🔍 Scan",
                                variant="secondary",
                                scale=1,
                            )

                        scan_status = gr.Textbox(
                            label="Scan Status",
                            interactive=False,
                        )

                gr.HTML("<hr>")

                # =============================================================
                # Dataset Files / Dataset Settings
                # =============================================================

                with gr.Row():

                    # ---------------------------------------------------------
                    # Audio Table
                    # ---------------------------------------------------------

                    with gr.Column(scale=2):

                        audio_files_table = gr.Dataframe(
                            headers=[
                                "#",
                                "Filename",
                                "Duration",
                                "Labeled",
                                "BPM",
                                "Key",
                                "Caption",
                            ],
                            datatype=[
                                "number",
                                "str",
                                "str",
                                "str",
                                "str",
                                "str",
                                "str",
                            ],
                            label="Synapse Dataset Audio Files",
                            interactive=False,
                            wrap=True,
                        )

                    # ---------------------------------------------------------
                    # Dataset Settings
                    # ---------------------------------------------------------

                    with gr.Column(scale=1):
                        gr.HTML(
                            "<h3>⚙️ Dataset Settings</h3>"
                        )

                        dataset_name = gr.Textbox(
                            label="Dataset Name",
                            value="my_synapse_dataset",
                            placeholder="Enter dataset name",
                        )

                        all_instrumental = gr.Checkbox(
                            label="All Instrumental",
                            value=True,
                            info=(
                                "Enable if all tracks are "
                                "instrumental with no vocals"
                            ),
                        )

                        need_lyrics = gr.Checkbox(
                            label="Transcribe Lyrics",
                            value=False,
                            info=(
                                "Attempt to transcribe lyrics "
                                "from dataset audio"
                            ),
                            interactive=False,
                        )

                        custom_tag = gr.Textbox(
                            label="Custom Activation Tag",
                            placeholder=(
                                "e.g. synapse_hardtekk, "
                                "my_style"
                            ),
                            info=(
                                "Unique activation tag used to "
                                "trigger this LoRA's learned style"
                            ),
                        )

                        tag_position = gr.Radio(
                            choices=[
                                (
                                    "Prepend (tag, caption)",
                                    "prepend",
                                ),
                                (
                                    "Append (caption, tag)",
                                    "append",
                                ),
                                (
                                    "Replace caption",
                                    "replace",
                                ),
                            ],
                            value="replace",
                            label="Tag Position",
                            info=(
                                "Choose where the activation tag "
                                "is placed in the training caption"
                            ),
                        )

                # =============================================================
                # Step 2 - Synapse AI Metadata
                # =============================================================

                gr.HTML(
                    "<hr>"
                    "<h3>🤖 Step 2: Auto-Label with Synapse AI</h3>"
                )

                with gr.Row():

                    with gr.Column(scale=3):
                        gr.Markdown(
                            """
Use Synapse Music's metadata pipeline to automatically analyze
the imported audio files and prepare training metadata:

- **Caption** — genre, style, mood, instrumentation and production description
- **BPM** — estimated beats per minute
- **Key** — musical key such as C Major or A Minor
- **Time Signature** — meter such as 4/4 or 3/4
- **Language / Vocal Metadata** — information used by the training pipeline

You can review and edit the generated metadata before preprocessing.
                            """
                        )

                        skip_metas = gr.Checkbox(
                            label="Skip Synapse Metadata Generation",
                            value=False,
                            info=(
                                "Skip AI metadata generation. "
                                "BPM, key and time signature will "
                                "remain unavailable where not "
                                "already present."
                            ),
                        )

                    with gr.Column(scale=1):
                        auto_label_btn = gr.Button(
                            "🏷️ Auto-Label All",
                            variant="primary",
                            size="lg",
                        )

                label_progress = gr.Textbox(
                    label="Synapse Labeling Progress",
                    interactive=False,
                    lines=2,
                )

                # =============================================================
                # Step 3 - Preview / Edit
                # =============================================================

                gr.HTML(
                    "<hr>"
                    "<h3>👀 Step 3: Preview & Edit</h3>"
                )

                with gr.Row():

                    # ---------------------------------------------------------
                    # Audio Preview
                    # ---------------------------------------------------------

                    with gr.Column(scale=1):
                        sample_selector = gr.Slider(
                            minimum=0,
                            maximum=0,
                            step=1,
                            value=0,
                            label="Select Sample #",
                            info=(
                                "Choose a dataset sample to "
                                "preview and edit"
                            ),
                        )

                        preview_audio = gr.Audio(
                            label="Audio Preview",
                            type="filepath",
                            interactive=False,
                        )

                        preview_filename = gr.Textbox(
                            label="Filename",
                            interactive=False,
                        )

                    # ---------------------------------------------------------
                    # Metadata Editor
                    # ---------------------------------------------------------

                    with gr.Column(scale=2):

                        with gr.Row():
                            edit_caption = gr.Textbox(
                                label="Music Caption",
                                lines=3,
                                placeholder=(
                                    "Music style, genre, mood, "
                                    "instrumentation and production..."
                                ),
                            )

                        with gr.Row():
                            edit_lyrics = gr.Textbox(
                                label="Lyrics",
                                lines=4,
                                placeholder=(
                                    "[Verse 1]\n"
                                    "Lyrics here...\n\n"
                                    "[Chorus]\n"
                                    "..."
                                ),
                            )

                        with gr.Row():
                            edit_bpm = gr.Number(
                                label="BPM",
                                precision=0,
                            )

                            edit_keyscale = gr.Textbox(
                                label="Key",
                                placeholder="C Major",
                            )

                            edit_timesig = gr.Dropdown(
                                choices=[
                                    "",
                                    "2",
                                    "3",
                                    "4",
                                    "6",
                                ],
                                label="Time Signature",
                            )

                            edit_duration = gr.Number(
                                label="Duration (s)",
                                precision=1,
                                interactive=False,
                            )

                        with gr.Row():
                            edit_language = gr.Dropdown(
                                choices=[
                                    "instrumental",
                                    "en",
                                    "zh",
                                    "ja",
                                    "ko",
                                    "es",
                                    "fr",
                                    "de",
                                    "pt",
                                    "ru",
                                    "unknown",
                                ],
                                value="instrumental",
                                label="Language",
                            )

                            edit_instrumental = gr.Checkbox(
                                label="Instrumental",
                                value=True,
                            )

                            save_edit_btn = gr.Button(
                                "💾 Save Changes",
                                variant="secondary",
                            )

                        edit_status = gr.Textbox(
                            label="Edit Status",
                            interactive=False,
                        )

                # =============================================================
                # Step 4 - Save Dataset
                # =============================================================

                gr.HTML(
                    "<hr>"
                    "<h3>💾 Step 4: Save Synapse Dataset</h3>"
                )

                with gr.Row():

                    with gr.Column(scale=3):
                        save_path = gr.Textbox(
                            label="Save Path",
                            value=(
                                "./datasets/"
                                "my_synapse_dataset.json"
                            ),
                            placeholder=(
                                "./datasets/dataset_name.json"
                            ),
                            info=(
                                "Path where the Synapse dataset "
                                "JSON will be saved"
                            ),
                        )

                    with gr.Column(scale=1):
                        save_dataset_btn = gr.Button(
                            "💾 Save Dataset",
                            variant="primary",
                            size="lg",
                        )

                save_status = gr.Textbox(
                    label="Save Status",
                    interactive=False,
                    lines=2,
                )

                # =============================================================
                # Step 5 - Preprocessing
                # =============================================================

                gr.HTML(
                    "<hr>"
                    "<h3>⚡ Step 5: Preprocess to Tensors</h3>"
                )

                gr.Markdown(
                    """
**Synapse preprocessing converts the prepared music dataset into
pre-computed tensors for faster LoRA training.**

You can either:

- Use the dataset created in Steps 1–4 above
- Load an existing Synapse dataset JSON file
                    """
                )

                with gr.Row():

                    with gr.Column(scale=3):
                        load_existing_dataset_path = gr.Textbox(
                            label=(
                                "Load Existing Dataset "
                                "(Optional)"
                            ),
                            placeholder=(
                                "./datasets/"
                                "my_synapse_dataset.json"
                            ),
                            info=(
                                "Path to a previously saved "
                                "Synapse dataset JSON file"
                            ),
                        )

                    with gr.Column(scale=1):
                        load_existing_dataset_btn = gr.Button(
                            "📂 Load Dataset",
                            variant="secondary",
                            size="lg",
                        )

                load_existing_status = gr.Textbox(
                    label="Load Status",
                    interactive=False,
                )

                gr.Markdown(
                    """
The Synapse preprocessing stage:

- Encodes audio into VAE latents
- Encodes music captions and lyrics into text embeddings
- Runs the conditioning encoder
- Pre-computes training representations
- Saves the resulting tensors as `.pt` files

⚠️ **The Synapse Music model must be loaded before preprocessing.
Depending on the dataset size and hardware, this process may take
several minutes.**
                    """
                )

                with gr.Row():

                    with gr.Column(scale=3):
                        preprocess_output_dir = gr.Textbox(
                            label="Tensor Output Directory",
                            value=(
                                "./datasets/"
                                "preprocessed_tensors"
                            ),
                            placeholder=(
                                "./datasets/"
                                "preprocessed_tensors"
                            ),
                            info=(
                                "Directory where Synapse "
                                "preprocessed tensor files "
                                "will be stored"
                            ),
                        )

                    with gr.Column(scale=1):
                        preprocess_btn = gr.Button(
                            "⚡ Preprocess",
                            variant="primary",
                            size="lg",
                        )

                preprocess_progress = gr.Textbox(
                    label="Preprocessing Progress",
                    interactive=False,
                    lines=3,
                )

            # =================================================================
            # LoRA Training
            # =================================================================

            with gr.Tab("🚀 Train Synapse LoRA"):

                with gr.Row():

                    # ---------------------------------------------------------
                    # Preprocessed Dataset Selection
                    # ---------------------------------------------------------

                    with gr.Column(scale=2):
                        gr.HTML(
                            "<h3>📊 Preprocessed Dataset Selection</h3>"
                        )

                        gr.Markdown(
                            """
Select the directory containing the Synapse preprocessed tensor
files (`.pt` files).

These files are created in the **Dataset Builder** tab using the
**Preprocess** button.
                            """
                        )

                        training_tensor_dir = gr.Textbox(
                            label=(
                                "Preprocessed Tensors Directory"
                            ),
                            placeholder=(
                                "./datasets/"
                                "preprocessed_tensors"
                            ),
                            value=(
                                "./datasets/"
                                "preprocessed_tensors"
                            ),
                            info=(
                                "Directory containing Synapse "
                                "preprocessed .pt tensor files"
                            ),
                        )

                        load_dataset_btn = gr.Button(
                            "📂 Load Dataset",
                            variant="secondary",
                        )

                        training_dataset_info = gr.Textbox(
                            label="Training Dataset Info",
                            interactive=False,
                            lines=3,
                        )

                    # ---------------------------------------------------------
                    # LoRA Configuration
                    # ---------------------------------------------------------

                    with gr.Column(scale=1):
                        gr.HTML(
                            "<h3>⚙️ Synapse LoRA Settings</h3>"
                        )

                        lora_rank = gr.Slider(
                            minimum=4,
                            maximum=256,
                            step=4,
                            value=64,
                            label="LoRA Rank (r)",
                            info=(
                                "Higher values provide more "
                                "adapter capacity but use more "
                                "memory"
                            ),
                        )

                        lora_alpha = gr.Slider(
                            minimum=4,
                            maximum=512,
                            step=4,
                            value=128,
                            label="LoRA Alpha",
                            info=(
                                "LoRA scaling factor "
                                "(commonly around 2× rank)"
                            ),
                        )

                        lora_dropout = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            step=0.05,
                            value=0.1,
                            label="LoRA Dropout",
                        )

                # =============================================================
                # Training Parameters
                # =============================================================

                gr.HTML(
                    "<hr>"
                    "<h3>🎛️ Synapse Training Parameters</h3>"
                )

                with gr.Row():

                    learning_rate = gr.Number(
                        label="Learning Rate",
                        value=1e-4,
                        info=(
                            "Initial optimizer learning rate"
                        ),
                    )

                    train_epochs = gr.Slider(
                        minimum=100,
                        maximum=4000,
                        step=100,
                        value=500,
                        label="Max Epochs",
                    )

                    train_batch_size = gr.Slider(
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=1,
                        label="Batch Size",
                        info=(
                            "Increase only when sufficient "
                            "GPU memory is available"
                        ),
                    )

                    gradient_accumulation = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=1,
                        label="Gradient Accumulation",
                        info=(
                            "Effective batch size = "
                            "batch size × accumulation"
                        ),
                    )

                with gr.Row():

                    save_every_n_epochs = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        step=50,
                        value=200,
                        label="Save Every N Epochs",
                    )

                    training_shift = gr.Slider(
                        minimum=1.0,
                        maximum=5.0,
                        step=0.5,
                        value=3.0,
                        label="Shift",
                        info=(
                            "Timestep shift used during "
                            "Synapse Music training"
                        ),
                    )

                    training_seed = gr.Number(
                        label="Seed",
                        value=42,
                        precision=0,
                    )

                with gr.Row():

                    lora_output_dir = gr.Textbox(
                        label="LoRA Output Directory",
                        value="./lora_output",
                        placeholder="./lora_output",
                        info=(
                            "Directory used to store trained "
                            "Synapse LoRA weights"
                        ),
                    )

                gr.HTML("<hr>")

                # =============================================================
                # Training Controls
                # =============================================================

                with gr.Row():

                    with gr.Column(scale=1):
                        start_training_btn = gr.Button(
                            "🚀 Start Synapse Training",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        stop_training_btn = gr.Button(
                            "⏹️ Stop Training",
                            variant="stop",
                            size="lg",
                        )

                training_progress = gr.Textbox(
                    label="Synapse Training Progress",
                    interactive=False,
                    lines=2,
                )

                with gr.Row():

                    training_log = gr.Textbox(
                        label="Training Log",
                        interactive=False,
                        lines=10,
                        max_lines=15,
                        scale=1,
                    )

                    training_loss_plot = gr.LinePlot(
                        x="step",
                        y="loss",
                        title="Synapse LoRA Training Loss",
                        x_title="Step",
                        y_title="Loss",
                        scale=1,
                    )

                # =============================================================
                # LoRA Export
                # =============================================================

                gr.HTML(
                    "<hr>"
                    "<h3>📦 Export Synapse LoRA</h3>"
                )

                with gr.Row():

                    export_path = gr.Textbox(
                        label="Export Path",
                        value=(
                            "./lora_output/"
                            "final_synapse_lora"
                        ),
                        placeholder=(
                            "./lora_output/"
                            "my_synapse_lora"
                        ),
                    )

                    export_lora_btn = gr.Button(
                        "📦 Export LoRA",
                        variant="secondary",
                    )

                export_status = gr.Textbox(
                    label="Export Status",
                    interactive=False,
                )

    # =====================================================================
    # Training State
    # =====================================================================

    dataset_builder_state = gr.State(
        None
    )

    training_state = gr.State(
        {
            "is_training": False,
            "should_stop": False,
        }
    )

    # =====================================================================
    # Component Registry
    #
    # Existing component keys are deliberately preserved so the current
    # Synapse event-handler wiring remains compatible during migration.
    # =====================================================================

    return {
        # Dataset Builder - Load or Scan
        "load_json_path": load_json_path,
        "load_json_btn": load_json_btn,
        "load_json_status": load_json_status,
        "audio_directory": audio_directory,
        "scan_btn": scan_btn,
        "scan_status": scan_status,
        "audio_files_table": audio_files_table,
        "dataset_name": dataset_name,
        "all_instrumental": all_instrumental,
        "need_lyrics": need_lyrics,
        "custom_tag": custom_tag,
        "tag_position": tag_position,
        "skip_metas": skip_metas,
        "auto_label_btn": auto_label_btn,
        "label_progress": label_progress,
        "sample_selector": sample_selector,
        "preview_audio": preview_audio,
        "preview_filename": preview_filename,
        "edit_caption": edit_caption,
        "edit_lyrics": edit_lyrics,
        "edit_bpm": edit_bpm,
        "edit_keyscale": edit_keyscale,
        "edit_timesig": edit_timesig,
        "edit_duration": edit_duration,
        "edit_language": edit_language,
        "edit_instrumental": edit_instrumental,
        "save_edit_btn": save_edit_btn,
        "edit_status": edit_status,
        "save_path": save_path,
        "save_dataset_btn": save_dataset_btn,
        "save_status": save_status,

        # Preprocessing
        "load_existing_dataset_path": (
            load_existing_dataset_path
        ),
        "load_existing_dataset_btn": (
            load_existing_dataset_btn
        ),
        "load_existing_status": load_existing_status,
        "preprocess_output_dir": preprocess_output_dir,
        "preprocess_btn": preprocess_btn,
        "preprocess_progress": preprocess_progress,
        "dataset_builder_state": dataset_builder_state,

        # Training
        "training_tensor_dir": training_tensor_dir,
        "load_dataset_btn": load_dataset_btn,
        "training_dataset_info": training_dataset_info,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "train_epochs": train_epochs,
        "train_batch_size": train_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "save_every_n_epochs": save_every_n_epochs,
        "training_shift": training_shift,
        "training_seed": training_seed,
        "lora_output_dir": lora_output_dir,
        "start_training_btn": start_training_btn,
        "stop_training_btn": stop_training_btn,
        "training_progress": training_progress,
        "training_log": training_log,
        "training_loss_plot": training_loss_plot,
        "export_path": export_path,
        "export_lora_btn": export_lora_btn,
        "export_status": export_status,
        "training_state": training_state,
    }