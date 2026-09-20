"""
Synapse Music V12 - Training Event Handlers

Contains all event handler functions for the Synapse Music dataset builder,
audio preprocessing pipeline, AI metadata generation, LoRA training,
training progress tracking, and LoRA export workflow.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from loguru import logger

from synapse.training.dataset_builder import DatasetBuilder, AudioSample


def create_dataset_builder() -> DatasetBuilder:
    """
    Create a new Synapse Music DatasetBuilder instance.

    Returns:
        DatasetBuilder instance.
    """
    return DatasetBuilder()


def scan_directory(
    audio_dir: str,
    dataset_name: str,
    custom_tag: str,
    tag_position: str,
    all_instrumental: bool,
    builder_state: Optional[DatasetBuilder],
) -> Tuple[Any, str, Any, DatasetBuilder]:
    """
    Scan a directory for audio files and populate the Synapse dataset.

    Returns:
        Tuple of:
            table_data,
            status,
            slider_update,
            builder_state
    """
    if not audio_dir or not audio_dir.strip():
        return (
            [],
            "❌ Please enter a directory path",
            gr.Slider(maximum=0, value=0),
            builder_state,
        )

    # Create a new builder or reuse the current dataset state.
    builder = (
        builder_state
        if builder_state
        else DatasetBuilder()
    )

    # Configure dataset metadata before scanning.
    builder.metadata.name = dataset_name
    builder.metadata.custom_tag = custom_tag
    builder.metadata.tag_position = tag_position
    builder.metadata.all_instrumental = all_instrumental

    # Scan the requested directory.
    samples, status = builder.scan_directory(
        audio_dir.strip()
    )

    if not samples:
        return (
            [],
            status,
            gr.Slider(maximum=0, value=0),
            builder,
        )

    # Apply dataset-wide instrumental and activation-tag settings.
    builder.set_all_instrumental(
        all_instrumental
    )

    if custom_tag:
        builder.set_custom_tag(
            custom_tag,
            tag_position,
        )

    table_data = (
        builder.get_samples_dataframe_data()
    )

    slider_max = max(
        0,
        len(samples) - 1,
    )

    return (
        table_data,
        status,
        gr.Slider(
            maximum=slider_max,
            value=0,
        ),
        builder,
    )


def auto_label_all(
    dit_handler,
    llm_handler,
    builder_state: Optional[DatasetBuilder],
    skip_metas: bool = False,
    progress=None,
) -> Tuple[List[List[Any]], str, DatasetBuilder]:
    """
    Automatically label all samples in the Synapse Music dataset.

    Args:
        dit_handler:
            Synapse Music DiT handler used for audio processing.

        llm_handler:
            Synapse Composer handler used for music metadata and
            caption generation.

        builder_state:
            Current Synapse dataset builder state.

        skip_metas:
            If True, skip Synapse Composer metadata generation and
            populate samples with safe default values.

        progress:
            Optional progress callback.

    Returns:
        Tuple of:
            table_data,
            status,
            builder_state
    """
    if builder_state is None:
        return (
            [],
            "❌ Please scan a directory first",
            builder_state,
        )

    if not builder_state.samples:
        return (
            [],
            "❌ No samples to label. Please scan a directory first.",
            builder_state,
        )

    # Skip Synapse Composer metadata generation and apply defaults.
    if skip_metas:
        for sample in builder_state.samples:
            sample.bpm = None
            sample.keyscale = "N/A"
            sample.timesignature = "N/A"
            sample.language = "unknown"

            if builder_state.metadata.custom_tag:
                sample.caption = (
                    builder_state.metadata.custom_tag
                )
            else:
                sample.caption = sample.filename

        table_data = (
            builder_state.get_samples_dataframe_data()
        )

        return (
            table_data,
            (
                "✅ Skipped Synapse Composer labeling. "
                f"{len(builder_state.samples)} samples were "
                "configured with default metadata."
            ),
            builder_state,
        )

    # Ensure the Synapse Music generation model is available.
    if (
        dit_handler is None
        or dit_handler.model is None
    ):
        return (
            builder_state.get_samples_dataframe_data(),
            (
                "❌ Synapse Music model is not initialized. "
                "Please initialize the service first."
            ),
            builder_state,
        )

    # Ensure Synapse Composer is available.
    if (
        llm_handler is None
        or not llm_handler.llm_initialized
    ):
        return (
            builder_state.get_samples_dataframe_data(),
            (
                "❌ Synapse Composer is not initialized. "
                "Please initialize the service with "
                "Synapse Composer enabled."
            ),
            builder_state,
        )

    def progress_callback(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    # Generate metadata for all samples.
    samples, status = (
        builder_state.label_all_samples(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            progress_callback=progress_callback,
        )
    )

    table_data = (
        builder_state.get_samples_dataframe_data()
    )

    return (
        table_data,
        status,
        builder_state,
    )


def get_sample_preview(
    sample_idx: int,
    builder_state: Optional[DatasetBuilder],
) -> Tuple[
    str,
    str,
    str,
    str,
    Optional[int],
    str,
    str,
    float,
    str,
    bool,
]:
    """
    Get preview data for a specific Synapse dataset sample.

    Returns:
        Tuple containing:
            audio_path,
            filename,
            caption,
            lyrics,
            bpm,
            keyscale,
            timesignature,
            duration,
            language,
            instrumental
    """
    if (
        builder_state is None
        or not builder_state.samples
    ):
        return (
            None,
            "",
            "",
            "",
            None,
            "",
            "",
            0.0,
            "instrumental",
            True,
        )

    idx = int(sample_idx)

    if (
        idx < 0
        or idx >= len(builder_state.samples)
    ):
        return (
            None,
            "",
            "",
            "",
            None,
            "",
            "",
            0.0,
            "instrumental",
            True,
        )

    sample = builder_state.samples[idx]

    return (
        sample.audio_path,
        sample.filename,
        sample.caption,
        sample.lyrics,
        sample.bpm,
        sample.keyscale,
        sample.timesignature,
        sample.duration,
        sample.language,
        sample.is_instrumental,
    )


def save_sample_edit(
    sample_idx: int,
    caption: str,
    lyrics: str,
    bpm: Optional[int],
    keyscale: str,
    timesig: str,
    language: str,
    is_instrumental: bool,
    builder_state: Optional[DatasetBuilder],
) -> Tuple[List[List[Any]], str, DatasetBuilder]:
    """
    Save edits to a Synapse dataset sample.

    Returns:
        Tuple of:
            table_data,
            status,
            builder_state
    """
    if builder_state is None:
        return (
            [],
            "❌ No dataset loaded",
            builder_state,
        )

    idx = int(sample_idx)

    sample, status = (
        builder_state.update_sample(
            idx,
            caption=caption,
            lyrics=(
                lyrics
                if not is_instrumental
                else "[Instrumental]"
            ),
            bpm=(
                int(bpm)
                if bpm
                else None
            ),
            keyscale=keyscale,
            timesignature=timesig,
            language=(
                "instrumental"
                if is_instrumental
                else language
            ),
            is_instrumental=is_instrumental,
            labeled=True,
        )
    )

    table_data = (
        builder_state.get_samples_dataframe_data()
    )

    return (
        table_data,
        status,
        builder_state,
    )


def update_settings(
    custom_tag: str,
    tag_position: str,
    all_instrumental: bool,
    builder_state: Optional[DatasetBuilder],
) -> DatasetBuilder:
    """
    Update Synapse dataset-wide settings.

    Returns:
        Updated DatasetBuilder state.
    """
    if builder_state is None:
        return builder_state

    if custom_tag:
        builder_state.set_custom_tag(
            custom_tag,
            tag_position,
        )

    builder_state.set_all_instrumental(
        all_instrumental
    )

    return builder_state


def save_dataset(
    save_path: str,
    dataset_name: str,
    builder_state: Optional[DatasetBuilder],
) -> str:
    """
    Save the Synapse dataset to a JSON file.

    Returns:
        Status message.
    """
    if builder_state is None:
        return (
            "❌ No dataset to save. "
            "Please scan a directory first."
        )

    if not builder_state.samples:
        return "❌ No samples in dataset."

    if not save_path or not save_path.strip():
        return "❌ Please enter a save path."

    labeled_count = (
        builder_state.get_labeled_count()
    )

    if labeled_count == 0:
        return (
            "⚠️ Warning: No samples have been labeled. "
            "Consider using Synapse auto-labeling first.\n"
            "Saving anyway..."
        )

    return builder_state.save_dataset(
        save_path.strip(),
        dataset_name,
    )


def load_existing_dataset_for_preprocess(
    dataset_path: str,
    builder_state: Optional[DatasetBuilder],
) -> Tuple[
    str,
    Any,
    Any,
    DatasetBuilder,
    str,
    str,
    str,
    str,
    Optional[int],
    str,
    str,
    float,
    str,
    bool,
]:
    """
    Load an existing Synapse dataset JSON file for preprocessing.

    This allows a previously prepared dataset to be loaded and sent
    directly into preprocessing without rescanning and relabeling all
    source audio.

    Returns:
        Tuple containing:
            status,
            table_data,
            slider_update,
            builder_state,
            audio_path,
            filename,
            caption,
            lyrics,
            bpm,
            keyscale,
            timesig,
            duration,
            language,
            instrumental
    """
    empty_preview = (
        None,
        "",
        "",
        "",
        None,
        "",
        "",
        0.0,
        "instrumental",
        True,
    )

    if (
        not dataset_path
        or not dataset_path.strip()
    ):
        return (
            "❌ Please enter a dataset path",
            [],
            gr.Slider(
                maximum=0,
                value=0,
            ),
            builder_state,
        ) + empty_preview

    dataset_path = dataset_path.strip()

    if not os.path.exists(dataset_path):
        return (
            f"❌ Dataset not found: {dataset_path}",
            [],
            gr.Slider(
                maximum=0,
                value=0,
            ),
            builder_state,
        ) + empty_preview

    # Always use a fresh builder when loading a saved dataset.
    builder = DatasetBuilder()

    samples, status = builder.load_dataset(
        dataset_path
    )

    if not samples:
        return (
            status,
            [],
            gr.Slider(
                maximum=0,
                value=0,
            ),
            builder,
        ) + empty_preview

    table_data = (
        builder.get_samples_dataframe_data()
    )

    slider_max = max(
        0,
        len(samples) - 1,
    )

    labeled_count = (
        builder.get_labeled_count()
    )

    info = (
        f"✅ Loaded Synapse dataset: "
        f"{builder.metadata.name}\n"
    )
    info += (
        f"📊 Samples: {len(samples)} "
        f"({labeled_count} labeled)\n"
    )
    info += (
        f"🏷️ Custom Tag: "
        f"{builder.metadata.custom_tag or '(none)'}\n"
    )
    info += (
        "📝 Ready for Synapse preprocessing! "
        "You can also edit samples below."
    )

    first_sample = builder.samples[0]

    preview = (
        first_sample.audio_path,
        first_sample.filename,
        first_sample.caption,
        first_sample.lyrics,
        first_sample.bpm,
        first_sample.keyscale,
        first_sample.timesignature,
        first_sample.duration,
        first_sample.language,
        first_sample.is_instrumental,
    )

    return (
        info,
        table_data,
        gr.Slider(
            maximum=slider_max,
            value=0,
        ),
        builder,
    ) + preview


def preprocess_dataset(
    output_dir: str,
    dit_handler,
    builder_state: Optional[DatasetBuilder],
    progress=None,
) -> str:
    """
    Preprocess a Synapse dataset into tensor files for fast LoRA training.

    The preprocessing pipeline converts audio into VAE latents and
    converts text conditioning information into model-ready embeddings.

    Returns:
        Status message.
    """
    if builder_state is None:
        return (
            "❌ No dataset loaded. "
            "Please scan a directory first."
        )

    if not builder_state.samples:
        return "❌ No samples in dataset."

    labeled_count = (
        builder_state.get_labeled_count()
    )

    if labeled_count == 0:
        return (
            "❌ No labeled samples. "
            "Please auto-label or manually label "
            "samples first."
        )

    if not output_dir or not output_dir.strip():
        return "❌ Please enter an output directory."

    if (
        dit_handler is None
        or dit_handler.model is None
    ):
        return (
            "❌ Synapse Music model is not initialized. "
            "Please initialize the service first."
        )

    def progress_callback(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    output_paths, status = (
        builder_state.preprocess_to_tensors(
            dit_handler=dit_handler,
            output_dir=output_dir.strip(),
            progress_callback=progress_callback,
        )
    )

    return status


def load_training_dataset(
    tensor_dir: str,
) -> str:
    """
    Load a preprocessed Synapse tensor dataset for training.

    Returns:
        Information about the loaded training dataset.
    """
    if (
        not tensor_dir
        or not tensor_dir.strip()
    ):
        return (
            "❌ Please enter a tensor directory path"
        )

    tensor_dir = tensor_dir.strip()

    if not os.path.exists(tensor_dir):
        return (
            f"❌ Directory not found: {tensor_dir}"
        )

    if not os.path.isdir(tensor_dir):
        return (
            f"❌ Not a directory: {tensor_dir}"
        )

    manifest_path = os.path.join(
        tensor_dir,
        "manifest.json",
    )

    if os.path.exists(manifest_path):
        try:
            with open(
                manifest_path,
                "r",
                encoding="utf-8",
            ) as file:
                manifest = json.load(file)

            num_samples = manifest.get(
                "num_samples",
                0,
            )

            metadata = manifest.get(
                "metadata",
                {},
            )

            name = metadata.get(
                "name",
                "Unknown",
            )

            custom_tag = metadata.get(
                "custom_tag",
                "",
            )

            info = (
                "✅ Loaded Synapse preprocessed "
                f"dataset: {name}\n"
            )
            info += (
                f"📊 Samples: {num_samples} "
                "preprocessed tensors\n"
            )
            info += (
                f"🏷️ Custom Tag: "
                f"{custom_tag or '(none)'}"
            )

            return info

        except Exception as error:
            logger.warning(
                "Failed to read Synapse training "
                f"manifest: {error}"
            )

    # Fallback to counting preprocessed tensor files.
    pt_files = [
        filename
        for filename in os.listdir(tensor_dir)
        if filename.endswith(".pt")
    ]

    if not pt_files:
        return (
            "❌ No .pt tensor files found in "
            f"{tensor_dir}"
        )

    info = (
        f"✅ Found {len(pt_files)} Synapse tensor "
        f"files in {tensor_dir}\n"
    )
    info += (
        "⚠️ No manifest.json found - "
        "using all .pt files"
    )

    return info


# =========================================================================
# Synapse LoRA Training Handlers
# =========================================================================


def _format_duration(seconds):
    """
    Format seconds into a compact human-readable duration string.
    """
    seconds = int(seconds)

    if seconds < 60:
        return f"{seconds}s"

    if seconds < 3600:
        return (
            f"{seconds // 60}m "
            f"{seconds % 60}s"
        )

    return (
        f"{seconds // 3600}h "
        f"{(seconds % 3600) // 60}m"
    )


def start_training(
    tensor_dir: str,
    dit_handler,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    train_epochs: int,
    train_batch_size: int,
    gradient_accumulation: int,
    save_every_n_epochs: int,
    training_shift: float,
    training_seed: int,
    lora_output_dir: str,
    training_state: Dict,
    progress=None,
):
    """
    Start Synapse Music LoRA training from preprocessed tensors.

    This generator yields progress, logs, loss history, and training state
    while the Synapse training process is running.
    """
    if (
        not tensor_dir
        or not tensor_dir.strip()
    ):
        yield (
            "❌ Please enter a tensor directory path",
            "",
            None,
            training_state,
        )
        return

    tensor_dir = tensor_dir.strip()

    if not os.path.exists(tensor_dir):
        yield (
            f"❌ Tensor directory not found: {tensor_dir}",
            "",
            None,
            training_state,
        )
        return

    if (
        dit_handler is None
        or dit_handler.model is None
    ):
        yield (
            (
                "❌ Synapse Music model is not initialized. "
                "Please initialize the service first."
            ),
            "",
            None,
            training_state,
        )
        return

    # Verify optional training dependencies.
    try:
        from lightning.fabric import Fabric
        from peft import (
            LoraConfig,
            get_peft_model,
        )

    except ImportError as error:
        yield (
            (
                f"❌ Missing required packages: {error}\n"
                "Please install: pip install peft lightning"
            ),
            "",
            None,
            training_state,
        )
        return

    training_state["is_training"] = True
    training_state["should_stop"] = False

    try:
        from synapse.training.trainer import LoRATrainer
        from synapse.training.configs import (
            LoRAConfig as LoRAConfigClass,
            TrainingConfig,
        )

        import pandas as pd

        # -------------------------------------------------------------
        # Build Synapse LoRA Configuration
        # -------------------------------------------------------------

        lora_config = LoRAConfigClass(
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        training_config = TrainingConfig(
            shift=training_shift,
            learning_rate=learning_rate,
            batch_size=train_batch_size,
            gradient_accumulation_steps=(
                gradient_accumulation
            ),
            max_epochs=train_epochs,
            save_every_n_epochs=(
                save_every_n_epochs
            ),
            seed=training_seed,
            output_dir=lora_output_dir,
        )

        # -------------------------------------------------------------
        # Initialize Training State
        # -------------------------------------------------------------

        log_lines = []

        loss_data = pd.DataFrame(
            {
                "step": [0],
                "loss": [0.0],
            }
        )

        start_time = time.time()

        yield (
            (
                "🚀 Starting Synapse LoRA training "
                f"from {tensor_dir}..."
            ),
            "",
            loss_data,
            training_state,
        )

        # -------------------------------------------------------------
        # Create Synapse LoRA Trainer
        # -------------------------------------------------------------

        trainer = LoRATrainer(
            dit_handler=dit_handler,
            lora_config=lora_config,
            training_config=training_config,
        )

        step_list = []
        loss_list = []

        # -------------------------------------------------------------
        # Training Loop
        # -------------------------------------------------------------

        for (
            step,
            loss,
            status,
        ) in trainer.train_from_preprocessed(
            tensor_dir,
            training_state,
        ):
            elapsed_seconds = (
                time.time() - start_time
            )

            time_info = (
                "⏱️ Elapsed: "
                f"{_format_duration(elapsed_seconds)}"
            )

            # Parse epoch progress to estimate remaining time.
            match = re.search(
                r"Epoch\s+(\d+)/(\d+)",
                str(status),
            )

            if match:
                current_ep = int(
                    match.group(1)
                )
                total_ep = int(
                    match.group(2)
                )

                if current_ep > 0:
                    eta_seconds = (
                        elapsed_seconds
                        / current_ep
                    ) * (
                        total_ep
                        - current_ep
                    )

                    time_info += (
                        " | ETA: ~"
                        f"{_format_duration(eta_seconds)}"
                    )

            display_status = (
                f"{status}\n{time_info}"
            )

            log_msg = (
                f"[{_format_duration(elapsed_seconds)}] "
                f"Step {step}: {status}"
            )

            logger.info(log_msg)

            log_lines.append(
                str(status)
            )

            if len(log_lines) > 15:
                log_lines = log_lines[-15:]

            log_text = "\n".join(
                log_lines
            )

            # Record valid loss values for the UI chart.
            if (
                step > 0
                and loss is not None
                and loss == loss
            ):
                step_list.append(
                    step
                )

                loss_list.append(
                    float(loss)
                )

                loss_data = pd.DataFrame(
                    {
                        "step": step_list,
                        "loss": loss_list,
                    }
                )

            yield (
                display_status,
                log_text,
                loss_data,
                training_state,
            )

            if training_state.get(
                "should_stop",
                False,
            ):
                logger.info(
                    "⏹️ Synapse training stopped by user"
                )

                log_lines.append(
                    "⏹️ Synapse training stopped by user"
                )

                yield (
                    f"⏹️ Stopped ({time_info})",
                    "\n".join(
                        log_lines[-15:]
                    ),
                    loss_data,
                    training_state,
                )

                break

        total_time = (
            time.time() - start_time
        )

        training_state["is_training"] = False

        completion_msg = (
            "✅ Synapse LoRA training completed! "
            "Total time: "
            f"{_format_duration(total_time)}"
        )

        logger.info(
            completion_msg
        )

        log_lines.append(
            completion_msg
        )

        yield (
            completion_msg,
            "\n".join(
                log_lines[-15:]
            ),
            loss_data,
            training_state,
        )

    except Exception as error:
        logger.exception(
            "Synapse training error"
        )

        training_state["is_training"] = False

        import pandas as pd

        empty_df = pd.DataFrame(
            {
                "step": [],
                "loss": [],
            }
        )

        yield (
            f"❌ Error: {str(error)}",
            str(error),
            empty_df,
            training_state,
        )


def stop_training(
    training_state: Dict,
) -> Tuple[str, Dict]:
    """
    Stop the currently running Synapse LoRA training process.

    Returns:
        Tuple containing:
            status,
            training_state
    """
    if not training_state.get(
        "is_training",
        False,
    ):
        return (
            "⚠️ No training in progress",
            training_state,
        )

    training_state["should_stop"] = True

    return (
        "⏹️ Stopping Synapse training...",
        training_state,
    )


def export_lora(
    export_path: str,
    lora_output_dir: str,
) -> str:
    """
    Export trained Synapse LoRA weights.

    The final trained adapter is preferred. If a final adapter is not
    available, the latest epoch checkpoint is exported instead.

    Returns:
        Export status message.
    """
    if (
        not export_path
        or not export_path.strip()
    ):
        return (
            "❌ Please enter an export path"
        )

    final_dir = os.path.join(
        lora_output_dir,
        "final",
    )

    checkpoint_dir = os.path.join(
        lora_output_dir,
        "checkpoints",
    )

    # Prefer the completed adapter.
    if os.path.exists(final_dir):
        source_path = final_dir

    # Otherwise export the newest available checkpoint.
    elif os.path.exists(checkpoint_dir):
        checkpoints = [
            directory
            for directory in os.listdir(
                checkpoint_dir
            )
            if directory.startswith(
                "epoch_"
            )
        ]

        if not checkpoints:
            return (
                "❌ No Synapse LoRA checkpoints found"
            )

        checkpoints.sort(
            key=lambda value: int(
                value.split("_")[1]
            )
        )

        latest = checkpoints[-1]

        source_path = os.path.join(
            checkpoint_dir,
            latest,
        )

    else:
        return (
            "❌ No trained Synapse LoRA model "
            f"found in {lora_output_dir}"
        )

    try:
        import shutil

        export_path = (
            export_path.strip()
        )

        parent_directory = (
            os.path.dirname(export_path)
            if os.path.dirname(export_path)
            else "."
        )

        os.makedirs(
            parent_directory,
            exist_ok=True,
        )

        if os.path.exists(export_path):
            shutil.rmtree(
                export_path
            )

        shutil.copytree(
            source_path,
            export_path,
        )

        return (
            "✅ Synapse LoRA exported to "
            f"{export_path}"
        )

    except Exception as error:
        logger.exception(
            "Synapse LoRA export error"
        )

        return (
            "❌ Synapse LoRA export failed: "
            f"{str(error)}"
        )