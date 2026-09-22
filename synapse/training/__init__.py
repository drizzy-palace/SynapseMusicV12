"""
Synapse Music V12 Training Module

This module provides LoRA training functionality for Synapse Music V12 models,
including dataset building, audio labeling, and training utilities.
"""

from synapse.training.dataset_builder import DatasetBuilder, AudioSample
from synapse.training.configs import LoRAConfig, TrainingConfig
from synapse.training.lora_utils import (
    inject_lora_into_dit,
    save_lora_weights,
    load_lora_weights,
    merge_lora_weights,
    check_peft_available,
)
from synapse.training.data_module import (
    # Preprocessed (recommended)
    PreprocessedTensorDataset,
    PreprocessedDataModule,
    collate_preprocessed_batch,
    # Legacy (raw audio)
    SynapseTrainingDataset,
    SynapseDataModule,
    collate_training_batch,
    load_dataset_from_json,
)
from synapse.training.trainer import LoRATrainer, PreprocessedLoRAModule, LIGHTNING_AVAILABLE

def check_lightning_available():
    """Check if Lightning Fabric is available."""
    return LIGHTNING_AVAILABLE

__all__ = [
    # Dataset Builder
    "DatasetBuilder",
    "AudioSample",
    # Configs
    "LoRAConfig",
    "TrainingConfig",
    # LoRA Utils
    "inject_lora_into_dit",
    "save_lora_weights",
    "load_lora_weights",
    "merge_lora_weights",
    "check_peft_available",
    # Data Module (Preprocessed - Recommended)
    "PreprocessedTensorDataset",
    "PreprocessedDataModule",
    "collate_preprocessed_batch",
    # Data Module (Legacy)
    "SynapseTrainingDataset",
    "SynapseDataModule",
    "collate_training_batch",
    "load_dataset_from_json",
    # Trainer
    "LoRATrainer",
    "PreprocessedLoRAModule",
    "check_lightning_available",
    "LIGHTNING_AVAILABLE",
]
