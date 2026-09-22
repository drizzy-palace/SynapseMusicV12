"""
Synapse Music V12 - Hugging Face Space Entry Point

This file serves as the main Hugging Face Space deployment entry point
for Synapse Music V12.

It initializes the Synapse Music generation engine, Synapse Composer,
dataset services, model routing, persistent storage, ZeroGPU integration,
and the Gradio interface.

ZeroGPU Support:
- ZeroGPU uses the `spaces` package to intercept CUDA operations.
- Models are loaded to "cuda" during startup, while actual GPU allocation
  is deferred by Hugging Face.
- Handlers are registered globally so forked processes inherit them
  without pickling.
- @spaces.GPU decorators are placed on top-level Gradio event handlers.
- nano-vllm uses direct CUDA APIs that bypass Spaces interception,
  therefore the PyTorch Composer backend is used on ZeroGPU.

Synapse Namespace:
- Internal Python package: `synapse`
- Primary generation handler: `SynapseHandler`
- Composer service: `LLMHandler`
- Dataset service: `DatasetHandler`
- UI package: `synapse.gradio_ui`

Model identifiers use the Synapse Music V12 naming system.
"""

import os
import sys


# =============================================================================
# Synapse Music V12
# =============================================================================

SYNAPSE_VERSION = "12.0.0"
SYNAPSE_PRODUCT_NAME = "Synapse Music V12"

# Main Synapse Hugging Face organization.
SYNAPSE_HF_ORG = os.environ.get(
    "SYNAPSE_HF_ORG",
    "SYNAPSEai1",
).strip()


# =============================================================================
# Synapse Music Model Ecosystem
# =============================================================================

SYNAPSE_MAIN_MODEL_REPO = os.environ.get(
    "SYNAPSE_MAIN_MODEL_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12",
).strip()

SYNAPSE_BASE_REPO = os.environ.get(
    "SYNAPSE_BASE_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-Base",
).strip()

SYNAPSE_SFT_REPO = os.environ.get(
    "SYNAPSE_SFT_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-SFT",
).strip()

SYNAPSE_XL_TURBO_REPO = os.environ.get(
    "SYNAPSE_XL_TURBO_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-XL-Turbo",
).strip()

SYNAPSE_XL_BASE_REPO = os.environ.get(
    "SYNAPSE_XL_BASE_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-XL-Base",
).strip()

SYNAPSE_XL_SFT_REPO = os.environ.get(
    "SYNAPSE_XL_SFT_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-XL-SFT",
).strip()

SYNAPSE_COMPOSER_17B_REPO = os.environ.get(
    "SYNAPSE_COMPOSER_17B_REPO",
    SYNAPSE_MAIN_MODEL_REPO,
).strip()

SYNAPSE_COMPOSER_06B_REPO = os.environ.get(
    "SYNAPSE_COMPOSER_06B_REPO",
    SYNAPSE_MAIN_MODEL_REPO,
).strip()


# =============================================================================
# Paths
# =============================================================================

current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

# Synapse-local nano-vllm package.
nano_vllm_path = os.path.join(
    current_dir,
    "synapse",
    "third_parts",
    "nano-vllm",
)

if os.path.exists(nano_vllm_path):
    sys.path.insert(
        0,
        nano_vllm_path,
    )


# =============================================================================
# Environment
# =============================================================================

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Expose Synapse runtime metadata.
os.environ.setdefault(
    "SYNAPSE_MUSIC_VERSION",
    SYNAPSE_VERSION,
)

os.environ.setdefault(
    "SYNAPSE_MUSIC_PRODUCT_NAME",
    SYNAPSE_PRODUCT_NAME,
)

os.environ.setdefault(
    "SYNAPSE_HF_ORG",
    SYNAPSE_HF_ORG,
)

os.environ.setdefault(
    "SYNAPSE_MAIN_MODEL_REPO",
    SYNAPSE_MAIN_MODEL_REPO,
)

os.environ.setdefault(
    "SYNAPSE_BASE_REPO",
    SYNAPSE_BASE_REPO,
)

os.environ.setdefault(
    "SYNAPSE_SFT_REPO",
    SYNAPSE_SFT_REPO,
)

os.environ.setdefault(
    "SYNAPSE_XL_TURBO_REPO",
    SYNAPSE_XL_TURBO_REPO,
)

os.environ.setdefault(
    "SYNAPSE_XL_BASE_REPO",
    SYNAPSE_XL_BASE_REPO,
)

os.environ.setdefault(
    "SYNAPSE_XL_SFT_REPO",
    SYNAPSE_XL_SFT_REPO,
)

os.environ.setdefault(
    "SYNAPSE_COMPOSER_17B_REPO",
    SYNAPSE_COMPOSER_17B_REPO,
)

os.environ.setdefault(
    "SYNAPSE_COMPOSER_06B_REPO",
    SYNAPSE_COMPOSER_06B_REPO,
)


# Clear proxy settings that may interfere with Gradio or model downloads.
for proxy_var in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
):
    os.environ.pop(
        proxy_var,
        None,
    )


# =============================================================================
# Hugging Face Spaces / ZeroGPU
# =============================================================================

# Import spaces before torch so ZeroGPU can intercept CUDA operations.
try:
    import spaces

    HAS_SPACES = True

except ImportError:
    spaces = None
    HAS_SPACES = False


import torch

from synapse.handler import SynapseHandler
from synapse.llm_inference import LLMHandler
from synapse.dataset_handler import DatasetHandler
from synapse.gradio_ui import create_gradio_interface


# Detect Hugging Face Space environment.
IS_HUGGINGFACE_SPACE = (
    os.environ.get("SPACE_ID") is not None
)

# Treat Hugging Face Spaces as ZeroGPU-compatible unless explicitly
# running in another environment.
IS_ZEROGPU = (
    IS_HUGGINGFACE_SPACE
    or os.environ.get("ZEROGPU") is not None
)


# =============================================================================
# Utility Functions
# =============================================================================

def get_gpu_memory_gb():
    """
    Return total CUDA GPU memory in GiB.

    Returns:
        float:
            GPU memory in GiB, or 0 when CUDA is unavailable or memory
            detection fails.
    """

    try:
        if torch.cuda.is_available():
            total_memory = (
                torch.cuda
                .get_device_properties(0)
                .total_memory
            )

            return total_memory / (1024 ** 3)

        return 0

    except Exception as exc:
        print(
            f"Warning: Failed to detect GPU memory: {exc}",
            file=sys.stderr,
        )

        return 0


def get_persistent_storage_path():
    """
    Detect and return a writable Synapse Music storage path.

    Hugging Face persistent storage:
    1. Persistent storage must be enabled in Space settings.
    2. The persistent path is normally /data.
    3. If /data is unavailable or not writable, Synapse falls back to
       ./data inside the Space.

    Local development:
    Set SYNAPSE_CHECKPOINT_DIR to override the storage root.

    Legacy CHECKPOINT_DIR is also accepted as a fallback while the
    deployment environment is migrated.

    Example:

        SYNAPSE_CHECKPOINT_DIR=/path/to/storage python app.py

    If the configured path points directly at a `checkpoints` directory,
    its parent directory is used as the storage root.
    """

    checkpoint_dir_override = (
        os.environ.get("SYNAPSE_CHECKPOINT_DIR")
        or os.environ.get("CHECKPOINT_DIR")
    )

    if checkpoint_dir_override:
        checkpoint_dir_override = (
            checkpoint_dir_override.strip()
        )

        if (
            checkpoint_dir_override.endswith("/checkpoints")
            or checkpoint_dir_override.endswith("\\checkpoints")
        ):
            checkpoint_dir_override = os.path.dirname(
                checkpoint_dir_override
            )

        if os.path.exists(checkpoint_dir_override):
            print(
                "Using Synapse checkpoint directory: "
                f"{checkpoint_dir_override}"
            )

            return checkpoint_dir_override

        print(
            "Warning: Synapse checkpoint directory "
            "does not exist: "
            f"{checkpoint_dir_override}"
        )

    # Hugging Face persistent storage.
    hf_data_path = "/data"

    if os.path.exists(hf_data_path):
        try:
            test_file = os.path.join(
                hf_data_path,
                ".synapse_write_test",
            )

            with open(
                test_file,
                "w",
                encoding="utf-8",
            ) as file:
                file.write("synapse")

            os.remove(test_file)

            print(
                "Using Hugging Face persistent storage: "
                f"{hf_data_path}"
            )

            return hf_data_path

        except (PermissionError, OSError) as exc:
            print(
                "Warning: /data exists but is not writable: "
                f"{exc}"
            )

    # Non-persistent fallback.
    fallback_path = os.path.join(
        current_dir,
        "data",
    )

    os.makedirs(
        fallback_path,
        exist_ok=True,
    )

    print(
        "Using local Synapse storage "
        "(non-persistent): "
        f"{fallback_path}"
    )

    print(
        "Note: Enable persistent storage in the "
        "Hugging Face Space settings to preserve "
        "downloaded Synapse checkpoints."
    )

    return fallback_path


def print_synapse_model_configuration():
    """
    Print the configured Synapse Music V12 model ecosystem.
    """

    print("=" * 64)
    print(f"{SYNAPSE_PRODUCT_NAME} Model Ecosystem")
    print("=" * 64)

    print(f"Main model:       {SYNAPSE_MAIN_MODEL_REPO}")
    print(f"Base:             {SYNAPSE_BASE_REPO}")
    print(f"SFT:              {SYNAPSE_SFT_REPO}")
    print(f"XL Turbo:         {SYNAPSE_XL_TURBO_REPO}")
    print(f"XL Base:          {SYNAPSE_XL_BASE_REPO}")
    print(f"XL SFT:           {SYNAPSE_XL_SFT_REPO}")
    print(f"Composer 1.7B:    {SYNAPSE_COMPOSER_17B_REPO}")
    print(f"Composer 0.6B:    {SYNAPSE_COMPOSER_06B_REPO}")

    print("=" * 64)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """
    Launch Synapse Music V12.
    """

    print("=" * 64)
    print(SYNAPSE_PRODUCT_NAME)
    print(f"Version: {SYNAPSE_VERSION}")
    print("Synapse Music Generation Engine")
    print("=" * 64)

    print_synapse_model_configuration()

    # =========================================================================
    # DEBUG_UI
    # =========================================================================

    debug_ui = os.environ.get(
        "DEBUG_UI",
        "",
    ).lower() in (
        "1",
        "true",
        "yes",
    )

    if debug_ui:
        print("=" * 64)
        print("DEBUG_UI mode enabled")
        print("- Model initialization will be skipped")
        print("- UI will remain available")
        print("- Music generation will be disabled")
        print("=" * 64)

    # =========================================================================
    # ZeroGPU
    # =========================================================================

    if IS_ZEROGPU:
        print("=" * 64)
        print("Hugging Face ZeroGPU environment detected")
        print("- Using Spaces GPU allocation")
        print("- PyTorch backend forced for Synapse Composer")
        print("- GPU allocated on demand during generation")
        print("=" * 64)

    # =========================================================================
    # Storage
    # =========================================================================

    persistent_storage_path = (
        get_persistent_storage_path()
    )

    # =========================================================================
    # Hardware Detection
    # =========================================================================

    gpu_memory_gb = get_gpu_memory_gb()

    if IS_ZEROGPU:
        auto_offload = False

        print(
            "ZeroGPU: CPU offload disabled "
            "(GPU allocated on demand)"
        )

    else:
        auto_offload = (
            gpu_memory_gb > 0
            and gpu_memory_gb < 16
        )

    if not debug_ui and not IS_ZEROGPU:
        if auto_offload:
            print(
                f"Detected GPU memory: "
                f"{gpu_memory_gb:.2f} GiB (<16 GiB)"
            )

            print(
                "Auto-enabling CPU offload to reduce "
                "GPU memory usage"
            )

        elif gpu_memory_gb > 0:
            print(
                f"Detected GPU memory: "
                f"{gpu_memory_gb:.2f} GiB (>=16 GiB)"
            )

            print(
                "CPU offload disabled by default"
            )

        else:
            print(
                "No CUDA GPU detected; running on CPU"
            )

    # =========================================================================
    # Handler Creation
    # =========================================================================

    print(
        "Creating Synapse Music handlers..."
    )

    dit_handler = SynapseHandler(
        persistent_storage_path=persistent_storage_path
    )

    llm_handler = LLMHandler(
        persistent_storage_path=persistent_storage_path
    )

    dataset_handler = DatasetHandler()

    # =========================================================================
    # Synapse Model Configuration
    # =========================================================================

    config_path = os.environ.get(
        "SYNAPSE_SERVICE_DIT_MODEL",
        os.environ.get(
            "SERVICE_MODE_DIT_MODEL",
            "synapse-v12-xl-turbo",
        ),
    ).strip()

    # Secondary model provides compatibility and fallback generation.
    config_path_2 = os.environ.get(
        "SYNAPSE_SERVICE_DIT_MODEL_2",
        os.environ.get(
            "SERVICE_MODE_DIT_MODEL_2",
            "synapse-v12-turbo",
        ),
    ).strip()

    lm_model_path = os.environ.get(
        "SYNAPSE_SERVICE_COMPOSER_MODEL",
        os.environ.get(
            "SERVICE_MODE_LM_MODEL",
            "synapse-composer-1.7B",
        ),
    ).strip()

    # =========================================================================
    # Backend
    # =========================================================================

    # nano-vllm accesses CUDA directly and therefore bypasses the Spaces
    # ZeroGPU interception layer.
    if IS_ZEROGPU:
        backend = "pt"

    else:
        backend = os.environ.get(
            "SYNAPSE_SERVICE_BACKEND",
            os.environ.get(
                "SERVICE_MODE_BACKEND",
                "vllm",
            ),
        ).strip()

    device = os.environ.get(
        "SYNAPSE_DEVICE",
        "auto",
    ).strip()

    # =========================================================================
    # Configuration Logging
    # =========================================================================

    print("=" * 64)
    print("Synapse Music Service Configuration")
    print("=" * 64)

    print(f"Generation model 1:  {config_path}")

    if config_path_2:
        print(f"Generation model 2:  {config_path_2}")

    print(f"Synapse Composer:    {lm_model_path}")
    print(f"Backend:             {backend}")
    print(f"Device:              {device}")
    print(f"CPU offload:         {auto_offload}")
    print(f"DEBUG_UI:            {debug_ui}")
    print(f"Hugging Face:        {IS_HUGGINGFACE_SPACE}")
    print(f"ZeroGPU:             {IS_ZEROGPU}")
    print(f"Spaces installed:    {HAS_SPACES}")

    # =========================================================================
    # Flash Attention
    # =========================================================================

    use_flash_attention = (
        dit_handler.is_flash_attention_available()
    )

    print(
        f"Flash Attention:     {use_flash_attention}"
    )

    print("=" * 64)

    # =========================================================================
    # Model Initialization
    # =========================================================================

    init_status = ""
    enable_generate = False
    dit_handler_2 = None

    if debug_ui:
        init_status = (
            "⚠️ DEBUG_UI mode - models not loaded\n"
            "UI is functional but generation is disabled"
        )

        enable_generate = False

        print(
            "Skipping Synapse model initialization "
            "(DEBUG_UI mode)"
        )

    else:
        # =====================================================================
        # Primary Generation Model
        # =====================================================================

        print(
            "Initializing Synapse primary generation model: "
            f"{config_path}..."
        )

        init_status, enable_generate = (
            dit_handler.initialize_service(
                project_root=current_dir,
                config_path=config_path,
                device=device,
                use_flash_attention=use_flash_attention,
                compile_model=False,
                offload_to_cpu=auto_offload,
                offload_dit_to_cpu=False,
            )
        )

        if not enable_generate:
            print(
                "Warning: Synapse primary generation "
                "model initialization issue: "
                f"{init_status}",
                file=sys.stderr,
            )

        else:
            print(
                "Synapse primary generation model "
                "initialized successfully"
            )

        # =====================================================================
        # Secondary Generation Model
        # =====================================================================

        if config_path_2:
            print(
                "Initializing Synapse secondary "
                "generation model: "
                f"{config_path_2}..."
            )

            dit_handler_2 = SynapseHandler(
                persistent_storage_path=persistent_storage_path
            )

            # Share large common components with the primary model.
            init_status_2, enable_generate_2 = (
                dit_handler_2.initialize_service(
                    project_root=current_dir,
                    config_path=config_path_2,
                    device=device,
                    use_flash_attention=use_flash_attention,
                    compile_model=False,
                    offload_to_cpu=auto_offload,
                    offload_dit_to_cpu=False,

                    shared_vae=dit_handler.vae,
                    shared_text_encoder=dit_handler.text_encoder,
                    shared_text_tokenizer=dit_handler.text_tokenizer,
                    shared_silence_latent=dit_handler.silence_latent,
                )
            )

            if not enable_generate_2:
                print(
                    "Warning: Synapse secondary "
                    "generation model initialization issue: "
                    f"{init_status_2}",
                    file=sys.stderr,
                )

                init_status += (
                    "\n⚠️ Secondary generation model failed: "
                    f"{init_status_2}"
                )

            else:
                print(
                    "Synapse secondary generation model "
                    "initialized successfully"
                )

                init_status += (
                    "\n✅ Secondary generation model: "
                    f"{config_path_2}"
                )

        # =====================================================================
        # Synapse Composer
        # =====================================================================

        checkpoint_dir = (
            dit_handler._get_checkpoint_dir()
        )

        print(
            "Initializing Synapse Composer: "
            f"{lm_model_path}..."
        )

        lm_status, lm_success = (
            llm_handler.initialize(
                checkpoint_dir=checkpoint_dir,
                lm_model_path=lm_model_path,
                backend=backend,
                device=device,
                offload_to_cpu=auto_offload,
                dtype=dit_handler.dtype,
            )
        )

        if lm_success:
            print(
                "Synapse Composer initialized successfully"
            )

            init_status += (
                f"\n{lm_status}"
            )

        else:
            print(
                "Warning: Synapse Composer initialization "
                "failed: "
                f"{lm_status}",
                file=sys.stderr,
            )

            init_status += (
                f"\n{lm_status}"
            )

    # =========================================================================
    # Available Models
    # =========================================================================

    available_dit_models = [
        config_path
    ]

    if (
        config_path_2
        and dit_handler_2 is not None
    ):
        available_dit_models.append(
            config_path_2
        )

    # =========================================================================
    # UI Initialization Parameters
    # =========================================================================

    init_params = {
        # ---------------------------------------------------------------------
        # Synapse Metadata
        # ---------------------------------------------------------------------

        "product_name": SYNAPSE_PRODUCT_NAME,
        "version": SYNAPSE_VERSION,

        # ---------------------------------------------------------------------
        # Service State
        # ---------------------------------------------------------------------

        "pre_initialized": True,
        "service_mode": True,

        # ---------------------------------------------------------------------
        # Generation Models
        # ---------------------------------------------------------------------

        "checkpoint": None,
        "config_path": config_path,
        "config_path_2": (
            config_path_2
            if config_path_2
            else None
        ),

        # ---------------------------------------------------------------------
        # Runtime
        # ---------------------------------------------------------------------

        "device": device,

        # ---------------------------------------------------------------------
        # Synapse Composer
        # ---------------------------------------------------------------------

        "init_llm": True,
        "lm_model_path": lm_model_path,
        "backend": backend,

        # ---------------------------------------------------------------------
        # Performance
        # ---------------------------------------------------------------------

        "use_flash_attention": use_flash_attention,
        "offload_to_cpu": auto_offload,
        "offload_dit_to_cpu": False,

        # ---------------------------------------------------------------------
        # Initialization State
        # ---------------------------------------------------------------------

        "init_status": init_status,
        "enable_generate": enable_generate,

        # ---------------------------------------------------------------------
        # Handlers
        # ---------------------------------------------------------------------

        "dit_handler": dit_handler,
        "dit_handler_2": dit_handler_2,
        "available_dit_models": available_dit_models,
        "llm_handler": llm_handler,

        # ---------------------------------------------------------------------
        # UI
        # ---------------------------------------------------------------------

        "language": "en",

        # ---------------------------------------------------------------------
        # Storage
        # ---------------------------------------------------------------------

        "persistent_storage_path": persistent_storage_path,

        # ---------------------------------------------------------------------
        # Development
        # ---------------------------------------------------------------------

        "debug_ui": debug_ui,

        # ---------------------------------------------------------------------
        # Synapse Model Repositories
        # ---------------------------------------------------------------------

        "synapse_model_repos": {
            "main": SYNAPSE_MAIN_MODEL_REPO,
            "base": SYNAPSE_BASE_REPO,
            "sft": SYNAPSE_SFT_REPO,
            "xl_turbo": SYNAPSE_XL_TURBO_REPO,
            "xl_base": SYNAPSE_XL_BASE_REPO,
            "xl_sft": SYNAPSE_XL_SFT_REPO,
            "composer_17b": SYNAPSE_COMPOSER_17B_REPO,
            "composer_06b": SYNAPSE_COMPOSER_06B_REPO,
        },
    }

    print("=" * 64)
    print(
        "Synapse Music service initialization completed"
    )
    print("=" * 64)

    # =========================================================================
    # Gradio
    # =========================================================================

    print(
        "Creating Synapse Music Gradio interface..."
    )

    demo = create_gradio_interface(
        dit_handler,
        llm_handler,
        dataset_handler,
        init_params=init_params,
        language="en",
    )

    # Multi-user request queue.
    queue_size = int(
        os.environ.get(
            "SYNAPSE_QUEUE_SIZE",
            "20",
        )
    )

    print(
        "Enabling Synapse Music request queue "
        f"(max size: {queue_size})..."
    )

    demo.queue(
        max_size=queue_size
    )

    # =========================================================================
    # Launch
    # =========================================================================

    server_name = os.environ.get(
        "SYNAPSE_SERVER_NAME",
        "0.0.0.0",
    ).strip()

    server_port = int(
        os.environ.get(
            "SYNAPSE_SERVER_PORT",
            "7860",
        )
    )

    print(
        f"Launching {SYNAPSE_PRODUCT_NAME} "
        f"on {server_name}:{server_port}..."
    )

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
