"""
Synapse Music V12 - Hugging Face Space Entry Point

Synapse Music V12 is built on the ACE-Step 1.5 architecture.

This file serves as the main Hugging Face Space deployment entry point.
It initializes the Synapse Music generation services and launches the
Gradio interface.

ZeroGPU Support:
- ZeroGPU uses the `spaces` package to intercept CUDA operations.
- Models are loaded to "cuda" during startup, but actual GPU allocation
  is deferred by Hugging Face.
- Handlers are registered globally so forked processes inherit them
  without pickling.
- @spaces.GPU decorators are placed on top-level Gradio event handlers.
- nano-vllm uses direct CUDA APIs that bypass Spaces interception,
  therefore the PyTorch LM backend is used on ZeroGPU.

Important:
The internal Python package namespace remains `acestep` for compatibility
with the upstream ACE-Step implementation. Synapse-specific model source
routing is configured separately and can be overridden through environment
variables.
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

# Synapse model repositories.
#
# These variables define the intended Synapse Music model ecosystem.
# The existing ACE-Step config/checkpoint names are intentionally preserved
# below until the checkpoint downloader/handler is wired to these repositories.
SYNAPSE_MAIN_MODEL_REPO = os.environ.get(
    "SYNAPSE_MAIN_MODEL_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12",
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

SYNAPSE_COMPOSER_4B_REPO = os.environ.get(
    "SYNAPSE_COMPOSER_4B_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-Composer-4B",
).strip()

SYNAPSE_COMPOSER_06B_REPO = os.environ.get(
    "SYNAPSE_COMPOSER_06B_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-Composer-0.6B",
).strip()

SYNAPSE_CAPTIONER_REPO = os.environ.get(
    "SYNAPSE_CAPTIONER_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-Captioner",
).strip()

SYNAPSE_TRANSCRIBER_REPO = os.environ.get(
    "SYNAPSE_TRANSCRIBER_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-Transcriber",
).strip()

SYNAPSE_VAE_REPO = os.environ.get(
    "SYNAPSE_VAE_REPO",
    f"{SYNAPSE_HF_ORG}/SynapseMusicV12-VAE",
).strip()


# =============================================================================
# Paths
# =============================================================================

# Get current directory (app.py location).
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add nano-vllm to Python path (local ACE-Step third-party package).
#
# Keep this path under `acestep` until the internal package itself is renamed.
nano_vllm_path = os.path.join(
    current_dir,
    "acestep",
    "third_parts",
    "nano-vllm",
)

if os.path.exists(nano_vllm_path):
    sys.path.insert(0, nano_vllm_path)


# =============================================================================
# Environment
# =============================================================================

# Disable Gradio analytics.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Expose Synapse runtime metadata to child processes/modules.
os.environ.setdefault("SYNAPSE_MUSIC_VERSION", SYNAPSE_VERSION)
os.environ.setdefault("SYNAPSE_MUSIC_PRODUCT_NAME", SYNAPSE_PRODUCT_NAME)

# Clear proxy settings that may affect Gradio.
for proxy_var in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
):
    os.environ.pop(proxy_var, None)


# =============================================================================
# Hugging Face Spaces / ZeroGPU
# =============================================================================

# Import spaces before torch so ZeroGPU can intercept CUDA operations.
# This becomes a no-op outside Hugging Face Spaces.
try:
    import spaces

    HAS_SPACES = True
except ImportError:
    spaces = None
    HAS_SPACES = False


import torch

# Keep upstream package imports for compatibility.
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.dataset_handler import DatasetHandler
from acestep.gradio_ui import create_gradio_interface


# Detect Hugging Face Space environment.
IS_HUGGINGFACE_SPACE = os.environ.get("SPACE_ID") is not None

# ZeroGPU detection.
#
# SPACE_HARDWARE has historically not been completely reliable, so the
# upstream-safe behaviour is retained: Hugging Face Spaces are treated as
# ZeroGPU-compatible unless explicitly running elsewhere.
IS_ZEROGPU = (
    IS_HUGGINGFACE_SPACE
    or os.environ.get("ZEROGPU") is not None
)


# =============================================================================
# Utility functions
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
            total_memory = torch.cuda.get_device_properties(0).total_memory
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
    Detect and return a writable storage path.

    Hugging Face persistent storage:
    1. Persistent storage must be enabled in Space settings.
    2. The persistent path is normally /data.
    3. If /data is unavailable or not writable, Synapse falls back to
       ./data inside the Space.

    Local development:
    Set CHECKPOINT_DIR to override the checkpoint storage root.

    Example:

        CHECKPOINT_DIR=/path/to/storage python app.py

    If CHECKPOINT_DIR points directly at a `checkpoints` directory, its
    parent directory is used because ACE-Step expects the storage root.
    """

    checkpoint_dir_override = os.environ.get("CHECKPOINT_DIR")

    if checkpoint_dir_override:
        checkpoint_dir_override = checkpoint_dir_override.strip()

        if (
            checkpoint_dir_override.endswith("/checkpoints")
            or checkpoint_dir_override.endswith("\\checkpoints")
        ):
            checkpoint_dir_override = os.path.dirname(
                checkpoint_dir_override
            )

        if os.path.exists(checkpoint_dir_override):
            print(
                "Using local Synapse checkpoint directory "
                f"(CHECKPOINT_DIR): {checkpoint_dir_override}"
            )
            return checkpoint_dir_override

        print(
            "Warning: CHECKPOINT_DIR path does not exist: "
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

            with open(test_file, "w", encoding="utf-8") as file:
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
        "Using local Synapse storage (non-persistent): "
        f"{fallback_path}"
    )

    print(
        "Note: Enable persistent storage in the Hugging Face "
        "Space settings to preserve downloaded checkpoints."
    )

    return fallback_path


def print_synapse_model_configuration():
    """
    Print the configured Synapse Music Hugging Face model ecosystem.

    These are repository sources. The actual ACE-Step checkpoint/config
    routing remains separate until handler-level model download routing
    is configured.
    """

    print("=" * 60)
    print(f"{SYNAPSE_PRODUCT_NAME} Model Ecosystem")
    print("=" * 60)

    print(f"Main model:      {SYNAPSE_MAIN_MODEL_REPO}")
    print(f"XL Turbo:        {SYNAPSE_XL_TURBO_REPO}")
    print(f"XL Base:         {SYNAPSE_XL_BASE_REPO}")
    print(f"XL SFT:          {SYNAPSE_XL_SFT_REPO}")
    print(f"Composer 4B:     {SYNAPSE_COMPOSER_4B_REPO}")
    print(f"Composer 0.6B:   {SYNAPSE_COMPOSER_06B_REPO}")
    print(f"Captioner:       {SYNAPSE_CAPTIONER_REPO}")
    print(f"Transcriber:     {SYNAPSE_TRANSCRIBER_REPO}")
    print(f"VAE:             {SYNAPSE_VAE_REPO}")

    print("=" * 60)


# =============================================================================
# Main application
# =============================================================================

def main():
    """
    Launch Synapse Music V12.
    """

    print("=" * 60)
    print(SYNAPSE_PRODUCT_NAME)
    print(f"Version: {SYNAPSE_VERSION}")
    print("Foundation: ACE-Step 1.5")
    print("=" * 60)

    print_synapse_model_configuration()

    # -------------------------------------------------------------------------
    # DEBUG_UI
    # -------------------------------------------------------------------------

    debug_ui = os.environ.get(
        "DEBUG_UI",
        "",
    ).lower() in (
        "1",
        "true",
        "yes",
    )

    if debug_ui:
        print("=" * 60)
        print("DEBUG_UI mode enabled")
        print("- Model initialization will be skipped")
        print("- UI will remain available")
        print("- Music generation will be disabled")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # ZeroGPU
    # -------------------------------------------------------------------------

    if IS_ZEROGPU:
        print("=" * 60)
        print("Hugging Face ZeroGPU environment detected")
        print("- Using Spaces GPU allocation")
        print("- PyTorch backend forced for Composer LM")
        print("- GPU allocated on demand during generation")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    persistent_storage_path = get_persistent_storage_path()

    # -------------------------------------------------------------------------
    # Hardware detection
    # -------------------------------------------------------------------------

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
                f"Detected GPU memory: {gpu_memory_gb:.2f} GiB (<16 GiB)"
            )
            print(
                "Auto-enabling CPU offload to reduce GPU memory usage"
            )

        elif gpu_memory_gb > 0:
            print(
                f"Detected GPU memory: {gpu_memory_gb:.2f} GiB (>=16 GiB)"
            )
            print(
                "CPU offload disabled by default"
            )

        else:
            print(
                "No CUDA GPU detected; running on CPU"
            )

    # -------------------------------------------------------------------------
    # Handler creation
    # -------------------------------------------------------------------------

    print("Creating Synapse Music handlers...")

    dit_handler = AceStepHandler(
        persistent_storage_path=persistent_storage_path
    )

    llm_handler = LLMHandler(
        persistent_storage_path=persistent_storage_path
    )

    dataset_handler = DatasetHandler()

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # These remain ACE-Step checkpoint/config identifiers for now.
    #
    # Do NOT replace these with Hugging Face repository IDs until we have
    # updated AceStepHandler's checkpoint download/resolution logic.
    #
    # The Synapse repository IDs defined at the top of this file describe
    # where those checkpoints should eventually be downloaded from.
    # -------------------------------------------------------------------------

    config_path = os.environ.get(
        "SERVICE_MODE_DIT_MODEL",
        "acestep-v15-xl-turbo",
    ).strip()

    # Secondary model provides compatibility/fallback generation.
    config_path_2 = os.environ.get(
        "SERVICE_MODE_DIT_MODEL_2",
        "acestep-v15-turbo",
    ).strip()

    lm_model_path = os.environ.get(
        "SERVICE_MODE_LM_MODEL",
        "acestep-5Hz-lm-1.7B",
    ).strip()

    # -------------------------------------------------------------------------
    # Backend
    # -------------------------------------------------------------------------

    # nano-vllm accesses CUDA directly and therefore bypasses the Spaces
    # ZeroGPU interception layer.
    if IS_ZEROGPU:
        backend = "pt"
    else:
        backend = os.environ.get(
            "SERVICE_MODE_BACKEND",
            "vllm",
        ).strip()

    device = os.environ.get(
        "SYNAPSE_DEVICE",
        "auto",
    ).strip()

    # -------------------------------------------------------------------------
    # Configuration logging
    # -------------------------------------------------------------------------

    print("=" * 60)
    print("Synapse Music Service Configuration")
    print("=" * 60)

    print(f"DiT model 1:       {config_path}")

    if config_path_2:
        print(f"DiT model 2:       {config_path_2}")

    print(f"Composer LM:       {lm_model_path}")
    print(f"Backend:           {backend}")
    print(f"Device:            {device}")
    print(f"CPU offload:       {auto_offload}")
    print(f"DEBUG_UI:          {debug_ui}")
    print(f"Hugging Face:      {IS_HUGGINGFACE_SPACE}")
    print(f"ZeroGPU:           {IS_ZEROGPU}")
    print(f"Spaces installed:  {HAS_SPACES}")

    # -------------------------------------------------------------------------
    # Flash Attention
    # -------------------------------------------------------------------------

    use_flash_attention = (
        dit_handler.is_flash_attention_available()
    )

    print(
        f"Flash Attention:   {use_flash_attention}"
    )

    print("=" * 60)

    # -------------------------------------------------------------------------
    # Model initialization
    # -------------------------------------------------------------------------

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
            "Skipping model initialization "
            "(DEBUG_UI mode)"
        )

    else:
        # ---------------------------------------------------------------------
        # Primary DiT
        # ---------------------------------------------------------------------

        print(
            f"Initializing Synapse primary DiT: {config_path}..."
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
                "Warning: Primary DiT initialization issue: "
                f"{init_status}",
                file=sys.stderr,
            )

        else:
            print(
                "Synapse primary DiT initialized successfully"
            )

        # ---------------------------------------------------------------------
        # Secondary DiT
        # ---------------------------------------------------------------------

        if config_path_2:
            print(
                "Initializing Synapse secondary DiT: "
                f"{config_path_2}..."
            )

            dit_handler_2 = AceStepHandler(
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
                    "Warning: Secondary DiT initialization issue: "
                    f"{init_status_2}",
                    file=sys.stderr,
                )

                init_status += (
                    "\n⚠️ Secondary DiT failed: "
                    f"{init_status_2}"
                )

            else:
                print(
                    "Synapse secondary DiT initialized successfully"
                )

                init_status += (
                    "\n✅ Secondary DiT: "
                    f"{config_path_2}"
                )

        # ---------------------------------------------------------------------
        # Composer LM
        # ---------------------------------------------------------------------

        checkpoint_dir = (
            dit_handler._get_checkpoint_dir()
        )

        print(
            f"Initializing Synapse Composer LM: {lm_model_path}..."
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
                "Synapse Composer LM initialized successfully"
            )

            init_status += (
                f"\n{lm_status}"
            )

        else:
            print(
                "Warning: Synapse Composer LM initialization failed: "
                f"{lm_status}",
                file=sys.stderr,
            )

            init_status += (
                f"\n{lm_status}"
            )

    # -------------------------------------------------------------------------
    # Available models
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # UI initialization parameters
    # -------------------------------------------------------------------------

    init_params = {
        # Synapse metadata
        "product_name": SYNAPSE_PRODUCT_NAME,
        "version": SYNAPSE_VERSION,

        # Service state
        "pre_initialized": True,
        "service_mode": True,

        # Checkpoints
        "checkpoint": None,
        "config_path": config_path,
        "config_path_2": (
            config_path_2
            if config_path_2
            else None
        ),

        # Runtime
        "device": device,

        # Composer
        "init_llm": True,
        "lm_model_path": lm_model_path,
        "backend": backend,

        # Performance
        "use_flash_attention": use_flash_attention,
        "offload_to_cpu": auto_offload,
        "offload_dit_to_cpu": False,

        # Initialization state
        "init_status": init_status,
        "enable_generate": enable_generate,

        # Handlers
        "dit_handler": dit_handler,
        "dit_handler_2": dit_handler_2,
        "available_dit_models": available_dit_models,
        "llm_handler": llm_handler,

        # UI
        "language": "en",

        # Storage
        "persistent_storage_path": persistent_storage_path,

        # Development
        "debug_ui": debug_ui,

        # Synapse repository metadata.
        #
        # These become useful as we update the downstream handler and UI.
        "synapse_model_repos": {
            "main": SYNAPSE_MAIN_MODEL_REPO,
            "xl_turbo": SYNAPSE_XL_TURBO_REPO,
            "xl_base": SYNAPSE_XL_BASE_REPO,
            "xl_sft": SYNAPSE_XL_SFT_REPO,
            "composer_4b": SYNAPSE_COMPOSER_4B_REPO,
            "composer_06b": SYNAPSE_COMPOSER_06B_REPO,
            "captioner": SYNAPSE_CAPTIONER_REPO,
            "transcriber": SYNAPSE_TRANSCRIBER_REPO,
            "vae": SYNAPSE_VAE_REPO,
        },
    }

    print("=" * 60)
    print("Synapse Music service initialization completed")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Gradio
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Launch
    # -------------------------------------------------------------------------

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