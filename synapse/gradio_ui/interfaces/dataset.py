"""
Synapse Music V12 - Gradio UI Dataset Section

Contains the Synapse Music dataset explorer component definitions used for
dataset inspection, training-data review, source/reference audio inspection,
and transferring dataset metadata into the generation interface.
"""

import gradio as gr


def create_dataset_section(dataset_handler) -> dict:
    """
    Create the Synapse Music V12 dataset explorer section.

    Args:
        dataset_handler:
            Synapse dataset handler responsible for importing, searching,
            retrieving, and processing dataset items.

    Returns:
        dict:
            Dictionary containing all dataset explorer Gradio components.
    """

    with gr.Accordion(
        "📊 Synapse Dataset Explorer",
        open=False,
        visible=False,
    ):
        # =====================================================================
        # Dataset Selection / Search
        # =====================================================================

        with gr.Row(equal_height=True):
            dataset_type = gr.Dropdown(
                choices=[
                    "train",
                    "test",
                ],
                value="train",
                label="Synapse Dataset",
                info="Choose a dataset split to explore",
                scale=2,
            )

            import_dataset_btn = gr.Button(
                "📥 Import Dataset",
                variant="primary",
                scale=1,
            )

            search_type = gr.Dropdown(
                choices=[
                    "keys",
                    "idx",
                    "random",
                ],
                value="random",
                label="Search Type",
                info="Choose how to find dataset items",
                scale=1,
            )

            search_value = gr.Textbox(
                label="Search Value",
                placeholder=(
                    "Enter a key or index "
                    "(leave empty for random)"
                ),
                info=(
                    "Keys: exact match. "
                    "Index: 0 to dataset size - 1."
                ),
                scale=2,
            )

        # =====================================================================
        # Generation Instruction
        # =====================================================================

        instruction_display = gr.Textbox(
            label="📝 Generation Instruction",
            interactive=False,
            placeholder="No generation instruction available",
            lines=1,
        )

        # =====================================================================
        # Repaint Visualization
        # =====================================================================

        repaint_viz_plot = gr.Plot(
            label="Repaint / Audio Region Visualization"
        )

        # =====================================================================
        # Dataset Metadata
        # =====================================================================

        with gr.Accordion(
            "📋 Dataset Item Metadata",
            open=False,
        ):
            item_info_json = gr.Code(
                label="Complete Item Metadata (JSON)",
                language="json",
                interactive=False,
                lines=15,
            )

        # =====================================================================
        # Source Audio
        # =====================================================================

        with gr.Row(equal_height=True):
            item_src_audio = gr.Audio(
                label="Source Audio",
                type="filepath",
                interactive=False,
                scale=8,
            )

            get_item_btn = gr.Button(
                "🔍 Load Dataset Item",
                variant="secondary",
                interactive=False,
                scale=2,
            )

        # =====================================================================
        # Target / Reference Audio
        # =====================================================================

        with gr.Row(equal_height=True):
            item_target_audio = gr.Audio(
                label="Target Audio",
                type="filepath",
                interactive=False,
                scale=8,
            )

            item_refer_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                interactive=False,
                scale=2,
            )

        # =====================================================================
        # Dataset Source Controls
        # =====================================================================

        with gr.Row():
            use_src_checkbox = gr.Checkbox(
                label="Use Dataset Source Audio",
                value=True,
                info=(
                    "Use the selected dataset item's source audio "
                    "as the generation source"
                ),
            )

        # =====================================================================
        # Dataset Status
        # =====================================================================

        data_status = gr.Textbox(
            label="📊 Synapse Dataset Status",
            interactive=False,
            value="❌ No dataset imported",
        )

        # =====================================================================
        # Transfer Dataset Item to Generator
        # =====================================================================

        auto_fill_btn = gr.Button(
            "📋 Send Dataset Item to Synapse Generator",
            variant="primary",
        )

    # =========================================================================
    # Component Registry
    #
    # Keep these keys stable because other Synapse UI/event modules use them
    # to register callbacks and transfer dataset information into generation.
    # =========================================================================

    return {
        "dataset_type": dataset_type,
        "import_dataset_btn": import_dataset_btn,
        "search_type": search_type,
        "search_value": search_value,
        "instruction_display": instruction_display,
        "repaint_viz_plot": repaint_viz_plot,
        "item_info_json": item_info_json,
        "item_src_audio": item_src_audio,
        "get_item_btn": get_item_btn,
        "item_target_audio": item_target_audio,
        "item_refer_audio": item_refer_audio,
        "use_src_checkbox": use_src_checkbox,
        "data_status": data_status,
        "auto_fill_btn": auto_fill_btn,
    }