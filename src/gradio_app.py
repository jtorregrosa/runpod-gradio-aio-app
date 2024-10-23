import gradio as gr
import json

from UIFlux.ui_flux import UIFlux
from UIFlux.ui_flux_upscaler import UIFluxUpscaler

def run_app():
    # Create an instance of the selected UI
    ui_flux = UIFlux().initialize().interface()
    ui_flux_upscaler = UIFluxUpscaler().initialize().interface()

    root = gr.TabbedInterface([ui_flux, ui_flux_upscaler], ["Flux.1 LoRA Bulk", "Flux.1 Upscaler"])

    # Launch the Gradio app
    root.launch(server_name='0.0.0.0')

if __name__ == "__main__":
    run_app()