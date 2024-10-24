import gradio as gr
import logging
from UIFlux.ui_flux import UIFlux
from UIFlux.ui_flux_upscaler import UIFluxUpscaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_app(server_name: str = '0.0.0.0', server_port: int = 7860):
    """
    Initializes and launches the Gradio web application using provided UI classes.
    
    Parameters:
        server_name (str): The address where the Gradio server will be hosted.
        server_port (int): The port on which the Gradio server will be accessible.
    """
    try:
        logging.info("Initializing interfaces.")
        
        # Initialize the UI components
        ui_flux = UIFlux().initialize().interface()
        ui_flux_upscaler = UIFluxUpscaler().initialize().interface()

        # Create a tabbed interface using the initialized UIs
        root = gr.TabbedInterface(
            interface_list=[
                ui_flux,
                ui_flux_upscaler
            ],
            tab_names=[
                "Flux.1 LoRA Bulk",
                "Flux.1 Upscaler"
            ],
            title="Gradio AIO Application",
            theme="saq1b/gradio-theme",
            css="""
            .prose { font-size: var(--text-sd) !important; }
            """
        )

        logging.info(f"Launching the Gradio AIO Application on {server_name}:{server_port}.")
        
        # Launch the application
        root.launch(server_name=server_name, server_port=server_port)
    
    except Exception as e:
        # Log the error with traceback for better debugging
        logging.error(f"An error occurred while running the Gradio app: {e}", exc_info=True)
    
    finally:
        logging.info("Shutting down the Gradio application.")

if __name__ == "__main__":
    run_app(server_name='0.0.0.0', server_port=7860)