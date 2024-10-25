import gradio as gr
import logging
from UIFlux.ui_flux import UIFlux
from UIFlux.ui_flux_upscaler import UIFluxUpscaler
from Utils.utils import get_cpu_info, get_memory_info, get_gpu_info, get_filesystem_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

css="""
body, gradio-app {
    background: #243B55 !important;  /* fallback for old browsers */
    background: -webkit-linear-gradient(to top, #141E30, #243B55) !important;  /* Chrome 10-25, Safari 5.1-6 */
    background: linear-gradient(to top, #141E30, #243B55) !important; /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */

}

.prose { 
    font-size: var(--text-sd) !important; 
}
"""

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
        with gr.Blocks(
            theme="saq1b/gradio-theme",
            css=css,
        ) as app:
            gr.HTML('<h1 style="text-align: center; margin:1rem;">AI Gradio Lab</h1>')
            with gr.Accordion("ℹ️ System Information", open=False):
                with gr.Row():
                    with gr.Column(scale=1):
                        cpuinfo_output = gr.HTML(
                            value=get_cpu_info(),
                        )
                    with gr.Column(scale=1):
                        memoryinfo_output = gr.HTML(
                            value=get_memory_info(),
                        )
                with gr.Row():
                    with gr.Column(scale=1):
                        gpuinfo_output = gr.HTML(
                            value=get_gpu_info(),
                        )
                    with gr.Column(scale=1):
                        filesysteminfo_output = gr.HTML(
                            value=get_filesystem_info(),
                        )
            
                timer = gr.Timer(5)
                def refresh_state():
                    return [
                        get_cpu_info(),
                        get_memory_info(),
                        get_gpu_info(),
                        get_filesystem_info(),
                    ]
            
                timer.tick(
                    fn=refresh_state, 
                    outputs=[
                        cpuinfo_output,
                        memoryinfo_output,
                        gpuinfo_output,
                        filesysteminfo_output,
                    ]
                )
 
            with gr.Tab("Text2Image"):
                ui_flux.render()
            with gr.Tab("Flux.1 Upscaler"):
                ui_flux_upscaler.render()

        logging.info(f"Launching the Gradio AIO Application on {server_name}:{server_port}.")
    
        # Launch the application
        app.launch(server_name=server_name, server_port=server_port)
    
    except Exception as e:
        # Log the error with traceback for better debugging
        logging.error(f"An error occurred while running the Gradio app: {e}", exc_info=True)
    
    finally:
        logging.info("Shutting down the Gradio application.")

if __name__ == "__main__":
    run_app(server_name='0.0.0.0', server_port=7860)