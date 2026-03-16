"""Entry point for kizuna-voice-designer."""

import os
import sys
from pathlib import Path


def main():
    # Use the downloader to set up paths and ensure models
    from kizuna_voice_designer.downloader import setup_paths

    setup_paths()

    # Import the Gradio app module from within the package
    from kizuna_voice_designer.gradio_app import build_interface

    port = int(os.environ.get("GRADIO_SERVER_PORT", "8811"))
    share = os.environ.get("SHARE", "false").lower() in ("1", "true", "yes")
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)


if __name__ == "__main__":
    main()
