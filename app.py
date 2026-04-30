"""
Entry point for Hugging Face Spaces (Gradio SDK).

HF Spaces looks for an app.py at the repo root and runs it. We just import
the existing Gradio app from app/gradio_app.py — that file is the source of
truth and stays usable for local development too.
"""

from app.gradio_app import demo


if __name__ == "__main__":
    demo.launch()
