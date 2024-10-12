import gradio as gr

from gradio_demo import build_app

if __name__ == "__main__":
    app = build_app()
    #app.queue().launch(server_name="0.0.0.0", server_port=4700, share=True)
    app.launch(share=False)