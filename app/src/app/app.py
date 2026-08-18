import gradio as gr

from .prediction import handle_prediction


def get_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        # TODO add placeholder text
        user_input_title = gr.Textbox(label="Title")
        user_input_description = gr.Textbox(label="Description")

        button = gr.Button("Submit")
        out = gr.Textbox(label="Result")

        button.click(
            fn=handle_prediction,
            inputs=[user_input_title, user_input_description],
            outputs=out,
        )

    return demo


# demo here for hot reloading functionality
if __name__ == "__main__":
    demo = get_demo()
    demo.launch()
