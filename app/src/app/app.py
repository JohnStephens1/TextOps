import gradio as gr


def greet(name: str, intensity: str) -> str:
    return "Hello, " + name + "!" * int(intensity)


def get_app() -> gr.Interface:
    return gr.Interface(
        fn=greet,
        inputs=["text", "slider"],
        outputs=["text"],
        flagging_mode="never",
    )


# duplicate for run compatibility
def main() -> None:
    get_app().launch()


# app here for hot reloading functionality
if __name__ == "__main__":
    app = get_app()
    app.launch()
