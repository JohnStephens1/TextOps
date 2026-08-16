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


def main() -> None:
    app = get_app()
    app.launch()


if __name__ == "__main__":
    main()
