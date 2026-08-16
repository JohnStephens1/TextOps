import gradio as gr
import requests

FASTAPI_URL = "http://localhost:8000/predict"


def send_to_api(title: str, description: str) -> dict[str, str | float]:
    response = requests.post(
        FASTAPI_URL,
        # json={"title": 5, "description": description},
        json={"title": title, "description": description},
    )

    if response.ok:
        return response.json()
    else:
        print(f"""
            Error in response:
            code: {response.status_code}
            text: {response.text}
        """)

        return {"title": "error", "description": 0.5}


def get_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        user_input_title = gr.Textbox(label="Title")
        user_input_description = gr.Textbox(label="Description")

        button = gr.Button("Submit")
        output = gr.Textbox(label="Response")

        button.click(
            fn=send_to_api,
            inputs=[user_input_title, user_input_description],
            outputs=output,
        )

    return demo


# duplicate for run compatibility
def main() -> None:
    get_demo().launch()


# demo here for hot reloading functionality
if __name__ == "__main__":
    demo = get_demo()
    demo.launch()
