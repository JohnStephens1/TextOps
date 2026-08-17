import gradio as gr
import requests

FASTAPI_URL = "http://localhost:8000/predict"


def get_response(title: str, description: str) -> requests.Response:
    return requests.post(
        FASTAPI_URL,
        json={"title": title, "description": description},
    )


def format_response(title: str, description: str, response: requests.Response) -> str:
    out_str = f"""
        Input:
        - title: {title}
        - description: {description}\n
    """

    if response.ok:
        result = response.json()

        out_str += f"""
            Output:
            - prediction: {result["label"]}
            - certainty: {f"{max(result['pred_proba']):.4f}"}

            - all possible labels: {", ".join(result["all_labels"])}
            - assigned probabilities: {", ".join([f"{x:.4f}" for x in result["pred_proba"]])}
        """
    else:
        # TODO log error
        out_str += f"""
            Error in response:
            - code: {response.status_code}
            - text: {response.text}
        """

    return out_str


def handle_prediction(title: str, description: str) -> str:
    response = get_response(title, description)
    out_str = format_response(title, description, response)

    return out_str


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
