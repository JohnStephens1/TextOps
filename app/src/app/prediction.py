import requests
from common.environment import API_URL


def get_response(title: str, description: str) -> requests.Response:
    return requests.post(
        f"{API_URL}/predict",
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
