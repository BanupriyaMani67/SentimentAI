"""Small helper that talks to the FastAPI backend (kept separate so it is easy to test)."""
import requests


class ApiError(Exception):
    """Raised with a user-friendly message when the API call fails."""


def predict_sentiment(text: str, api_url: str, timeout: float = 60) -> dict:
    url = api_url.rstrip("/") + "/predict"
    try:
        response = requests.post(url, json={"text": text}, timeout=timeout)
    except requests.exceptions.Timeout:
        raise ApiError("The server took too long to respond. If it was asleep, wait a minute and try again.")
    except requests.exceptions.ConnectionError:
        raise ApiError(f"Cannot reach the backend at {api_url}. Is it running?")
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Request failed: {exc}")

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            raise ApiError("The server returned an unreadable response.")
    if response.status_code == 422:
        raise ApiError(_detail(response, "Invalid input."))
    if response.status_code == 503:
        raise ApiError("The model is not ready yet. Please try again shortly.")
    if response.status_code >= 500:
        raise ApiError("The server had an internal error. Please try again later.")
    raise ApiError(_detail(response, f"Unexpected response (HTTP {response.status_code})."))


def _detail(response, default: str) -> str:
    try:
        return str(response.json().get("detail", default))
    except Exception:
        return default
