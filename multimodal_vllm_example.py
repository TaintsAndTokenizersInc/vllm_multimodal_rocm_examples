"""
An example showing how to use vLLM to serve multimodal models
and run online serving with OpenAI client.

This script demonstrates how to interact with multimodal models (text + image)
via the OpenAI-compatible API provided by vLLM. It supports:
- Text-only inference
- Single image input (via URL or base64 encoding)
- Multi-image input (multiple images in a single prompt)

The script uses the OpenAI Python client to send requests to a vLLM server
running a multimodal model (e.g., Gemma-3-27b-it or similar).

Environment variables:
- OPENAI_API_KEY: API key for authentication (if required by your vLLM setup)
- OPENAI_BASE_URL: Base URL of the vLLM server (e.g., http://localhost:8000/v1)
- OPENAI_MODEL: Default model name (optional, can be overridden via CLI)

Usage:
    python src/main.py --chat-type single-image
"""

import argparse
import base64
import os
import requests
from openai import OpenAI

# Create a client. API key is read from the OPENAI_API_KEY env var by default.
# You can also pass --api-key on the CLI if you prefer.
def make_client(api_key: str | None, base_url: str | None) -> OpenAI:
    """Create and return an OpenAI client instance with optional authentication and base URL."""
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def check_vllm_health(base_url: str) -> bool:
    """Check if the vLLM server is alive by calling the /health endpoint.

    Args:
        base_url: Base URL of the vLLM server (e.g., http://localhost:8000/v1)

    Returns:
        True if server is responsive, False otherwise.
    """
    try:
        # Remove trailing /v1 to get the root health endpoint
        health_url = f"{base_url.rstrip('/v1')}/health"
        print(f"Checking health of {health_url}")
        response = requests.get(health_url, timeout=10)
        print(f"Response Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ vLLM server is healthy and responsive.")
            return True
        else:
            print("❌ vLLM health check failed:", response.status_code, response.text)
            return False
    except requests.RequestException as e:
        print("❌ Error contacting vLLM health endpoint:", e)
        exit(1)
    return False


def encode_base64_content_from_url(content_url: str) -> str:
    """Download content from a URL and return a base64-encoded string.

    Args:
        content_url: URL of the image or file to download

    Returns:
        Base64-encoded string of the downloaded content
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    with requests.get(content_url, headers=headers) as response:
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")


# ---------- Demo runners ----------

def run_text_only(client: OpenAI, model: str) -> None:
    """Run a text-only inference with the model.

    Args:
        client: OpenAI client instance
        model: Model name to use for inference
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the capital of France?"}],
        model=model,
        max_tokens=128,
    )
    print(chat_completion.choices)
    result = chat_completion.choices[0].message.content
    print("Chat completion output:\n", result)


def run_single_image(client: OpenAI, model: str) -> None:
    """Run inference with a single image using both URL and base64 encoding.

    Args:
        client: OpenAI client instance
        model: Model name to use for inference
    """
    print("Running single image inference")
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # Use image URL in the payload
    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )
    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output from image url:\n", result)

    # Use base64 encoded image in the payload
    image_base64 = encode_base64_content_from_url(image_url)
    chat_completion_from_base64 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )
    result = chat_completion_from_base64.choices[0].message.content
    print("\nChat completion output from base64 encoded image:\n", result)


# Multi-image input inference
def run_multi_image(client: OpenAI, model: str) -> None:
    """Run inference with multiple images in a single prompt.

    Args:
        client: OpenAI client instance
        model: Model name to use for inference
    """
    print("Running multi image inference")
    image_url_duck = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzyżowka_w_wodzie_%28samiec%29.jpg"
    image_url_lion = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"

    chat_completion_from_url = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What are the animals in these images?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_duck},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_lion},
                    },
                ],
            }
        ],
        model=model,
        max_completion_tokens=64,
    )
    result = chat_completion_from_url.choices[0].message.content
    print("Chat completion output:\n", result)


EXAMPLES = {
    "text-only": run_text_only,
    "single-image": run_single_image,
    "multi-image": run_multi_image,
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Demo using the OpenAI client for multimodal prompts in vLLM."
    )
    parser.add_argument(
        "--chat-type", "-c",
        type=str,
        default="single-image",
        choices=list(EXAMPLES.keys()),
        help="Conversation type (text-only, single-image, multi-image).",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "/google/gemma-3-27b-it"),
        help="vLLM model to use",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key (otherwise read from OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL"),
        help="Base vLLM URL",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    # Load environment variables from .env file
    load_dotenv()

    args = parse_args()
    client = make_client(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
    base_url = os.environ.get("OPENAI_BASE_URL")
    check_vllm_health(base_url)
    runner = EXAMPLES[args.chat_type]
    runner(client, args.model)