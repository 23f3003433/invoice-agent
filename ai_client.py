
import json
import os

from openai import OpenAI

def get_ai_client():
    """Create an AI Pipe client using environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://aipipe.org/openai/v1"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def ask_ai_for_json(system_prompt, user_prompt):
    """Ask the model for a JSON response."""
    client = get_ai_client()

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The AI returned an empty response.")

    result = json.loads(content)

    if not isinstance(result, dict):
        raise ValueError("The AI response must be a JSON object.")

    return result
