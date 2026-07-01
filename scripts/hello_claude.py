"""Smoke test: confirm Anthropic API works."""
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()  # reads ANTHROPIC_API_KEY from env

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=200,
    messages=[
        {
            "role": "user",
            "content": "In one sentence: what is contrast-enhanced spectral mammography?",
        }
    ],
)

print(response.content[0].text)
print(f"\nTokens used: in={response.usage.input_tokens}, out={response.usage.output_tokens}")