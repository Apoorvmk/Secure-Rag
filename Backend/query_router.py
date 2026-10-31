import json
from groq import Groq


def route_query(client, model, message, history):
    prompt = f"""
You are a query router for an enterprise AI assistant.

Decide whether the user's message requires information
from internal company documents.

Return ONLY JSON:
{{"route": "general"}}
or
{{"route": "retrieval"}}

Conversation history:
{json.dumps(history[-6:])}

User message:
{message}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)

    if result.get("route") not in ["general", "retrieval"]:
        raise ValueError("Invalid route")

    return result["route"]