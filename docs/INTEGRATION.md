# Integration Guide

OneMem is packaged as a Python library first and a CLI second.

## Install Locally

From another Python project:

```bash
pip install -e /Users/nezir/Documents/Projects/onemem
```

From a Git repository later:

```bash
pip install git+https://github.com/YOUR_USER/onemem.git
```

## Minimal Agent Integration

```python
from onemem import MemoryRuntime

memory = MemoryRuntime("memory")
memory.init()

def answer(user_message: str) -> str:
    memory_context = memory.retrieve(user_message, limit=8).context()
    prompt = f"""Use this memory when relevant:
{memory_context}

User:
{user_message}
"""

    assistant_answer = call_your_llm(prompt)

    memory.capture(
        f"User asked: {user_message}\nAssistant answered: {assistant_answer}",
        source="my-agent",
        session="chat",
    )
    memory.flush()
    return assistant_answer
```

## Stable V1 API

```python
from onemem import MemoryRuntime

memory = MemoryRuntime("memory")
memory.init()
memory.capture("Observation text", source="agent", session="chat")
memory.flush()
result = memory.retrieve("question", limit=8)
context = result.context()
memory.invalidate("fact_id", reason="superseded")
memory.rebuild_index()
```

## Adapter Rule

Keep framework-specific code outside the core.

Good:

```text
onemem.adapters.langchain
onemem.adapters.fastapi
your_project/onemem_adapter.py
```

Avoid:

```text
runtime.py importing LangChain, FastAPI, CrewAI, or a specific LLM SDK
```

This keeps storage, retrieval, consolidation, and maintenance reusable across agents and chatbots.

## Optional Extras

The package declares optional dependency groups:

```bash
pip install "onemem[openai]"
pip install "onemem[server]"
pip install "onemem[dev]"
```

The current core does not require these extras. They are reserved for clean adapters and packaging workflows.

