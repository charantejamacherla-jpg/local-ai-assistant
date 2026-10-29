# Laya — Local AI Assistant

A local-first AI assistant built with Python, Ollama, semantic retrieval, conversational memory, and tool execution.

Laya runs its language model locally, without requiring a cloud LLM API.

## Features

- **Local AI conversations:** Uses Qwen through Ollama.
- **Semantic search (RAG):** Retrieves relevant information from a local knowledge base.
- **Answerability filtering:** Rejects retrieved documents that do not appear to contain the requested answer.
- **Conversation memory:** Maintains a limited history within the active session.
- **Tool execution:** Supports calculator operations through a tool-selection system.
- **Request classification:** Uses Laya Router to classify requests by department, urgency, and need for human assistance.
- **Human assistance guidance:** Identifies certain requests for human support without claiming to contact an actual agent.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Local language model | Qwen 2.5 1.5B |
| Model runtime | Ollama |
| Embedding model | nomic-embed-text |
| Request classification | Laya Router |
| Semantic retrieval | Local vector search |
| Testing | Python unittest |

## Installation

### 1. Prerequisites

Install Python, Git, and Ollama.

### 2. Clone the repository

```bash
git clone https://github.com/charantejamacherla-jpg/local-ai-assistant.git
cd local-ai-assistant
```

### 3. Create a virtual environment

On Windows CMD:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

### 4. Install Python dependencies

```bat
pip install -r requirements.txt
```

### 5. Download the required models

```bat
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the assistant.

## Run Laya

```bat
python app.py
```

The default language model is `qwen2.5:1.5b`.

You can override it through the `OLLAMA_MODEL` environment variable.

## Testing

Run the automated unit tests:

```bat
python -m unittest test_phase9 -v
```

Run the semantic retrieval demonstration:

```bat
python test_rag.py
```

Run the answerability verification:

```bat
python test_answerability.py
```

## Architecture

User input flows through a local processing pipeline:

1. Tool selection checks for supported operations.
2. Laya Router classifies the request.
3. Routing logic determines whether human assistance should be recommended.
4. Semantic retrieval searches the local knowledge base.
5. Answerability filtering evaluates retrieved information.
6. Conversation memory supplies recent context.
7. Ollama generates the assistant's response.

## Known Limitations

- The project is a portfolio prototype, not a production customer-support system.
- Human assistance routing does not contact a real human agent.
- The routing safeguard recognizes only a limited set of explicit human-assistance phrases.
- Laya Router may emit warnings about uncalibrated classification confidence.
- Retrieval similarity does not guarantee factual correctness.
- Conversation memory is limited to the current session.
- The local language model may produce incorrect answers.
- The project currently provides a command-line interface.

## Development Approach

**Research → decide → build → test → improve.**

## Project Status

Version 1.0 — Portfolio MVP release candidate.
