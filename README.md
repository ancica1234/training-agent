# Training Progress Agent

An AI-powered agentic system that evaluates student training progress, generates remediation plans, and requires human supervisor approval before finalizing.

## Demo

![Agent Flow](https://img.shields.io/badge/LangGraph-Agent-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![Groq](https://img.shields.io/badge/Groq-LLM-green)

## Features

- Multi-tool AI agent built with LangGraph
- Human-in-the-loop approval for remediation plans
- Persistent memory per student using MemorySaver
- Web UI built with Streamlit
- Safety step limit to prevent infinite loops
- Powered by Groq LLM (llama-3.3-70b)

## How It Works

The agent follows this flow for each student:

```
START
  → agent  (LLM reasons about what to do)
  → tools  (fetches schedule, history, remediation options)
  → agent  (LLM evaluates progress)
  → human_approval  (supervisor approves or rejects)
  → agent  (LLM finalizes response)
  → END
```

Students who are ON TRACK skip the remediation and approval steps entirely.

## Agent Tools

| Tool | Description |
|------|-------------|
| `get_class_schedule` | Fetches all scheduled training events and dates for a class |
| `get_student_history` | Retrieves completed and incomplete events plus workdays behind |
| `get_remediation_options` | Generates remediation strategies for students who are behind |

## Tech Stack

| Technology | Purpose |
|------------|----------|
| LangGraph | Agent orchestration and graph workflow |
| LangChain | Tool definitions and LLM integration |
| Groq LLM | Fast inference with llama-3.3-70b |
| Streamlit | Web UI |
| MemorySaver | Persistent conversation memory per student |

## Installation

### Prerequisites
- Python 3.11+
- Groq API key (free at console.groq.com)

### Setup

```bash
# Clone the repo
git clone https://github.com/ancica1234/training-agent.git
cd training-agent

# Install dependencies
pip install langgraph langchain-core langchain-groq streamlit

# Set your Groq API key
export GROQ_API_KEY=your_key_here

# Run the web app
python3 -m streamlit run app.py
```

## Usage

1. Open the web app in your browser at http://localhost:8501
2. Enter your Groq API key in the sidebar
3. Select a student from the dropdown
4. Click Run Agent
5. Watch the agent fetch data, evaluate progress, and generate recommendations
6. If the student is behind, approve or reject the remediation plan

## Project Structure

```
training-agent/
├── training_agent.py   # LangGraph agent, tools, graph definition
├── app.py             # Streamlit web UI
└── README.md          # This file
```

## Author

Built as a demonstration of agentic AI with human-in-the-loop workflows.
