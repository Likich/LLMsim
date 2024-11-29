# LLMsim

## Overview

**LLMsim** is a Python-based framework for simulating discussions among large language models (LLMs). It supports thematic analysis, coding of textual data, and collaborative discussion between multiple LLMs to reach consensus on thematic codes. This repository explores how different LLMs process, discuss, and agree on themes extracted from text.

### Key Features
- Simulates multiple LLMs discussing and debating their thematic codes.
- Supports both API-based (e.g., OpenAI's GPT-4 and GPT-3.5) and open-source LLMs (e.g., Meta's Llama 2).
- Incorporates role-playing scenarios (e.g., leader, intern, neutral) for studying power dynamics in discussions.
- Facilitates experiments in thematic coding, role-based discussions, and group decision-making.

---

## Repository Structure

```plaintext
.
├── LLMsimu.py                         # Main Python script containing all functionality
├── keys.txt                           # File for storing API keys for various models
├── text.txt                           # Example transcript file to be analyzed
├── discussion_transcript_experiment_1.json   # Output for Experiment 1
├── discussion_transcript_experiment_2.json   # Output for Experiment 2
├── thematic_analysis_results.json            # Output of thematic analysis
├── README.md                          # Project documentation
```


# How It Works

### Step 1: Thematic Coding
Each LLM analyzes the transcript and generates thematic codes independently.

### Step 2: Discussion Simulation
The LLMs discuss their codes and attempt to reach a consensus on a final set of themes. The discussion is facilitated through simulated rounds, where each LLM contributes iteratively.

### Role-Based Discussions
Roles such as "leader," "intern," and "neutral" can be assigned to LLMs to simulate different levels of influence and explore power dynamics.

---

# Requirements

### Dependencies
Install the required Python libraries:

```
pip install openai transformers torch langchain faiss-cpu ```


markdown
Copy code
# API Keys

- **OpenAI GPT models (e.g., GPT-4, GPT-3.5):** Obtain API keys from OpenAI.
- **Open-source models (e.g., Llama 2):** Ensure access to the Hugging Face repository or API.

---

# Usage

### 1. Add API Keys
Create a `keys.txt` file to store your API keys. Use the format:

```plaintext
gpt-4:<your_gpt4_api_key>
gpt-3.5-turbo:<your_gpt35_api_key>
llama:<your_llama_api_key>
