# LangGraph Agent Tutorial – Simple and Complex AI Agents

This repository contains a tutorial project for building two different types of AI agents using [LangGraph](https://github.com/techwithtim/LangGraph-Tutorial) and the Groq API:

## 🧠 Agents Overview

### 1. Simple Agent – Chatbot
A basic conversational AI agent that uses:
- **Groq API** for fast inference
- **LangGraph** for managing the flow

It acts like a straightforward chatbot for general-purpose conversations.

### 2. Complex Agent – Emotional & Logical Assistant
An advanced AI agent that routes user prompts to either:
- An **Emotional Assistant**, or
- A **Logical Assistant**

The routing is done based on the nature of the input, allowing for more context-aware and appropriate responses.

## 📦 Features
- Groq integration for LLM access
- LangGraph to manage agent behavior and flow
- Prompt routing logic in the complex agent
- Inspired by [TechWithTim's LangGraph tutorial](https://github.com/techwithtim/LangGraph-Tutorial)

## 🚀 Getting Started

### Requirements
- Python 3.8+
- `groq` SDK
- `langgraph`
- `langchain` (optional for prompt formatting)

### Installation

```bash
pip install -r requirements.txt
