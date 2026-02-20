
# Project Objective

This repo aims to build different usecases for Langgraph based AI agents.

## Description

This repo shows different examples of AI agents
* Blog writing
  
## Getting Started

### Setup

*   Create a virtual environment
*   Download this github repo
*   Install prerequesites from requirements.txt
*   Get your API keys for GOOGLE_API, SERPER_API and LANGCHAIN_API (for langsmith tracing)
*   Create a .env file in the respective application folder (ex. blog_writer_agent) and add your API_KEYS there

```python
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
SERPER_API_KEY=YOUR_SERPER_API_KEY
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=YOUR_LANGCHAIN_API_KEY
LANGCHAIN_PROJECT=LangGraph_Exp
```

### Executing program
Open a terminal in the respective application folder, activate the virtual environment and run below cmd
```bash
cd blog_writer_agent
python ./blog_writer_agent.py
```
## Applications

### Blog writing
This is an AI blog writer. Outcome of this includes
* a text blog on the given topic
* an image suitable for posting on Medium
* an image for instagram posting summarising the blog

Workflow of the graph is
```bash
User Topic -> Domain Agent -> Web Research Agent -> Writer Agent <-> Editor Agent-> Image Prompt Agent -> Image Generation Agent -> Publisher
```
