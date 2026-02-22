
# Project Objective

This repo aims to build different usecases for Langgraph based AI agents.

## Description

This repo shows different examples of AI agents
* 	[Blog Writing Agent](#Blog-Writing-Agent)
* 	[RAG health policy](#RAG_for_health_insurance_policy)

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

### Blog Writing Agent
This is an AI blog writer. Outcome of this includes
* a text blog on the given topic
* an image suitable for posting on Medium
* an image for instagram posting summarising the blog

Workflow of the graph is
```bash
User Topic -> Domain Agent -> Web Research Agent -> Writer Agent <-> Editor Agent-> Image Prompt Agent -> Image Generation Agent -> Publisher
```
### RAG for health insurance policy
This is a Q&A RAG which answers to user questions based on the knowledge provided (pdfs). If answer cannot be found, replies with "I could not find this in the policy document."

Workflow of the graph is 
```bash
PDF -> Chunk -> Embed (Gemini) -> Chroma (Vector DB)
Query -> Retrieve -> Generate Answer -> Verify Answer
```
#### Additional features
- [x] (Enable converstaional chat - Ex. follow-up to previous question)
- [ ] (Token optimisation for long history chats)
- [x] (Enable direct text search - Ex. search by clause number)
      
#### Enabling conversional chat

Change1 : Detection of topic change to handle context bleeding (classify)
- Method1: Using llm
- Method2: Using cosine similarity of last user question and new question

Change 2 : Question rewriting:
- Rewrite follow-up into standalone question.
- This improves the quality of embedding-based retrieval as the standalone question has better information than a follow-up question.
- Workflow of the modfied graph is 
```bash
PDF -> Chunk -> Embed (Gemini) -> Chroma (Vector DB)
Query -> classify if follow-up -> Rewrite Question -> Retrieve -> Generate Answer -> Verify Answer
```

#### Enable direct text (lexical) search

The embedding serach is quite good in semantic searching of documents.
But, this may fail when actual text search - such as clause number, which is common in health policies or legal documents.
To Handle this a ensembed retriever is designed which can handle bot semantic and lexical search capabilities
