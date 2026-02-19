import os
import json
from typing import TypedDict, List, Optional

import re
import json
import base64
from datetime import datetime
import yaml


from dotenv import load_dotenv, find_dotenv
from serpapi import GoogleSearch
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langsmith import traceable



import requests
from langgraph.graph import StateGraph, END

#from langsmith import traceable

# =========================================================
# Load Environment
# =========================================================
load_dotenv(find_dotenv())

SERPAPI_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Agent_blog_writer"

if not SERPAPI_KEY or not GOOGLE_API_KEY:
    raise ValueError("Missing SERPAPI_KEY or GOOGLE_API_KEY in environment")

genai.configure(api_key=GOOGLE_API_KEY)
text_model_name = "gemini-2.5-flash-lite"
image_model_name = "gemini-2.5-flash-image"#"gemini-3-pro-image-preview" #("gemini-2.5-flash-image")

# =========================================================
# State Definition
# =========================================================

class BlogState(TypedDict):
    topic: str
    domain: str
    research_sources: List[dict]
    research_summary: str
    draft_blog: str
    editor_feedback: Optional[str]
    approved: bool
    final_blog: Optional[str]
    citations: List[str]
    iteration_count: int
    image_concepts: Optional[str]
    image_prompt_medium: Optional[str]
    image_prompt_instagram: Optional[str]
    image_url_medium: Optional[str]
    image_url_instagram: Optional[str]

# =========================================================
# Domain role selection
# =========================================================
SUPPORTED_DOMAINS = [
    "finance",
    "healthcare",
    "technology",
    "general"
]


# =========================================================
# Domain based role description
# =========================================================
def load_domain_config(domain: str):
    with open("domain_roles.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config.get(domain, config["general"])

# =========================================================
# Gemini Helper
# =========================================================

def generate_text(prompt: str, temperature: float = 0.3) -> str:
    llm = ChatGoogleGenerativeAI(
        model=text_model_name,   # use stable model
        temperature=temperature,
        top_p=0.9,
        max_output_tokens=2048,
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content

"""
def generate_text(prompt: str, temperature: float = 0.3) -> str: 
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        generation_config={
            "temperature": temperature,
            "top_p": 0.9,
            "max_output_tokens": 2048,
        },
    )
    response = model.generate_content(prompt)
    return response.text.strip()
"""

# =========================================================
# Domain Classifier Node (LLM-Based)
# =========================================================

def domain_classifier_agent(state: BlogState):

    topic = state["topic"]

    prompt = f"""
    Classify the following blog topic into ONE of these domains:

    {SUPPORTED_DOMAINS}

    Respond ONLY with the domain name.
    No explanation.

    Topic:
    {topic}
    """

    llm = ChatGoogleGenerativeAI(
        model=text_model_name,   # use stable model
        temperature=0.0
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    predicted_domain = response.content.strip().lower()

    # Safety fallback
    if predicted_domain not in SUPPORTED_DOMAINS:
        predicted_domain = "general"

    print(f"Detected domain: {predicted_domain}")

    return {"domain": predicted_domain}


# =========================================================
# Research Agent (SerpAPI + Gemini Summary)
# =========================================================
#@traceable
def research_agent(state: BlogState):

    topic = state["topic"]

    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPAPI_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "q": topic,
        "num": 5
    }

    response = requests.post(url, headers=headers, json=payload)
    results = response.json()
    domain = state["domain"]

    sources = []
    citations = []

    for r in results.get("organic", []):
        link = r.get("link")
        snippet = r.get("snippet")

        if link and snippet:
            sources.append({
                "title": r.get("title"),
                "link": link,
                "snippet": snippet
            })
            citations.append(link)
    
    summary_prompt = f"""
    You are a certifide data research analyst in {domain} domain. 
    The final objective is to use this research to provide insights on the topic in a simple layman terms.

    Below are web search results.

    Extract:
    - Key factual insights
    - Important statistics
    - Expert claims
    - Industry data

    Do NOT speculate.
    If insufficient data exists, explicitly state: "Insufficient evidence."

    Sources:
    {sources}
    """


    if not sources:
        raise ValueError("Research agent found no sources.")

    summary = generate_text(summary_prompt, temperature=0.2)

    return {
        "research_sources": sources,
        "research_summary": summary,
        "citations": citations
    }

# =========================================================
# Writer Agent
# =========================================================

def writer_agent(state: BlogState):

    feedback = state.get("editor_feedback") or ""

    domain = state.get("domain", "general")
    config = load_domain_config(domain)

    writer_config = config["writer"]

    constraints_text = "\n".join(
        f"- {c}" for c in writer_config["constraints"]
    )

    task_text = "\n".join(
        f"- {c}" for c in writer_config["task"]
    )

    prompt = f"""
        You are a {writer_config['role']}.

        Tone:
        {writer_config['tone']}

        Constraints:
        {constraints_text}

        Topic:
        {state['topic']}

        Research Summary:
        {state['research_summary']}

        Available Source Links:
        {state['citations']}

        Editor Feedback (if any):
        {feedback}

        TASK:
        {task_text}
        """

    blog = generate_text(prompt, temperature=0.5)

    return {
        "draft_blog": blog,
        "iteration_count": state["iteration_count"] + 1
    }


def safe_parse_json(text: str):
    # Remove markdown ```json fences
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    return json.loads(text)

# =========================================================
# Editor Agent (Fact Checker)
# =========================================================

def extract_json(text: str):
    """
    Extract JSON from model response safely.
    """
    # Remove markdown code fences if present
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    # Find first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError("No JSON object found in response")


def editor_agent(state: BlogState):

    prompt = f"""
    You are a senior fact-checking editor.

    Research Summary:
    {state['research_summary']}

    Draft Blog:
    {state['draft_blog']}

    TASK:
    - Identify unsupported claims
    - Detect hallucinations
    - Verify logical consistency

    IMPORTANT:
    Respond ONLY with raw JSON.
    Do NOT use markdown.
    Do NOT use backticks.
    If approved, preserve inline citation markers like [1], [2]. Do not remove them.


    JSON FORMAT:
    {{
      "approved": true/false,
      "feedback": "...",
      "final_version": "..."
    }}
    """

    response = generate_text(prompt, temperature=0.1)

    try:
        clean_json = extract_json(response)
        result = json.loads(clean_json)
    except Exception as e:
        print("Editor parsing error:", e)
        return {
            "approved": False,
            "editor_feedback": "Editor output invalid JSON. Please fix format."
        }

    if result.get("approved"):
        return {
            "approved": True,
            "final_blog": result.get("final_version"),
            "editor_feedback": None
        }
    else:
        return {
            "approved": False,
            "editor_feedback": result.get("feedback")
        }

def image_concept_agent(state: BlogState):

    blog_text = state.get("final_blog") or state.get("draft_blog")

    prompt = f"""
    You are a visual content strategist and expert in the given topic.

    Blog Content:
    {blog_text}

    Extract:
    - 3 to 5 actionble items or tools or products or themes
    - Abstract concepts, key points or numbers
    - Keep the tone posivite and enthusiastic

    RULES:
    - Use only concepts explicitly mentioned in blog
    - Do NOT invent data
    - Do NOT include numeric claims
    - Do NOT fabricate branding
    - Do Not over generalise. Give specific points.
    - Keep response concise
    """

    concepts = generate_text(prompt, temperature=0.1)

    return {"image_concepts": concepts}


def image_prompt_builder(state: BlogState):

    domain = state.get("domain", "general")
    config = load_domain_config(domain)

    concepts = state["image_concepts"]

    agent_config = config["medium_image"]

    constraints_text = "\n".join(
        f"- {c}" for c in agent_config["constraints"]
    )
    task_text = agent_config["task"]
    style_text = agent_config["style"]

    medium_prompt = f"""
        topic:
        {state['topic']}

        Visual themes:
        {concepts}

        style:
        {style_text}

        Constraints:
        {constraints_text}

        TASK:
        {task_text}
        """


    agent_config = config["insta_image"]

    constraints_text = "\n".join(
        f"- {c}" for c in agent_config["constraints"]
    )
    task_text = agent_config["task"]
    style_text = agent_config["style"]

    instagram_prompt = f"""
        topic:
        {state['topic']}

        Visual themes:
        {concepts}

        style:
        {style_text}

        Constraints:
        {constraints_text}

        TASK:
        {task_text}
        """

    return {
        "image_prompt_medium": medium_prompt,
        "image_prompt_instagram": instagram_prompt
    }


def generate_image(prompt: str, filename_prefix: str):
    model = genai.GenerativeModel(model_name=image_model_name) 
    response = model.generate_content(prompt)
    # Extract image binary
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data:
                image_bytes = part.inline_data.data

                filename = f"{filename_prefix}_{datetime.now().timestamp()}.png"

                with open(filename, "wb") as f:
                    f.write(image_bytes)

                return filename

    raise ValueError("No image returned from model.")

@traceable
def generate_image_fn(state):
    prompt_medium = state["image_prompt_medium"]
    prompt_instagram = state["image_prompt_instagram"]

    medium_file = generate_image(prompt_medium, "medium_header")
    instagram_file = generate_image(prompt_instagram, "instagram_post")

    return {
        "image_url_medium": medium_file,
        "image_url_instagram": instagram_file,
    }

"""
def image_agent(state: BlogState):

    medium_file = generate_image(
        state["image_prompt_medium"],
        "medium_header"
    )

    instagram_file = generate_image(
        state["image_prompt_instagram"],
        "instagram_post"
    )

    return {
        "image_url_medium": medium_file,
        "image_url_instagram": instagram_file
    }
"""


# =========================================================

# Publish Agent
# =========================================================

def publish_agent(state: BlogState):

    print("================== publish agent====================")
    final_content = state.get("final_blog") or state.get("draft_blog")

    formatted = f"{final_content}\n\n---\n\nReferences:\n"

    for idx, link in enumerate(state["citations"], start=1):
        formatted += f"[{idx}] {link}\n"

    formatted += "\n\nMedium Header Image:\n"
    formatted += state.get("image_url_medium", "") + "\n"

    formatted += "\nInstagram Post Image:\n"
    formatted += state.get("image_url_instagram", "") + "\n"

    return {"final_blog": formatted}


# =========================================================
# Router Logic
# =========================================================

MAX_ITERATIONS = 5

def review_router(state: BlogState):

    if state["approved"]:
        #return "publish"
        return "image_concepts"

    if state["iteration_count"] >= MAX_ITERATIONS:
        print("\n⚠ Max revisions reached. Publishing last draft.\n")
        #return "publish"
        return "image_concepts"

    return "writer"

# =========================================================
# Build Graph
# =========================================================

def build_graph():

    image_runnable = RunnableLambda(generate_image_fn).with_config(
        run_name="Image Generation Agent"
        )


    builder = StateGraph(BlogState)

    builder.add_node("domain", domain_classifier_agent)
    builder.add_node("research", research_agent)
    builder.add_node("writer", writer_agent)
    builder.add_node("editor", editor_agent)
    builder.add_node("publish", publish_agent)
    builder.add_node("image_concepts", image_concept_agent)
    builder.add_node("image_prompt", image_prompt_builder)
    #builder.add_node("image_generate", image_agent)
    builder.add_node("image_generate", image_runnable)


    builder.set_entry_point("domain")

    builder.add_edge("domain", "research")
    builder.add_edge("research", "writer")
    builder.add_edge("writer", "editor")

    builder.add_conditional_edges(
        "editor",
        review_router,
        {
            "writer": "writer",
            "image_concepts": "image_concepts"
        }
    )

    #builder.add_edge("editor", "image_concepts")
    builder.add_edge("image_concepts", "image_prompt")
    builder.add_edge("image_prompt", "image_generate")
    builder.add_edge("image_generate", "publish")


    builder.add_edge("publish", END)

    return builder.compile()

# =========================================================
# Main Execution
# =========================================================

if __name__ == "__main__":

    graph = build_graph()

    topic = input("Enter blog topic: ")

    initial_state = {
        "topic": topic,
        "research_sources": [],
        "research_summary": "",
        "draft_blog": "",
        "editor_feedback": None,
        "approved": False,
        "final_blog": None,
        "citations": [],
        "iteration_count": 0
    }
    #save mermaid graph as PNG
    #print mermaid code
    #print(graph.get_graph().draw_mermaid())
    #graph.get_graph().draw_mermaid_png(output_file_path="blog_writer_agent.png")

    result = graph.invoke(initial_state)

    print("\n============================")
    print("        FINAL BLOG")
    print("============================\n")
    print(result["final_blog"])
