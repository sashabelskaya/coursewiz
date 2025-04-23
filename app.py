import streamlit as st
import requests
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- API Keys ----
GROQ_API_KEY = "your_API"
DEEPSEEK_API_KEY = "your_API"  # Example placeholder
MISTRAL_API_KEY = "your_API"  # Example placeholder
GEMINI_API_KEY = "your_API"

# ---- Google Custom Search API Setup ----
GOOGLE_API_KEY = "your_API"
SEARCH_ENGINE_ID = "your_ID"

# ---- Setup UI ----
st.set_page_config(page_title="LLMRank Lite", page_icon="📈")
st.title("🔍 LLMRank Lite – Does Your Content Rank in ChatGPT?")
st.markdown("Paste your content, simulate queries, and see if an LLM finds it relevant.")

# ---- User Inputs ----
content = st.text_area("📄 Paste Your Content Here", height=300)

default_queries = [
    "What is the best SEO tool for AI models?",
    "How can I optimize my content for ChatGPT?",
    "What is LLM SEO?",
]

st.markdown("### 🧠 Simulated LLM Queries")
queries = [st.text_input(f"Query {i+1}", q) for i, q in enumerate(default_queries)]

# ---- Model Selector ----
model_selector = st.radio("Choose LLM Model", ["Groq", "DeepSeek", "Mistral", "Gemini"])

# ---- Query Function for Different Models ----
def query_llm_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        st.error(f"❌ Error: {response.status_code}\n{response.text}")
        return "Error getting response from Groq LLM."

# Add DeepSeek Integration (using the placeholder API URL)
def query_llm_deepseek(prompt):
    url = "https://api.deepseek.ai/v1/llm"  # Example placeholder URL
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-3",
        "prompt": prompt,
        "max_tokens": 150,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["completion"].strip()
    else:
        st.error(f"❌ Error: {response.status_code}\n{response.text}")
        return "Error getting response from DeepSeek LLM."

# Add Mistral Integration
def query_llm_mistral(prompt):
    url = "https://api-inference.huggingface.co/models/mistral-7b-v0.1"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
    data = {"inputs": prompt}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()[0]["generated_text"].strip()
    else:
        st.error(f"❌ Error: {response.status_code}\n{response.text}")
        return "Error getting response from Mistral LLM."

# Add Gemini Integration
def query_llm_gemini(prompt):
    url = "https://api.paLM.google.com/v1/models/gemini-1/completions"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gemini-1",
        "prompt": prompt,
        "max_tokens": 150,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"].strip()
    else:
        st.error(f"❌ Error: {response.status_code}\n{response.text}")
        return "Error getting response from Gemini LLM."

# ---- Google Custom Search API Function ----
def fetch_google_search_results(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json().get("items", [])
        return results
    else:
        st.error(f"❌ Error fetching Google search results: {response.status_code}")
        return []

# ---- Calculate Boss Score (Relevance Score) ----
def calculate_boss_score(response, query):
    # Basic calculation of relevance (this can be refined further)
    relevance_keywords = ["relevant", "useful", "fit", "matches", "answer", "query"]
    score = sum([1 for word in relevance_keywords if word in response.lower()])
    return min(score * 20, 100)  # Ensure the score doesn't exceed 100

# ---- Querying the Selected Model ----
def query_llm(prompt, model_choice):
    if model_choice == "Groq":
        return query_llm_groq(prompt)
    elif model_choice == "DeepSeek":
        return query_llm_deepseek(prompt)
    elif model_choice == "Mistral":
        return query_llm_mistral(prompt)
    elif model_choice == "Gemini":
        return query_llm_gemini(prompt)

# ---- Run LLM Ranking ----
if st.button("🔍 Run LLM Rank Check") and content:
    with st.spinner("Asking the LLM..."):
        results = []
        relevance_scores = []
        for query in queries:
            prompt = f"""
You are an LLM simulating a search for the following user query:

Query: "{query}"

Based on the following content, would you use it to answer the query? 
Rate relevance from 0 (not at all) to 100 (perfect fit), then explain why.

Content:
\"\"\"
{content}
\"\"\"
"""
            answer = query_llm(prompt, model_selector)
            boss_score = calculate_boss_score(answer, query)
            relevance_scores.append(boss_score)
            results.append((query, answer, boss_score))

    st.success("LLM Rank Results")
    for query, answer, score in results:
        st.markdown(f"**📝 Query:** {query}")
        st.markdown(f"`🔎 LLM Response:`\n{answer}")
        st.markdown(f"**📊 Boss Score:** {score}")
        st.markdown("---")

    # ---- Google Search Comparison ----
    st.subheader("🔍 Google Search Comparison")
    for query in queries:
        search_results = fetch_google_search_results(query)
        st.markdown(f"**Google Search Results for:** {query}")
        if search_results:
            for idx, result in enumerate(search_results[:3], start=1):
                title = result.get("title")
                link = result.get("link")
                st.markdown(f"{idx}. [{title}]({link})")
        else:
            st.markdown(f"No results found for: **{query}**")

    # ---- Relevance Heatmap ----
    st.subheader("🔥 Relevance Heatmap")
    heatmap_data = np.array([relevance_scores])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", cbar=True, xticklabels=queries, yticklabels=["Relevance"])
    st.pyplot(fig)

# ---- Rewrite Section ----
if st.button("✍️ Suggest LLM-Optimized Rewrite") and content:
    with st.spinner("Rewriting with selected LLM..."):
        rewrite_prompt = f"""
Rewrite the following content to be highly retrievable and relevant for large language models (like ChatGPT).
Use a clear answer-style format, with structured headings, FAQs, or definitions if needed.

Content:
\"\"\"
{content}
\"\"\"
"""
        optimized = query_llm(rewrite_prompt, model_selector)

    st.subheader("✅ LLM-Optimized Rewrite")
    st.code(optimized, language="markdown")
