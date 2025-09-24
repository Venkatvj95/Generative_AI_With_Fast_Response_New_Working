import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.tools import GoogleSearchRun

# 🔐 API keys
groq_api_key = "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"
google_api_key = "AIzaSyBquYJm-4MzJiMyzyrcoWQaR0g-q62vwR0"
google_cse_id = "755f1b7126e8c49a6"

# 🔍 Google Search Tool
google_wrapper = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id,
    k=1  # only 1 result for speed
)
search_tool = GoogleSearchRun(api_wrapper=google_wrapper)

# ⚙️ Model (no streaming → faster)
llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", streaming=False)

# 🧠 Chat UI
st.title("⚡ Fast AI Chatbot with Google Search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I’m your AI assistant. Ask me anything — I’ll respond fast!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 🔍 Helper: Decide if search is needed
def needs_search(query: str) -> bool:
    keywords = ["latest", "news", "price", "search", "find", "who is", "where", "when"]
    return any(word in query.lower() for word in keywords)

# 💬 Handle user input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        try:
            if needs_search(prompt):
                # Run only one direct Google search
                search_result = search_tool.run(prompt)
                response = llm.invoke(f"Based on this search result, answer the query:\n\n{search_result}\n\nUser question: {prompt}")
                final_text = response.content if hasattr(response, "content") else str(response)
            else:
                response = llm.invoke(prompt)
                final_text = response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            final_text = f"❌ Error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": final_text})
        st.write(final_text)
