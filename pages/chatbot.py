import streamlit as st
from llama_index.llms.ollama import Ollama
from langchain_community.tools import DuckDuckGoSearchResults

# Create an instance of the Ollama LLM
llm = Ollama(model="llama3.2", request_timeout=60)

# Initialize DuckDuckGo search tool
tool = DuckDuckGoSearchResults()

# Define the chatbot page function
def chatbot():
    st.title("LLM Chatbot")

    # Initialize the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.buffered_memory = ""
        st.session_state.chat_history = []

    # Sidebar for New Chat button
    with st.sidebar:
        if st.button("New Chat"):
            if st.session_state.messages:
                st.session_state.chat_history.append({
                    "chat_id": len(st.session_state.chat_history) + 1,
                    "messages": st.session_state.messages.copy()
                })
            st.session_state.messages = []
            st.session_state.buffered_memory = ""

    # Display old chat history
    if st.session_state.chat_history:
        st.subheader("Previous Chats")
        for chat in st.session_state.chat_history:
            st.markdown(f"**Chat {chat['chat_id']}:**")
            for msg in chat['messages']:
                st.markdown(f"{msg['role'].capitalize()}: {msg['content']}")
            st.markdown("---")

    # Display current chat messages
    st.subheader("Current Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.buffered_memory += f"User: {prompt}\n"

        with st.spinner("Searching DuckDuckGo..."):
            search_results = duckduckgo_search(prompt)

            if search_results:
                response = refine_response(search_results, prompt)
            else:
                response = "Sorry, I couldn't find any information."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.buffered_memory += f"Assistant: {response}\n"

        with st.chat_message("assistant"):
            st.markdown(response)

# Function to perform a DuckDuckGo search
def duckduckgo_search(query):
    try:
        results = tool.run(query)
        if results and "results" in results:
            return "\n".join([f"{result['title']}: {result['link']}" for result in results['results']])
        else:
            return None  # Return None if no results are found
    except Exception as e:
        return f"Error during DuckDuckGo search: {str(e)}"

# Function to refine the response to ensure it's polite and appropriate
def refine_response(search_results, original_query):
    try:
        prompt = f"You just performed a search for: '{original_query}'. Here are the results:\n\n{search_results}\n\n"
        prompt += "Please provide a polite, well-structured, and concise response based on the information above."
        
        refined_response = llm.complete(prompt)

        if refined_response:
            return refined_response
        else:
            return "No refined response received."
    except Exception as e:
        return f"Error refining response: {str(e)}"

