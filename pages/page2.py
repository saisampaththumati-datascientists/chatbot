import streamlit as st
from llama_index.llms.ollama import Ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .llm_instance import get_llm_instance

# Initialize Ollama LLM
ollama_llm = get_llm_instance()


def generate_question_from_chunk(chunk):
    prompt = f"Create a question based on this content: {chunk}"
    response = ollama_llm.invoke(input=prompt)  # Pass the prompt as input
    return response.get("text", "No question generated.") if isinstance(response, dict) else response


def answer_question_with_llm(question, chunk):
    prompt = f"Answer the following question based on the provided text: \n\nText: {chunk}\n\nQuestion: {question}"
    response = ollama_llm.invoke(input=prompt)  # Pass the prompt as input
    return response.get("text", "No answer generated.") if isinstance(response, dict) else response


def create_multiple_choice_question(chunk):
    prompt = f"Create a multiple-choice question with 4 options based on this text: {chunk}"
    response = ollama_llm.invoke(input=prompt)  # Pass the prompt as input
    return response.get("text", "No question generated.") if isinstance(response, dict) else response

def find_best_matching_chunk(question, chunks):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([question] + chunks)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_match_index = np.argmax(cosine_similarities)
    return chunks[best_match_index]


def handle_qa(chunks, max_questions=10):
    st.header("Question & Answer Section")

    # Generate 10 questions from the chunks
    st.subheader("Generated Questions")
    generated_questions = []
    
    for i, chunk in enumerate(chunks[:max_questions]):  # Limit to 10 questions
        with st.spinner(f"Generating question {i + 1} of {max_questions}..."):
            question = generate_question_from_chunk(chunk)
            generated_questions.append(question)
            st.write(f"**Question {i + 1}:** {question}")

    # Allow the user to enter their own question for Q&A
    st.write("---")
    user_question = st.text_input("Enter your own question based on the text data:")
    if user_question:
        with st.spinner("Finding the best matching chunk..."):
            best_chunk = find_best_matching_chunk(user_question, chunks)
            st.write(f"**Best Matching Chunk:**\n{best_chunk}")

            with st.spinner("Generating answer..."):
                answer = answer_question_with_llm(user_question, best_chunk)
                st.write(f"**Answer:** {answer}")

        # Generate multiple-choice questions if desired
        if st.checkbox("Generate a multiple-choice question based on the best matching chunk?"):
            mcq = create_multiple_choice_question(best_chunk)
            st.write(f"**Multiple-Choice Question:** {mcq}")


def page2():
    st.title("LLM Page 2")
    st.write("This is the second page for viewing results or other features.")

    # Check if chunks are available in session state
    if 'chunks' not in st.session_state or not st.session_state.chunks:
        st.warning("No chunks found. Please process the document in the 'Document Processing' page first.")
        return

    # Handle Q&A with the available chunks and generate questions
    handle_qa(st.session_state.chunks)
