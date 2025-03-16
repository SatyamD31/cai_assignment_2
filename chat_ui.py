import streamlit as st
import torch
from rag import FinancialChatbot
import warnings
from transformers import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

torch.classes.__path__ = []

# Use session state to persist the chatbot instance
if "chatbot" not in st.session_state:
    st.session_state.chatbot = FinancialChatbot(data_path="Nestle_Financtial_report_till2023.xlsx")

def fetch_answer_from_backend(query):
    """Calls the backend function to get an answer."""
    return st.session_state.chatbot.get_answer(query)  # Use session state chatbot instance



# Initialize Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "loading" not in st.session_state:
    st.session_state.loading = False

# Layout and Title
st.title("Financial RAG Chat Assistant")


# st.markdown(
#     """
#     <style>
#     body, .stApp {
#         background: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQlSQ-kkQFs_BAOhel_cd_CkFoHN1V-5bUO7vv8NwRp0cjftW6Wm2utQSE&s=10');
#         background-size: cover;
#         # background-attachment: fixed;
#         # background-position: center;
#         # background-repeat: no-repeat;
#     }
#     # .stApp {
#     #     background: transparent;
#     # }
#     # .stTextInput, .stTextArea, .stButton {
#     #     background-color: rgba(0, 0, 0, 0.5);
#     #     border-radius: 10px;
#     #     color: #ffffff;
#     # }
#     # .stTextInput > div > input {
#     #     color: #ffffff;
#     # }
#     # .stButton button {
#     #     color: #ffffff;
#     #     background-color: #4CAF50;
#     #     border: none;
#     #     border-radius: 10px;
#     # }
#     </style>
#     """,
#     unsafe_allow_html=True
# )





# Display Chat History
for chat in st.session_state.chat_history:
    # User's query on the right
    st.markdown(
        f"""
        <div style='text-align: right; max-width: 75%; float: right; clear: both;'>
            <div style='font-size: 13px; color: #4a4a4a; margin: 0 10px;'>You</div>
            <div style='background-color: #d3d3d3; color: black; padding: 10px; border-radius: 10px; 
                        box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.1);'>
                {chat['question']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Assistant's label and response on the left
    if chat["answer"] is not None:
        st.markdown(
            f"""
            <div style='text-align: left; max-width: 75%; float: left; clear: both;'>
                <div style='font-size: 13px; color: #4a4a4a; margin: 0 10px;'>Assistant</div>
                <div style='background-color: #add8e6; color: black; padding: 10px; border-radius: 10px; 
                        box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.1);'>
                    {chat['answer']}
                </div>
            </div>
            """,
        unsafe_allow_html=True
    )
    
    # Confidence Score (Below the answer)
    if chat["confidence"] is not None:
        st.markdown(
            f"<div style='clear: both; color: #4a4a4a; font-size: 13px;'>Confidence: {chat['confidence'] * 100}%</div>",
            unsafe_allow_html=True
        )
        st.divider()  # Adds a visual divider between Q&A pairs

# User Input (Always at the Bottom)
user_input = st.chat_input("Ask a financial question...")

# If user inputs a question
if user_input:
    # Add question to chat history and show loading animation
    st.session_state.chat_history.append({
        "question": user_input,
        "answer": None,  # Placeholder for the answer
        "confidence": None  # Placeholder for the confidence score
    })
    st.session_state.loading = True
    st.rerun()  # Refresh to display the question immediately

# If loading, simulate the API call and update the last question's answer
if st.session_state.loading:
    with st.spinner("Fetching answer..."):
        # Get the last question
        last_question = st.session_state.chat_history[-1]["question"]

        # API Call
        answer, confidence = fetch_answer_from_backend(last_question)

        # Update the last chat history item with the answer and confidence
        st.session_state.chat_history[-1]["answer"] = answer
        st.session_state.chat_history[-1]["confidence"] = confidence

    # Stop loading and refresh to show the answer
    st.session_state.loading = False
    st.rerun()
