import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Custom CSS for message alignment
st.markdown(
    """
    <style>
    .user-msg { text-align: right; color: blue; }
    .bot-msg { text-align: left; color: green; }
    .msg-container { display: flex; justify-content: space-between; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the model and tokenizer
try:
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    st.error(f"Error loading model and tokenizer: {e}")
    st.stop()

# Streamlit interface
st.title("ChatBot using Streamlit and Hugging Face")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = []
if "bot_responses" not in st.session_state:
    st.session_state.bot_responses = []

def generate_response(user_input):
    try:
        # Encode the input and append to chat history
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
        if st.session_state.chat_history:
            bot_input_ids = torch.cat([torch.tensor(st.session_state.chat_history), new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Create attention mask
        attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

        # Generate response
        chat_history_ids = model.generate(bot_input_ids, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Update chat history
        st.session_state.chat_history = chat_history_ids
        st.session_state.user_inputs.append(user_input)
        st.session_state.bot_responses.append(response)
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process that."

# User input
user_input = st.text_input("You: ", key="input")

if user_input:
    response = generate_response(user_input)
    st.write(f"Bot: {response}")

# Display conversation history
for user_msg, bot_msg in zip(st.session_state.user_inputs, st.session_state.bot_responses):
    st.markdown(f'<div class="msg-container"><div class="bot-msg">{bot_msg}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="msg-container"><div class="user-msg">{user_msg}</div></div>', unsafe_allow_html=True)
