import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO
from PIL import Image
from gtts import gTTS
import base64
import os
import uuid
import json
from dotenv import load_dotenv
load_dotenv()

# Hugging Face API details
API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialize InferenceClient correctly (without 'provider')
client = InferenceClient(
    api_key=API_KEY,
)

# Set page configuration with icon
st.set_page_config(page_title="PolyMorphAI", page_icon="resources/i4.png")

# --- Chat History Management ---
def save_chat_history(history, chat_id):
    """Function to save chat history in a JSON file."""
    with open(f"chat_history/{chat_id}.json", "w") as file:
        json.dump(history, file)

def load_chat_history(chat_id):
    """Function to load chat history from a JSON file."""
    try:
        with open(f"chat_history/{chat_id}.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


# --- API Interaction and Response Generation ---
def generate_response(model, user_input, history, role_input=None):
    """Generate AI response based on user input, chat history, and role input."""
    
    # Combine the history into a single string
    history_input = "\n".join(history)
    
    # Prepare the role input if provided
    role_input_prompt = f"Respond as if you are: {role_input}\n" if role_input else ""
    
    # Construct the full input for the model
    full_input = f"{role_input_prompt}{history_input}\nUSER: {user_input}\nAI:"
    
    messages = [
        {"role": "user", "content": full_input}
    ]
    
    # Create the completion using InferenceClient
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
        )
        full_response = completion.choices[0].message['content']
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        full_response = "No response generated"

    # Clean the response to remove unnecessary parts
    return clean_ai_response(full_response)


def clean_ai_response(response_text):
    """Remove unnecessary metadata or labels from the AI response."""
    
    # Remove role instructions like "Respond as if you are: "
    response_text = response_text.split("USER:")[-1].strip()
    
    # If the response includes 'AI:', remove everything before the actual response
    if "AI:" in response_text:
        response_text = response_text.split("AI:")[-1].strip()
    
    return response_text

# --- Media Generation Functions ---
def generate_image(user_input):
    """Generate an image from user input using Hugging Face API."""
    payload = {"inputs": user_input}
    try:
        response = client.image.generate(
            model="stabilityai/stable-diffusion-2",  # Example image model
            inputs=payload
        )
        image = Image.open(BytesIO(response))
        return image
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def text_to_audio(text):
    """Convert user input text to audio using Google Text-to-Speech."""
    tts = gTTS(text)
    audio_path = f"temp/{uuid.uuid4()}.mp3"
    tts.save(audio_path)
    return audio_path


# --- Directory Setup ---
# Create directories if they don't exist
os.makedirs("chat_history", exist_ok=True)
os.makedirs("temp", exist_ok=True)


# --- Helper Functions for Download Links ---
def get_image_download_link(image):
    """Generate a download link for the generated image."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/jpeg;base64,{img_str}" download="generated_image.jpg">Download Image</a>'
    return href

def get_audio_download_link(audio_path):
    """Generate a download link for the generated audio."""
    with open(audio_path, "rb") as file:
        audio_bytes = file.read()
    audio_str = base64.b64encode(audio_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{audio_str}" download="generated_audio.mp3">Download Audio</a>'
    return href


# --- Streamlit UI Components ---
# Sidebar for chat management
chat_id = st.sidebar.selectbox("Select Chat", ["New Chat"] + [f[:-5] for f in os.listdir("chat_history") if f.endswith('.json')])
if chat_id == "New Chat":
    chat_id = str(uuid.uuid4())
    st.session_state.chat_history = []
else:
    st.session_state.chat_history = load_chat_history(chat_id)

# Display chat history
st.markdown("### Chat History")
for message in st.session_state.chat_history:
    st.markdown(message)

# Define the custom role input for the user
role_input = st.sidebar.text_input("Enter the role/theme (e.g., 'Talk to me like my girlfriend')")

# Main area for user input and clear button
st.markdown("### Enter your prompt:")

col1, col2, col3 = st.columns([6, 1, 1])
with col1:
    user_input = st.text_area("Enter your prompt:", key="user_input", height=100, label_visibility="hidden")
with col2:
    if st.button("🗑️", key="clear_button", help="Clear chat history"):
        st.session_state.chat_history = []
        if user_input:
            st.session_state.chat_history.append(f"You: {user_input}")
            model = "deepseek-ai/DeepSeek-V3"  # Default model
            response = generate_response(model, user_input, st.session_state.chat_history, role_input)
            st.session_state.chat_history.append(f"AI: {response}")
            save_chat_history(st.session_state.chat_history, chat_id)


# Save chat history when closing
if st.session_state.chat_history:
    save_chat_history(st.session_state.chat_history, chat_id)

# --- Mode Selection ---
mode = st.sidebar.radio("Choose mode:", ["Generative", "Text to Image", "Text to Audio"])

# Generative Mode
if mode == "Generative":
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "deepseek-ai/DeepSeek-V3"  # Default model
    selected_model = st.sidebar.selectbox("Select Model", [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-V3", 
    "openai-community/gpt2",
])  # List the models as separate items
    st.session_state.selected_model = selected_model  # Store the selected model in session_state

    if st.button("Generate", key="generate_button_generative"):
        # API URL for the selected model
        model = st.session_state.selected_model
        
        with st.spinner("Generating response..."):
            response = generate_response(model, user_input, st.session_state.chat_history, role_input)
            st.session_state.chat_history.append(f"You: {user_input}")
            st.session_state.chat_history.append(f"AI: {response}")
            st.markdown(f"\n{response}", unsafe_allow_html=True)
