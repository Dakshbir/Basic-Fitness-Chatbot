Fitness Chatbot with LLAMA2

🚀 AI-Powered Fitness Assistant!Welcome to "Fitsie" - your personal fitness chatbot powered by LLAMA2! This chatbot provides expert guidance on fitness, health, diet, and workouts using AI-driven responses. Whether you're looking for workout plans, diet advice, or general fitness tips, Fitsie has got you covered!

🏗️ Project Structure

fitness-chatbot/
│── localama.py                 # Basic Langchain & Streamlit Chatbot
│── localama_modified.py        # Enhanced Chainlit & Streamlit Chatbot
│── requirements.txt            # Dependencies
│── .env                        # API Keys (excluded from GitHub)
│── README.md                   # Project Documentation

🛠️ Tech Stack

LLM: Ollama (LLAMA2)

LangChain: Manages prompts, memory, and execution chains

Chainlit: Provides an interactive chatbot UI

Streamlit: Enables web-based chatbot interaction

Python: Core language for development

🏗️ Setup and Installation

Clone the repository:

git clone https://github.com/Dakshbir/fitness-chatbot.git
cd fitness-chatbot

Install dependencies:

pip install -r requirements.txt

Set up environment variables:
Create a .env file and add your API keys:

LANGCHAIN_API_KEY=your_api_key_here

Run the chatbot:

Using Chainlit:

chainlit run localama_modified.py

Using Streamlit:

streamlit run localama.py

📜 Features

✅ AI-powered fitness advice✅ Memory-based contextual chat✅ Web-based chatbot interface✅ Real-time, interactive responses


🔗 Deployed Link: Project Live Here📬 Contact: Dakshbir Singh💡 
