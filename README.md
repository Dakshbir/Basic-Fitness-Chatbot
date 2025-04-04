# Fitness Chatbot with LLAMA2

🚀 **AI-Powered Fitness Assistant!**  
Welcome to "Fitsie" - your personal fitness chatbot powered by LLAMA2! This chatbot provides expert guidance on fitness, health, diet, and workouts using AI-driven responses. Whether you're looking for workout plans, diet advice, or general fitness tips, Fitsie has got you covered!

## 🏗️ Project Structure
```
fitness-chatbot/
│── localama.py                 # Basic Langchain & Streamlit Chatbot
│── localama_modified.py        # Enhanced Chainlit & Streamlit Chatbot
│── requirements.txt            # Dependencies
│── .env                        # API Keys (excluded from GitHub)
│── README.md                   # Project Documentation
```

## 🛠️ Tech Stack
- **LLM**: [Ollama (LLAMA2)](https://ollama.ai/)
- **LangChain**: Manages prompts, memory, and execution chains
- **Chainlit**: Provides an interactive chatbot UI
- **Streamlit**: Enables web-based chatbot interaction
- **Python**: Core language for development

## 🏗️ Setup and Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/Dakshbir/fitness-chatbot.git
   cd fitness-chatbot
   ```

2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file and add your API keys:
   ```sh
   LANGCHAIN_API_KEY=your_api_key_here
   ```

4. **Run the chatbot**:
   - Using Chainlit:
     ```sh
     chainlit run localama_modified.py
     ```
   - Using Streamlit:
     ```sh
     streamlit run localama.py
     ```

## 📜 Features
✅ AI-powered fitness advice  
✅ Memory-based contextual chat  
✅ Web-based chatbot interface  
✅ Real-time, interactive responses  


---
🔗 **Deployed Link**:   
📬 **Contact**: [Dakshbir Singh](https://www.linkedin.com/in/dakshbir-singh-kapoor-26210b286/)  


