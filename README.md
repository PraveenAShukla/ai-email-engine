# 📧 AI-Powered Auto Email Responder

This Streamlit application automates email replies using AI and allows support teams to respond contextually based on uploaded PDFs. The app fetches unread Gmail messages, analyzes them with LangChain, and sends back personalized, professional replies.

## 🚀 Features

- ✅ Fetch unread emails via Gmail API
- 🤖 Generate AI-based email responses using OpenAI + LangChain
- 📂 Upload and process multiple PDFs for contextual reference
- 🔍 Context-aware and personalized replies
- 📤 Send replies directly from the app
- 💬 Beautiful and responsive UI with Streamlit

---

## 📸 App Preview

![App Banner](Lumio%20AI%20(7).png)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: OpenAI (via LangChain)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Vectorstore**: FAISS
- **Gmail Integration**: Google API (Read + Send)

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/PraveenAShukla/auto-email-responder.git
cd auto-email-responder ```


2. Install Dependencies
Make sure Python 3.8+ is installed.
pip install -r requirements.txt
3. Environment Variables
Create a .env file and add:

OPENAI_API_KEY=your-openai-api-key


4. Set Up Gmail API
Go to Google Cloud Console.

Enable Gmail API.

Download your credentials.json.

Place it in your project directory.

Run the app once to authenticate and create a token.json.

📂 How to Use
🔹 Upload PDFs
Upload support documents in the sidebar.

Click "Process Documents" to create a vector store.

🔹 Respond to Emails
Click "Fetch Latest Email".

View email preview.

Click "Generate Response".

Optionally review and edit.

Click "Send Response" to reply.

✅ Sample Flow
mermaid
graph TD
    A[User Uploads PDFs] --> B[Vectorstore Created]
    B --> C[User Clicks Fetch Email]
    C --> D[Email Displayed]
    D --> E[User Clicks Generate Response]
    E --> F[AI Generates Contextual Reply]
    F --> G[User Clicks Send]
    G --> H[Reply Sent via Gmail]




📁 Folder Structure
pgsql
├── app.py
├── credentials.json
├── token.json
├── requirements.txt
├── .env
├── Lumio AI (7).png



