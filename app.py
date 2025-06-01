
import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
import html

# Load environment variables
load_dotenv()

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set up OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Gmail Scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']

# Initialize state for sent emails and ensure it's a set
if "sent_emails" not in st.session_state:
    st.session_state.sent_emails = set()  # Track processed email IDs as a set
elif not isinstance(st.session_state.sent_emails, set):
    # Convert a previous list to a set if necessary
    st.session_state.sent_emails = set(st.session_state.sent_emails)

# Custom CSS for the app
st.markdown(
    """
    <style>
    body {
        background-color: #F8F9FA;
        font-family: 'Roboto', sans-serif;
    }
    .css-18e3th9 {
        padding: 2rem;
    }
    h1, h2, h3 {
        text-align: center;
        font-weight: bold;
        color: #2C3E50;
        margin-top: 20px;
    }
    .stTextInput > div > div > input {
        background: #FFFFFF;
        border: 1px solid #2C3E50;
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
    }
    .stButton > button {
        background: #3498DB;
        color: white;
        font-size: 18px;
        border: 0;
        padding: 10px 20px;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #2980B9;
    }
    .stSidebar > div:first-child {
        background: #2C3E50;
        color: white;
        border-radius: 12px;
        padding: 20px;
    }
    .email-box, .response-box {
        background: #FFFFFF;
        border: 1px solid #3498DB;
        border-radius: 12px;
        padding: 15px;
        margin: 15px 0;
    }
    .email-box h4, .response-box h4 {
        color: #2980B9;
        font-size: 18px;
    }
    .email-box p, .response-box p {
        color: #2C3E50;
        font-size: 16px;
        line-height: 1.5;
        background-color: transparent;
        word-wrap: break-word;
    }
    .response-box p::selection {
        background-color: #3498DB;
        color: white;
    }
    .response-box a {
        color: #3498DB;
    }
    .response-box a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main function
def main():

    # Display IITM  Team logo/banner at the top
    st.image(
        "Lumio AI (7).png",  # Replace with your image filename
        caption="",  # Caption below the image
        use_container_width=True  # Automatically adjust the image to fit the app's width
    )
    st.title("📧 Auto Email Responder")
    st.markdown("Automate your email responses with AI and explore document-based queries effortlessly.")

    # Sidebar for document upload
    with st.sidebar:
        st.subheader("📂 Upload Your Documents")
        uploaded_files = st.file_uploader("Upload your PDFs here", type=['pdf'], accept_multiple_files=True)
        if st.button("Process Documents"):
            with st.spinner("Processing your documents..."):
                raw_text = extract_text_from_pdfs(uploaded_files)
                if raw_text:
                    text_chunks = split_text_into_chunks(raw_text)
                    vectorstore = create_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.success("Documents processed successfully!")

    # Email interaction section
    st.subheader("📬 Email Interaction")
    if st.button("Fetch Latest Email"):
        email_data = fetch_latest_email()
        if email_data:
            st.markdown(
                f"""
                <div class="email-box">
                    <h4>📧 Email Details</h4>
                    <p><strong>From:</strong> {html.escape(email_data['from'])}</p>
                    <p><strong>Subject:</strong> {html.escape(email_data['subject'])}</p>
                    <p><strong>Body:</strong> {html.escape(email_data['body'])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.session_state.email_data = email_data
        else:
            st.error("No new, non-blacklisted, or unprocessed emails found!")

    if st.button("Generate Response"):
        if "email_data" in st.session_state and "vectorstore" in st.session_state:
            response = generate_customized_response(st.session_state.email_data, st.session_state.vectorstore)
            if response:
                st.markdown(
                    f"""
                    <div class="response-box">
                        <h4>🤖 AI-Generated Response</h4>
                        <p>{response}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.session_state.generated_response = response
        else:
            st.error("Ensure both the email and documents are processed before generating a response.")

    if st.button("Send Response"):
        if "generated_response" in st.session_state and "email_data" in st.session_state:
            email_id = st.session_state.email_data['id']
            success = send_email(
                st.session_state.email_data['from'],
                st.session_state.email_data['subject'],
                st.session_state.generated_response,
                email_id
            )
            if success:
                st.success("Response sent successfully!")
            else:
                st.error("Failed to send the response!")

# Supporting Functions
def generate_customized_response(email_data, vectorstore):
    """
    Generate a more customized AI response using email body and the context from uploaded PDFs, and ensure that the response is in detailed way, as you are the Support Team Executive.
    """
    # Extract sender's name (simple extraction, assumes format "Name <email>")
    sender_name = email_data['from'].split("<")[0].strip() if "<" in email_data['from'] else email_data['from']
    
    # Call the AI model to generate a general response to the email
    ai_response = generate_response(email_data, vectorstore)

    # Add the sender's name and personalized salutation to the response
    personalized_response = f"Dear {sender_name},\n\n{ai_response}\n\nBest regards,\nYour Support Team Executive"

    return personalized_response

def extract_text_from_pdfs(docs):
    text = ""
    for pdf in docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(raw_text)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    return faiss.FAISS.from_texts(texts=chunks, embedding=embeddings)

def fetch_latest_email():
    service = authenticate_gmail()
    blacklist = ["@bizbuysell.com", "noreply@", "newsletter@", "support@", "sales@", "offers@", "alerts@", "mumesh8080@gmail.com", "donotreply@"]

    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread", fields="messages(id)").execute()
        messages = results.get('messages', [])
        for msg in messages:
            msg_id = msg['id']
            if msg_id in st.session_state.sent_emails:
                continue
            message = service.users().messages().get(userId='me', id=msg_id, format="metadata", metadataHeaders=["From", "Subject"]).execute()
            sender = next(header['value'] for header in message['payload']['headers'] if header['name'] == 'From')
            if any(b in sender for b in blacklist):
                continue
            return {
                'id': msg_id,
                'from': sender,
                'subject': next(header['value'] for header in message['payload']['headers'] if header['name'] == 'Subject'),
                'body': message.get('snippet', '')
            }
        return None
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

def generate_response(email_data, vectorstore):
    llm = ChatOpenAI(temperature=0.0, max_tokens=256, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    response = conversation_chain({'question': email_data['body']})
    return response['answer']

def send_email(recipient, subject, content, email_id):
    service = authenticate_gmail()
    try:
        message = MIMEMultipart()
        message['To'] = recipient
        message['Subject'] = subject
        message.attach(MIMEText(content, 'plain'))
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        st.session_state.sent_emails.add(email_id)
        return True
    except HttpError as error:
        print(f"An error occurred: {error}")
        return False

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

if __name__ == "__main__":
    main()
