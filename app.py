import os
import json
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
import uuid # For generating unique session IDs
from database import ConversationDatabase
import datetime
import re # Import re here as well for use in fuzzy_match
from difflib import SequenceMatcher # Import SequenceMatcher for fuzzy matching

# Load environment variables from .env file (for API key)
load_dotenv()

app = Flask(__name__)

ASTROLOGY_DATA_FILE = "astrology_data.txt"
VECTOR_DB_DIRECTORY = "./chroma_db"

qa_chain = None
conversation_memories = {}
db = ConversationDatabase()  # Initialize database connection

# Global variable to store astrology data (loaded once)
ASTROLOGY_DATA = {}

# Global list of zodiac patterns for direct fact recall
zodiac_patterns = [
    'what is my zodiac', 'what is my sign', 'which zodiac sign am i',
    'what zodiac sign am i', 'my zodiac sign', 'tell me my zodiac',
    'tell me my sign', 'zodiac sign', 'zodiac', 'sign'
]

# Global list of full month names and their abbreviations for robust parsing
month_names_map = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}

# Common words that should NOT be extracted as names
NON_NAME_WORDS = set([
    "hello", "hi", "what", "when", "where", "how", "why", "my", "your",
    "name", "sign", "zodiac", "horoscope", "i", "am", "is", "was", "born",
    "on", "tell", "me", "about", "a", "an", "the", "libra", "aries", "taurus",
    "gemini", "cancer", "leo", "virgo", "scorpio", "sagittarius", "capricorn",
    "aquarius", "pisces"
])


def fuzzy_match(a, b, max_dist=2):
    """Compares two strings with Levenshtein distance, returning True if ratio > 0.8 or distance is small."""
    try:
        ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio() # Compare lowercase strings
        # Also consider small absolute difference in length and character mismatches
        return ratio > 0.8 or (abs(len(a) - len(b)) <= max_dist and sum(1 for x, y in zip(a.lower(), b.lower()) if x != y) <= max_dist)
    except Exception:
        # Fallback to exact match if difflib fails for some reason
        return a.lower() == b.lower()


def extract_user_facts(message):
    """Extracts user facts (name, birth date) from a message. Returns a dict."""
    facts = {}
    message_lower = message.lower()

    # --- Name Extraction ---
    name_extracted = False
    name_patterns = [
        r"(?:i am|i'm|my name is|call me|it's|this is|myself)\s+([a-z][a-z .'-]{0,48})[.!?, ]*",
        r"([a-z][a-z]{1,20}) here",  
    ]
    for pat in name_patterns:
        name_match = re.search(pat, message_lower)
        if name_match:
            potential_name = name_match.group(1).strip().capitalize()
            if potential_name.lower() not in NON_NAME_WORDS:
                facts['name'] = potential_name
                name_extracted = True
                break
    
    # Fallback for name: if the entire message is just a capitalized word, and it's not a common non-name word
    if not name_extracted and re.fullmatch(r"([A-Z][a-z]{1,20})", message):
        potential_name = message.strip()
        if potential_name.lower() not in NON_NAME_WORDS:
            facts['name'] = potential_name


    # --- Birth Date Extraction ---
    date_patterns = [
        # Explicit "birthday is on/born on" followed by a date string
        r"(?:birthday is|born on|birthday:|dob is|dob:)\s*(\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b)",
        r"(?:birthday is|born on|birthday:|dob is|dob:)\s*(\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:,\s*\d{4})?\b)",
        r"(?:birthday is|born on|birthday:|dob is|dob:)\s*(\d{1,2}/\d{1,2}(?:/\d{2,4})?)",
        r"(?:birthday is|born on|birthday:|dob is|dob:)\s*(\d{4}-\d{2}-\d{2})",
        # Standalone Month Day Year (e.g., "August 5th, 1990", "Jan 1 2000")
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
        # Standalone Day Month Year (e.g., "5th August, 1990", "1 Jan 2000")
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(?:,\s*\d{4})?\b",
        # Standalone MM/DD/YYYY or MM/DD (e.g., "08/05/1990", "8/5")
        r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b",
        # Standalone YYYY-MM-DD
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]

    for pat in date_patterns:
        date_match = re.search(pat, message_lower)
        if date_match:
            # If the pattern used a capturing group for the date part (e.g., after "birthday is"), use that group.
            # Otherwise, use the full matched string.
            facts['birth_date'] = date_match.group(1).strip() if len(date_match.groups()) > 0 else date_match.group(0).strip()
            break 

    print(f"[DEBUG] Extracted facts: {facts}")
    return facts

def get_zodiac_sign(birth_date_str):
    """Given a birth date string, return the zodiac sign name, or None if not found."""
    month = None
    day = None

    # Try parsing various formats
    try:
        # Format: MonthName Day (e.g., "August 5th", "August 5")
        m = re.search(r"([A-Za-z]+)\s*(\d{1,2})(?:st|nd|rd|th)?", birth_date_str, re.IGNORECASE)
        if m:
            month_name_part = m.group(1).lower()
            day = int(m.group(2))
            month = month_names_map.get(month_name_part)
            if month is None:
                # Fuzzy match for month name if direct lookup fails
                for full_month, num in month_names_map.items():
                    if fuzzy_match(month_name_part, full_month, max_dist=1): # Allow 1 character difference
                        month = num
                        break

        # Format: Day MonthName (e.g., "5th August", "5 August")
        if month is None or day is None:
            m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*([A-Za-z]+)", birth_date_str, re.IGNORECASE)
            if m:
                day = int(m.group(1))
                month_name_part = m.group(2).lower()
                month = month_names_map.get(month_name_part)
                if month is None:
                    for full_month, num in month_names_map.items():
                        if fuzzy_match(month_name_part, full_month, max_dist=1):
                            month = num
                            break

        # Format: MM/DD (e.g., "8/5")
        if month is None or day is None:
            m_slash = re.search(r"(\d{1,2})/(\d{1,2})", birth_date_str)
            if m_slash:
                month = int(m_slash.group(1))
                day = int(m_slash.group(2))

        # Format: YYYY-MM-DD (e.g., "2023-08-05")
        if month is None or day is None:
            m_dash = re.search(r"(\d{4})-(\d{2})-(\d{2})", birth_date_str)
            if m_dash:
                month = int(m_dash.group(2))
                day = int(m_dash.group(3))

        if month is None or day is None:
            print(f"[DEBUG] get_zodiac_sign: Could not parse date from '{birth_date_str}'")
            return None # Could not parse date

        # Basic validation for day and month
        if not (1 <= month <= 12 and 1 <= day <= 31):
            print(f"[DEBUG] get_zodiac_sign: Invalid month/day combination: Month={month}, Day={day}")
            return None

    except Exception as e:
        print(f"Error parsing date '{birth_date_str}': {e}")
        return None

    # Zodiac sign date ranges (Western)
    zodiac_dates = [
        ("Capricorn", (1, 1), (1, 19)),
        ("Aquarius", (1, 20), (2, 18)),
        ("Pisces", (2, 19), (3, 20)),
        ("Aries", (3, 21), (4, 19)),
        ("Taurus", (4, 20), (5, 20)),
        ("Gemini", (5, 21), (6, 20)),
        ("Cancer", (6, 21), (7, 22)),
        ("Leo", (7, 23), (8, 22)),
        ("Virgo", (8, 23), (9, 22)),
        ("Libra", (9, 23), (10, 22)),
        ("Scorpio", (10, 23), (11, 21)),
        ("Sagittarius", (11, 22), (12, 21)),
        ("Capricorn", (12, 22), (12, 31)), # Capricorn wraps around the year end
    ]

    for sign, (start_m, start_d), (end_m, end_d) in zodiac_dates:
        if (start_m <= month <= end_m and \
            (month != start_m or day >= start_d) and \
            (month != end_m or day <= end_d)) or \
           (start_m > end_m and \
            (month >= start_m or month <= end_m)): # Handles wrap-around like Capricorn
            print(f"[DEBUG] get_zodiac_sign: Successfully determined sign '{sign}' for Month={month}, Day={day}")
            return sign
    print(f"[DEBUG] get_zodiac_sign: No zodiac sign found for Month={month}, Day={day}")
    return None

def load_astrology_data(file_path):
    """Loads astrology data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Astrology data file '{file_path}' not found.")
        print("Please ensure 'astrology_data.txt' is in the same directory as the script.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'. Check file format.")
        exit()

def prepare_documents(astrology_data):
    """Prepares Langchain Document objects from the astrology data."""
    documents = []
    # Handle the general zodiac definition separately
    astrology_data_copy = astrology_data.copy() # Create a copy to avoid modifying the original dict during iteration
    if "Zodiac Signs General" in astrology_data_copy:
        general_data = astrology_data_copy["Zodiac Signs General"]
        # Include all general astrology related information in one document
        general_content = (
            f"What are Zodiac Signs? {general_data.get('definition', '')}\n"
            f"About Astrology: {general_data.get('about_astrology', '')}\n"
            f"How Astrology Works: {general_data.get('how_it_works', '')}"
        )
        documents.append(Document(
            page_content=general_content,
            metadata={"sign": "Zodiac Signs General"}
        ))
        del astrology_data_copy["Zodiac Signs General"] # Remove after processing to avoid errors in the loop below

    # Process individual zodiac signs
    for sign, data in astrology_data_copy.items():
        content = (
            f"Zodiac Sign: {sign}\n"
            f"Dates: {data.get('dates', 'N/A')}\n"
            f"Symbol: {data.get('symbol', 'N/A')}\n"
            f"Element: {data.get('element', 'N/A')}\n"
            f"Ruling Planet: {data.get('ruling_planet', 'N/A')}\n"
            f"Key Traits: {data.get('traits', 'N/A')}\n"
            f"Compatibility: {data.get('compatibility', 'N/A')}\n"
            f"General Outlook: {data.get('outlook', 'N/A')}"
        )
        # Add yearly forecasts if available
        if 'yearly_forecasts' in data and isinstance(data['yearly_forecasts'], dict):
            for year, forecast_data in data['yearly_forecasts'].items():
                content += f"\nYearly Forecast {year}: General: {forecast_data.get('general', 'N/A')}, Love: {forecast_data.get('love', 'N/A')}, Career: {forecast_data.get('career', 'N/A')}"
        
        # Add daily horoscope template and general love compatibility if available
        if 'daily_horoscope_template' in data:
            content += f"\nDaily Horoscope Template: {data.get('daily_horoscope_template', 'N/A')}"
        if 'love_compatibility_general' in data:
            content += f"\nGeneral Love Compatibility: {data.get('love_compatibility_general', 'N/A')}"
        if 'career_outlook' in data:
            content += f"\nCareer Outlook: {data.get('career_outlook', 'N/A')}"

        documents.append(Document(page_content=content, metadata={"sign": sign}))
    return documents

def setup_vector_store(documents, db_directory):
    """Initializes embeddings and sets up the Chroma vector store."""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except ValueError as e:
        print(f"Error initializing embeddings: {e}")
        print("Please set your GOOGLE_API_KEY environment variable.")
        exit()

    # Check if vector store already exists
    chroma_db_path = os.path.join(db_directory, "chroma.sqlite3")
    
    if os.path.exists(chroma_db_path):
        print(f"Loading existing vector store from {db_directory}")
        vectorstore = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    else:
        print(f"Creating new vector store at {db_directory}")
        # Split documents for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_directory
        )
        vectorstore.persist() # Save the vector store to disk
        print(f"Vector store created and persisted at {db_directory}")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Changed k from 2 to 4
    return retriever

def setup_conversational_chain(retriever):
    """Sets up the Langchain ConversationalRetrievalChain."""
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest", temperature=0.7)

    # Define Custom Prompt Templates
    question_generator_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    Chat History: {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

    # Refined QA Template
    qa_template = """You are an astrology bot. Your primary goal is to provide accurate and helpful information about zodiac signs and general astrology.
    Answer the user's question ONLY by strictly referring to the following context:
    {context}
    If the context directly contains the answer, provide it clearly and concisely.
    If the context does NOT contain enough information to answer the specific question, state clearly that you cannot answer based on the provided information. Then, suggest asking about a specific zodiac sign (e.g., "Tell me about Aries traits" or "What is the daily horoscope for Leo?") or a more general astrology topic. Do not invent information.
    Question: {question}
    Answer:"""

    #Memory is handled per session in the /chat endpoint, not globally here
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(qa_template)},
        condense_question_prompt=PromptTemplate.from_template(question_generator_template),
        return_source_documents=False
    )
    return chain

def initialize_bot():
    """Initializes the bot components once when the Flask app starts."""
    global qa_chain, ASTROLOGY_DATA
    print("Initializing Astrology Bot components...")
    ASTROLOGY_DATA = load_astrology_data(ASTROLOGY_DATA_FILE)
    documents = prepare_documents(ASTROLOGY_DATA)
    retriever = setup_vector_store(documents, VECTOR_DB_DIRECTORY)
    qa_chain = setup_conversational_chain(retriever)
    print("Astrology Bot initialized and ready.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin/stats')
def admin_stats():
    """Get database statistics."""
    stats = db.get_session_stats()
    sessions = db.get_all_sessions()
    return jsonify({
        "stats": stats,
        "recent_sessions": sessions[:10]
    })

@app.route('/admin/session/<session_id>')
def get_session_history(session_id):
    """Get full conversation history for a specific session."""
    history = db.get_conversation_history(session_id, limit=1000)
    return jsonify({"session_id": session_id, "history": history})

@app.route('/admin/cleanup', methods=['POST'])
def cleanup_old_sessions():
    """Clean up sessions older than 30 days."""
    deleted_count = db.cleanup_old_sessions(30)
    return jsonify({"message": f"Deleted {deleted_count} old sessions"})

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the frontend."""
    data = request.get_json()
    user_message = data.get('message')
    session_id = data.get('session_id')

    print(f"\n[DEBUG] Received message: '{user_message}' for session: '{session_id}'")

    if not user_message or not session_id:
        return jsonify({"error": "Message and session_id are required"}), 400

    # Get or create memory and facts for the session
    if session_id not in conversation_memories:
        history = db.get_conversation_history(session_id)
        session_facts = {}
        for msg in history:
            if msg["type"] == "user":
                session_facts.update(extract_user_facts(msg["content"]))
        
        # Ensure zodiac sign is computed for loaded history
        if 'birth_date' in session_facts:
            sign = get_zodiac_sign(session_facts['birth_date'])
            if sign:
                session_facts['zodiac_sign'] = sign

        conversation_memories[session_id] = {
            'memory': ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key='answer'
            ),
            'facts': session_facts # Store extracted facts from history
        }
        # Restore conversation history into memory after facts are extracted
        for msg in history:
            if msg["type"] == "user":
                conversation_memories[session_id]['memory'].chat_memory.add_user_message(msg["content"])
            elif msg["type"] == "assistant":
                conversation_memories[session_id]['memory'].chat_memory.add_ai_message(msg["content"])
        print(f"[DEBUG] Session loaded: {session_id} with {len(history)} messages and facts: {session_facts}")

    current_memory = conversation_memories[session_id]['memory']
    user_facts = conversation_memories[session_id]['facts'] # This now holds all accumulated facts

    try:
        # Extract and remember new user facts from the current message
        new_facts = extract_user_facts(user_message)
        if new_facts:
            user_facts.update(new_facts)
            # If birth_date was updated or newly added, recompute zodiac sign
            if 'birth_date' in new_facts:
                sign = get_zodiac_sign(user_facts['birth_date'])
                if sign:
                    user_facts['zodiac_sign'] = sign
                else:
                    # Clear zodiac sign if birth_date is present but invalid
                    user_facts.pop('zodiac_sign', None)
        print(f"[DEBUG] User facts after current message processing: {user_facts}")


        # --- Direct fact query handling (highest priority) ---
        fact_response = None
        user_message_lower = user_message.lower()
        user_message_clean = re.sub(r'[^a-z0-9 ]', '', user_message_lower) # Cleaned for fuzzy matching

        # Check for direct fact queries and set fact_response if found
        if 'name' in user_facts and (
            fuzzy_match("what is my name", user_message_clean) or
            fuzzy_match("tell me my name", user_message_clean) or
            fuzzy_match("who am i", user_message_clean)
        ):
            fact_response = f"Your Name is {user_facts['name']}."
        
        elif 'birth_date' in user_facts and (
            fuzzy_match("when was i born", user_message_clean) or
            fuzzy_match("when is my birthday", user_message_clean) or
            fuzzy_match("my birth date", user_message_clean)
        ):
            fact_response = f"You were born on {user_facts['birth_date']}."

        elif any(fuzzy_match(pat, user_message_clean) for pat in zodiac_patterns) and 'birth_date' in user_facts:
            sign = get_zodiac_sign(user_facts['birth_date'])
            if sign:
                fact_response = f"Your Zodiac Sign is {sign}."
            else:
                fact_response = "I know your birth date, but I couldn't determine your zodiac sign from it. Please ensure the date format is clear (e.g., 'August 5th', '8/5', '1990-08-05')."

        # If a direct fact query was matched, respond immediately and return
        if fact_response:
            current_memory.chat_memory.add_user_message(user_message)
            current_memory.chat_memory.add_ai_message(fact_response)
            db.add_message(session_id, "user", user_message)
            db.add_message(session_id, "assistant", fact_response)
            print(f"[DEBUG] Direct fact query response: {fact_response}")
            return jsonify({"response": fact_response})

        # --- Acknowledgement and Greeting handling (lower priority than direct queries) ---
        bot_response = None
        acknowledgement_parts = []
        is_greeting = fuzzy_match("hello", user_message_clean, max_dist=0) or fuzzy_match("hi", user_message_clean, max_dist=0)

        if is_greeting:
            if 'name' in user_facts:
                acknowledgement_parts.append(f"Hello, {user_facts['name']}!")
            else:
                acknowledgement_parts.append("Hello there!")

        # Acknowledge newly provided facts (name or birthday) if they were just stated
        # Only acknowledge if new_facts has content and the original message wasn't a direct query
        if new_facts and not any(fuzzy_match(pat, user_message_clean) for pat in ["what is", "when was", "tell me", "who am i"]):
            if 'name' in new_facts and 'name' in user_facts:
                # Avoid re-acknowledging name if it was part of a greeting already
                if not (is_greeting and f"Hello, {user_facts['name']}!" in acknowledgement_parts):
                    acknowledgement_parts.append(f"Got it, your name is {user_facts['name']}.")
            
            if 'birth_date' in new_facts and 'birth_date' in user_facts:
                date_ack = f"I've noted your birthday is on {user_facts['birth_date']}."
                if 'zodiac_sign' in user_facts:
                    date_ack += f" That makes you a {user_facts['zodiac_sign']}."
                else:
                    date_ack += " I couldn't determine your zodiac sign from that date, please ensure it's clear."
                acknowledgement_parts.append(date_ack)
        
        if acknowledgement_parts:
            bot_response = " ".join(acknowledgement_parts).strip()
            # Add a general follow-up question if the response is purely an acknowledgement or greeting
            if not bot_response.endswith((".", "?", "!")): # Avoid adding question if response already ends with punctuation
                bot_response += ". How can I help you with astrology today?"
            
            current_memory.chat_memory.add_user_message(user_message)
            current_memory.chat_memory.add_ai_message(bot_response)
            db.add_message(session_id, "user", user_message)
            db.add_message(session_id, "assistant", bot_response)
            print(f"[DEBUG] Acknowledgement/Greeting response: {bot_response}")
            return jsonify({"response": bot_response})

        # --- End direct fact recall/acknowledgement ---

        # If no direct response or acknowledgement was generated, proceed to LLM invocation
        context_prefix = ""
        if user_facts:
            context_lines = [f"{k.replace('_',' ').title()}: {v}" for k, v in user_facts.items()]
            context_prefix = "\n".join(context_lines) + "\n"
        print(f"[DEBUG] Context prefix for LLM: {context_prefix}")

        # If user asks about horoscope and we know their sign, add a hint
        if 'horoscope' in user_message_lower or 'my sign' in user_message_lower:
            if 'zodiac_sign' in user_facts:
                context_prefix += f"\nUser's Zodiac Sign: {user_facts['zodiac_sign']}\n"

        # Invoke the chain with the current message and session-specific memory
        response = qa_chain.invoke(
            {"question": context_prefix + user_message, "chat_history": current_memory.buffer}
        )
        bot_response = response['answer']

        # Add user message and bot response to the session's memory
        current_memory.chat_memory.add_user_message(user_message)
        current_memory.chat_memory.add_ai_message(bot_response)

        # Save messages to database
        db.add_message(session_id, "user", user_message)
        db.add_message(session_id, "assistant", bot_response)
        print(f"[DEBUG] LLM response: {bot_response}")

        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error during chat processing for session {session_id}: {e}")
        return jsonify({"error": "An internal error occurred. Please try again."}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Resets the conversation for the given session_id: clears memory and deletes from DB."""
    data = request.get_json()
    session_id = data.get('session_id')

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Remove from in-memory
    conversation_memories.pop(session_id, None)
    # Remove from DB
    db.delete_session(session_id)

    return jsonify({"message": "Session reset."})

if __name__ == '__main__':
    # Ensure the chroma_db directory exists
    os.makedirs(VECTOR_DB_DIRECTORY, exist_ok=True)
    
    # Load astrology data and set up vector store and conversational chain
    # This should only happen once when the application starts
    print("Initializing Astrology Bot components...")
    ASTROLOGY_DATA = load_astrology_data(ASTROLOGY_DATA_FILE)
    documents = prepare_documents(ASTROLOGY_DATA)
    retriever = setup_vector_store(documents, VECTOR_DB_DIRECTORY)
    qa_chain = setup_conversational_chain(retriever)
    print("Astrology Bot initialized and ready.")

    app.run(debug=True)
