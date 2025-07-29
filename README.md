# RAG Astrology Bot

A conversational AI-powered astrology assistant that provides users with personalized astrological insights and guidance. This bot leverages Retrieval-Augmented Generation (RAG) to answer user queries using a combination of stored astrology data and dynamic conversation management.

## Features

- **Chat-based Interface:** Ask astrology-related questions and get instant, personalized answers.
- **Persistent Conversation Storage:** All chat sessions are stored in a database for continuity and analytics.
- **Astrology Data Integration:** Uses curated astrology datasets for accurate and insightful responses.
- **Web Application:** Built with Flask and HTML templates for easy deployment and user interaction.
- **Easy to Extend:** Modular codebase for adding new features or integrating with other AI/ML models.

## Technologies Used

- Python 3
- Flask
- SQLite
- HTML/CSS

## Getting Started

### Prerequisites
- Python 3.x
- pip (Python package manager)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vedha24/RAG-Astrology-Bot.git
   cd RAG-Astrology-Bot
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in the required values (if applicable).

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Visit the app:**
   Open your browser and go to `http://localhost:5000`

## Project Structure

```
Astrology Bot/
├── app.py                  # Main Flask application
├── database.py             # Database logic for storing conversations
├── astrology_data.txt      # Astrology dataset
├── conversations.db        # SQLite database for chat history
├── templates/
│   ├── index.html          # Main chat UI
│   └── admin_dashboard.html# Admin dashboard (if implemented)
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies

```

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments
- Inspired by advances in Retrieval-Augmented Generation (RAG) and conversational AI.
- Built with the help of open-source tools and the Python community.
