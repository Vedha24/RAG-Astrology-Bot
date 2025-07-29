Astrology Bot
A conversational astrology chatbot built using Python and Flask with persistent chat storage. The bot provides astrological insights and interacts with users via a simple web interface, storing all conversations in a SQLite database for later retrieval.

Features
Conversational astrology bot with basic natural language support

Persistent chat history stored in a database (not memory-based)

Web-based interface built with Flask and Jinja2 templates

Easily extensible with additional astrology data

Secure management of environment variables using .env

Tech Stack
Backend: Python 3, Flask

Frontend: HTML (Jinja2 templates)

Database: SQLite (default, can be extended to other DBs)

Others: python-dotenv for environment variables

Project Structure
bash
Copy
Edit
Astrology-Bot/
├── app.py               # Main Flask application
├── database.py          # Database models and chat storage logic
├── astrology_data.txt   # Astrology-related data used by the bot
├── templates/
│   └── index.html       # Frontend UI template
├── .env                 # Environment variables (e.g., Flask secret key)
└── requirements.txt     # Python dependencies
Setup & Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/astrology-bot.git
cd astrology-bot
Create and activate a virtual environment (recommended)

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables

Copy .env.example to .env and set your FLASK_SECRET_KEY.

Initialize the database (optional)

The database is auto-created on first run, but you can manually initialize it:

bash
Copy
Edit
python
>>> from database import init_db
>>> init_db()
>>> exit()
Run the application

bash
Copy
Edit
flask run
Access the app at http://localhost:5000

Usage
Open your browser and start a conversation with the bot.

Chat history will be saved in the SQLite database and persist even after restarts.

You can modify astrology_data.txt to add or update astrology responses.

Customization
Astrology Data: Update astrology_data.txt for new signs or insights.

Database: Replace SQLite with PostgreSQL or MySQL by editing database.py.

Frontend: Modify templates/index.html for custom UI/UX.

Security
Do not commit .env files or sensitive credentials to version control.

Always set a strong FLASK_SECRET_KEY for production deployments.
