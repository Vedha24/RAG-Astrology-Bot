import sqlite3
import json
from datetime import datetime
import os

class ConversationDatabase:
    def __init__(self, db_path="conversations.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,  -- 'user' or 'assistant'
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Add index on session_id for fast lookup
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id ON conversations(session_id)
        ''')
        
        # Create sessions table to track session metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                message_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_message(self, session_id, message_type, message):
        """Add a message to the conversation history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert the message
        cursor.execute('''
            INSERT INTO conversations (session_id, message_type, message)
            VALUES (?, ?, ?)
        ''', (session_id, message_type, message))
        
        # Update or create session record
        cursor.execute('''
            INSERT OR REPLACE INTO sessions (session_id, created_at, last_activity, message_count)
            VALUES (?, 
                    COALESCE((SELECT created_at FROM sessions WHERE session_id = ?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP,
                    COALESCE((SELECT message_count FROM sessions WHERE session_id = ?), 0) + 1)
        ''', (session_id, session_id, session_id))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id, limit=50):
        """Retrieve conversation history for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT message_type, message, timestamp
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        messages = cursor.fetchall()
        conn.close()
        
        return [{"type": msg[0], "content": msg[1], "timestamp": msg[2]} for msg in messages]
    
    def get_all_sessions(self):
        """Get all session IDs with their metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, created_at, last_activity, message_count
            FROM sessions
            ORDER BY last_activity DESC
        ''')
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [{"session_id": s[0], "created_at": s[1], "last_activity": s[2], "message_count": s[3]} for s in sessions]
    
    def delete_session(self, session_id):
        """Delete all messages for a specific session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_sessions(self, days_old=30):
        """Delete sessions older than specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM conversations 
            WHERE session_id IN (
                SELECT session_id FROM sessions 
                WHERE last_activity < datetime('now', '-{} days')
            )
        '''.format(days_old))
        
        cursor.execute('''
            DELETE FROM sessions 
            WHERE last_activity < datetime('now', '-{} days')
        '''.format(days_old))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def get_session_stats(self):
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM sessions')
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM conversations')
        total_messages = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM sessions WHERE last_activity > datetime("now", "-1 day")')
        active_sessions_24h = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "active_sessions_24h": active_sessions_24h
        }
