import sqlite3
import hashlib
import uuid
# Hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
# Authenticate user and return session ID
def authenticate_user(username, password):
    conn = init_db()
    hashed_password = hash_password(password)
    
    # Check if user exists
    user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password)).fetchone()
    
    if user:
        # Create or update session ID
        session_id = str(uuid.uuid4())
        conn.execute('UPDATE users SET session_id = ? WHERE id = ?', (session_id, user[0]))
        conn.commit()
        conn.close()
        return session_id  # Return session ID
    conn.close()
    return None
# Register new user
def register_user(username, password):
    conn = init_db()
    hashed_password = hash_password(password)
    try:
        conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()
        return "User registered successfully!"
    except sqlite3.IntegrityError:
        return "Username already exists."
# Get user by session ID
def get_user_by_session(session_id):
    conn = init_db()
    result = conn.execute('SELECT id FROM users WHERE session_id = ?', (session_id,)).fetchone()
    conn.close()
    if result is not None:
        user_id = result[0]  # Extract the integer user_id from the tuple
        return user_id
    else:
        return None  # Handle the case where no user is found
# Log chat messages and responses
def log_chat(user_id, session_id, message, response):
    conn = init_db()
    conn.execute(
        'INSERT INTO chat_logs (user_id, session_id, message, response) VALUES (?, ?, ?, ?)',
        (user_id, session_id, message, response)
    )
    conn.commit()
    conn.close()
# Get chat history for a user session
def get_chat_history(session_id):
    conn = init_db()
    logs = conn.execute('SELECT message, response, timestamp FROM chat_logs WHERE session_id = ?', (session_id,)).fetchall()
    conn.close()
    return logs
def init_db():
    conn = sqlite3.connect('chat_logs.db')
    return conn
# Function to create necessary tables (users and chat_logs)
def create_tables():
    conn = init_db()
    # Create users table if it doesn't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            session_id TEXT
        )
    ''')
    
    # Create chat logs table if it doesn't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT,
            message TEXT,
            response TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()
if __name__ == "__main__":
    create_tables()
    print("Tables created successfully!")