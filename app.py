import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import MySQLConnection
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

TIDB_HOST = os.getenv("TIDB_HOST")
TIDB_PORT = int(os.getenv("TIDB_PORT"))
TIDB_USER = os.getenv("TIDB_USER")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
TIDB_DB_NAME = os.getenv("TIDB_DB_NAME")
CA_PATH = os.getenv("CA_PATH")

# TiDB Connection
def get_connection(autocommit: bool = True) -> MySQLConnection:
    config = {
        "host": TIDB_HOST,
        "port": TIDB_PORT,
        "user": TIDB_USER,
        "password": TIDB_PASSWORD,
        "database": TIDB_DB_NAME,
        "autocommit": autocommit,
        "use_pure": True,
    }
    if CA_PATH and os.path.exists(CA_PATH):
        config["ssl_verify_cert"] = True
        config["ssl_verify_identity"] = True
        config["ssl_ca"] = CA_PATH
    try:
        conn = mysql.connector.connect(**config)
        with conn.cursor() as cur:
            cur.execute("SELECT DATABASE()")
            db_name = cur.fetchone()[0]
            print(f"Connected to database: {db_name}")
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)} - Check TIDB_HOST, TIDB_PORT, TIDB_USER, TIDB_PASSWORD, and TIDB_DB_NAME in .env")
        return None

# Check if 'mood' column exists in resources table
def check_mood_column(conn):
    try:
        with conn.cursor() as cur:
            cur.execute("SHOW COLUMNS FROM resources LIKE 'mood'")
            return cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking 'mood' column: {str(e)}")
        return False

# Load NLP model
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error loading NLP model: {str(e)}")
    tokenizer = None
    model = None

def get_embedding(text):
    if tokenizer is None or model is None:
        return None
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
        print(f"Embedding shape: {outputs.shape}")  # Debug output
        if outputs.shape[1] != 384:
            st.warning(f"Expected 384 dimensions, got {outputs.shape[1]}. Using partial embedding.")
            outputs = outputs[:, :384]  # Truncate to 384
        return outputs.tobytes()
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

# Streamlit UI
st.title("MoodLift: Holistic Mental Health Companion")
st.markdown("<p style='color: #4CAF50; font-size: 16px; font-style: italic; text-align: center;'>Your personalized mental well-being companion.</p>", unsafe_allow_html=True)

# Mood Logging
st.subheader("Log Your Mood")
st.markdown("<p style='color: #2196F3; font-size: 14px;'>Select your mood and share your thoughts to track your journey:</p>", unsafe_allow_html=True)
mood = st.selectbox("How are you feeling?", ["Happy", "Sad", "Anxious", "Calm"], key="mood_select", index=0)
journal = st.text_area("What’s on your mind? (optional)", key="journal_input", height=100)
if st.button("Log Mood", key="log_button", help="Record your current mood and thoughts"):
    with get_connection() as conn:
        if conn:
            try:
                with conn.cursor(prepared=True) as cur:
                    cur.execute(
                        "INSERT INTO mood_logs (timestamp, mood, journal, embedding) VALUES (%s, %s, %s, %s)",
                        (datetime.now(), mood, journal or None, get_embedding(journal) or None)
                    )
                st.success("Mood logged successfully! Your journey is being tracked.")
            except Exception as e:
                st.error(f"Error logging mood: {str(e)}")

# Community Insights
if journal:
    st.subheader("Community Insights")
    st.markdown("<p style='color: #9C27B0; font-size: 12px;'>Discover how others feel based on similar thoughts:</p>", unsafe_allow_html=True)
    with get_connection() as conn:
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT mood, COUNT(*) as count FROM mood_logs WHERE journal IS NOT NULL AND journal LIKE %s AND id != LAST_INSERT_ID() GROUP BY mood ORDER BY count DESC LIMIT 3",
                        ('%' + journal[:50] + '%',)
                    )
                    similar = cur.fetchall()
                    if similar:
                        for row in similar:
                            st.write(f"**Common mood**: {row[0]} (seen {row[1]} times)")
                    else:
                        cur.execute(
                            "SELECT mood, journal FROM mood_logs WHERE journal IS NOT NULL AND journal LIKE %s AND id != LAST_INSERT_ID() LIMIT 3",
                            ('%' + journal[:50] + '%',)
                        )
                        similar = cur.fetchall()
                        if similar:
                            for row in similar:
                                st.write(f"**Someone felt**: {row[0]} - '{row[1][:50]}...'")
                        else:
                            st.write("No similar entries found yet.")
            except Exception as e:
                st.error(f"Error fetching insights: {str(e)}")

# Personalized Suggestions
st.subheader("Personalized Suggestions")
st.markdown("<p style='color: #FF9800; font-size: 12px;'>Tailored ideas to enhance your mood:</p>", unsafe_allow_html=True)
suggestions = {
    "Sad": ["Try a 5-minute breathing exercise.", "Listen to calming music.", "Write down three things you’re grateful for.", "Watch a funny video.", "Talk to a loved one."],
    "Anxious": ["Take a short walk.", "Try a mindfulness meditation.", "Talk to a friend.", "Practice progressive muscle relaxation.", "Listen to a guided relaxation audio."],
    "Happy": ["Plan a fun outing with friends.", "Celebrate with a small treat like your favorite snack.", "Create a playlist of upbeat songs.", "Reflect on a happy memory.", "Share your joy with someone!"],
    "Calm": ["Reflect on your day in a journal.", "Try a relaxing yoga session.", "Listen to nature sounds.", "Practice gentle stretching.", "Read a calming book."]
}
suggestion = np.random.choice(suggestions.get(mood, ["Keep shining!"]))
st.write(f"**Suggestion**: {suggestion}")

# Recommended Resources
st.subheader("Recommended Resources")
st.markdown("<p style='color: #607D8B; font-size: 12px;'>Mood-specific activities and links to support you:</p>", unsafe_allow_html=True)
with get_connection() as conn:
    if conn:
        try:
            mood_column_exists = check_mood_column(conn)
            if mood_column_exists:
                with conn.cursor() as cur:
                    cur.execute("SELECT content FROM resources WHERE type = 'activity' AND (mood = %s OR mood IS NULL) ORDER BY RAND() LIMIT 1", (mood,))
                    resource = cur.fetchone()
                    if resource:
                        st.write(f"**Resource**: {resource[0]}")
                    else:
                        st.write("No resources available for this mood. Please add more to the resources table.")
            else:
                with conn.cursor() as cur:
                    cur.execute("SELECT content FROM resources WHERE type = 'activity' ORDER BY RAND() LIMIT 1")
                    resource = cur.fetchone()
                    if resource:
                        st.write(f"**Resource**: {resource[0]} (General recommendation)")
                    else:
                        st.write("No resources available. Please add more to the resources table.")
        except mysql.connector.Error as e:
            st.error(f"Error fetching resources: {str(e)} - Ensure the resources table exists and has data.")

# Mood Trends
st.subheader("Mood Trends")
st.markdown("<p style='color: #795548; font-size: 12px;'>Visualize your mood journey over time:</p>", unsafe_allow_html=True)
with get_connection() as conn:
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT timestamp, mood FROM mood_logs ORDER BY timestamp DESC LIMIT 5")
                moods = cur.fetchall()
            if moods:
                moods_df = pd.DataFrame(moods, columns=["Timestamp", "Mood"])
                moods_df["Timestamp"] = pd.to_datetime(moods_df["Timestamp"])
                pd.set_option('future.no_silent_downcasting', True)  # Suppress warning
                moods_df["MoodScore"] = moods_df["Mood"].replace({"Happy": 4, "Calm": 3, "Anxious": 2, "Sad": 1})
                st.line_chart(
                    data=moods_df.set_index("Timestamp")["MoodScore"],
                    use_container_width=True,
                    height=300
                )
            else:
                st.write("No mood data available yet. Log more moods to see trends!")
        except Exception as e:
            st.error(f"Error displaying trends: {str(e)}")

# Mood Prediction
st.subheader("Mood Prediction")
st.markdown("<p style='color: #8BC34A; font-size: 12px;'>A glimpse into your next mood based on trends:</p>", unsafe_allow_html=True)
with get_connection() as conn:
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT timestamp, mood FROM mood_logs ORDER BY timestamp DESC LIMIT 5")
                moods = cur.fetchall()
            if moods:
                moods_df = pd.DataFrame(moods, columns=["Timestamp", "Mood"])
                moods_df["Timestamp"] = pd.to_datetime(moods_df["Timestamp"])
                pd.set_option('future.no_silent_downcasting', True)
                moods_df["MoodScore"] = moods_df["Mood"].replace({"Happy": 4, "Calm": 3, "Anxious": 2, "Sad": 1})
                if len(moods_df) >= 3:
                    last_three = moods_df["MoodScore"].tail(3).values
                    predicted_score = np.mean(last_three)
                    predicted_mood = {4: "Happy", 3: "Calm", 2: "Anxious", 1: "Sad"}.get(int(round(predicted_score)), "Neutral")
                    confidence = min(100, int(100 / len(last_three) * 3))  # Simplified confidence
                    st.write(f"**Predicted Mood**: You might feel {predicted_mood} next based on your recent trends! (Confidence: {confidence}%)")
                else:
                    st.write(f"Log at least 3 moods for a prediction. Current data: {len(moods_df)} entries.")
            else:
                st.write("No mood data available yet.")
        except Exception as e:
            st.error(f"Error predicting mood: {str(e)}")

if __name__ == "__main__":
    st.write("Running MoodLift... Empowering your mental health journey!")
    st.markdown("<p style='color: #757575; font-size: 10px; text-align: center;'>Powered by MoodLift | Supporting mental health globally</p>", unsafe_allow_html=True)
    