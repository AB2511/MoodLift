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
import requests
from dotenv import load_dotenv  # For local development

# Load .env file for local development
load_dotenv()

# Determine environment and load secrets
is_cloud = hasattr(st, 'secrets')
if is_cloud:
    st_secrets = st.secrets
    if not st_secrets:
        st.error("No secrets found in Streamlit Cloud dashboard.")
        st.stop()
else:
    st_secrets = {}  # Empty dict for local testing, fallback to .env

# Extract credentials with fallback to .env for local
TIDB_HOST = st_secrets.get("TIDB_HOST") if is_cloud else os.getenv("TIDB_HOST")
TIDB_PORT = int(st_secrets.get("TIDB_PORT") if is_cloud else os.getenv("TIDB_PORT"))
TIDB_USER = st_secrets.get("TIDB_USER") if is_cloud else os.getenv("TIDB_USER")
TIDB_PASSWORD = st_secrets.get("TIDB_PASSWORD") if is_cloud else os.getenv("TIDB_PASSWORD")
TIDB_DB_NAME = st_secrets.get("TIDB_DB_NAME") if is_cloud else os.getenv("TIDB_DB_NAME")
CA_PATH = st_secrets.get("CA_PATH") if is_cloud else os.getenv("CA_PATH")
SPOTIFY_CLIENT_ID = st_secrets.get("SPOTIFY_CLIENT_ID") if is_cloud else os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = st_secrets.get("SPOTIFY_CLIENT_SECRET") if is_cloud else os.getenv("SPOTIFY_CLIENT_SECRET")

# Validate required credentials
required_secrets = [TIDB_HOST, TIDB_PORT, TIDB_USER, TIDB_PASSWORD, TIDB_DB_NAME, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET]
if not all(required_secrets):
    missing = [key for key, value in [("TIDB_HOST", TIDB_HOST), ("TIDB_PORT", TIDB_PORT), ("TIDB_USER", TIDB_USER), 
                                     ("TIDB_PASSWORD", TIDB_PASSWORD), ("TIDB_DB_NAME", TIDB_DB_NAME), 
                                     ("SPOTIFY_CLIENT_ID", SPOTIFY_CLIENT_ID), ("SPOTIFY_CLIENT_SECRET", SPOTIFY_CLIENT_SECRET)] if not value]
    st.error(f"Missing required secrets: {', '.join(missing)}. Check .env file locally or Streamlit Secrets in cloud.")
    st.stop()

# UI Enhancements
st.set_page_config(page_title="MoodLift", page_icon=":smiley:", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background-color: #2c3e50; /* Darker background */
        color: #ecf0f1; /* Light text for contrast */
    }
    .stButton>button {
        background-color: #27ae60; /* Slightly darker green button */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stSelectbox, .stTextArea, .stWrite {
        color: #ecf0f1; /* Light text in inputs and text areas */
        background-color: #34495e; /* Darker input background */
        border: 1px solid #7f8c8d;
        border-radius: 5px;
    }
    @media (max-width: 600px) {
        .stApp {
            padding: 10px;
        }
        .stSelectbox, .stTextArea {
            width: 100%;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Global variables
tokenizer = None
model = None

# Load NLP model
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to('cpu')
except Exception as e:
    st.error(f"Error loading NLP model: {str(e)}")
    tokenizer = None
    model = None

# Database connection
@st.cache_resource
def get_connection():
    try:
        conn = mysql.connector.connect(
            host=TIDB_HOST,
            port=TIDB_PORT,
            user=TIDB_USER,
            password=TIDB_PASSWORD,
            database=TIDB_DB_NAME,
            ssl_ca=CA_PATH if CA_PATH else None
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# Embedding generation
@st.cache_data
def get_embedding(text):
    if tokenizer is None or model is None:
        return None
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}  # Ensure inputs are on CPU
        outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().detach().numpy()
        print(f"Embedding shape: {outputs.shape}")  # Debug output
        if outputs.shape[1] != 384:
            st.warning(f"Expected 384 dimensions, got {outputs.shape[1]}. Using partial embedding.")
            outputs = outputs[:, :384]  # Truncate to 384
        return outputs.tobytes()
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

# Spotify token
@st.cache_data(ttl=3600)
def get_spotify_token():
    try:
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
        )
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        st.error(f"Error getting Spotify token: {str(e)}")
        return None

# Top tracks
def get_top_mood_tracks(mood):
    token = get_spotify_token()
    if not token:
        return None
    try:
        mood_map = {"Happy": "happy", "Sad": "sad", "Anxious": "anxious", "Calm": "relax"}
        query = f"genre:{mood_map.get(mood, 'relax')} track:popular"
        response = requests.get(
            "https://api.spotify.com/v1/search",
            params={"q": query, "type": "track", "limit": 5},
            headers={"Authorization": f"Bearer {token}"}
        )
        response.raise_for_status()
        tracks = response.json()["tracks"]["items"]
        return [(track["name"], track["artists"][0]["name"], track["external_urls"]["spotify"]) for track in tracks]
    except Exception as e:
        st.error(f"Error fetching tracks: {str(e)}")
        return None

# Main app
st.title("MoodLift: Holistic Mental Health Companion")
st.markdown("<p style='color: #8BC34A; font-size: 12px;'>Your personalized mental well-being companion.</p>", unsafe_allow_html=True)

# Mood logging
st.subheader("Log Your Mood")
st.markdown("<p style='color: #8BC34A; font-size: 12px;'>Select your mood and share your thoughts to track your journey:</p>", unsafe_allow_html=True)
mood = st.selectbox("How are you feeling?", ["Calm", "Happy", "Sad", "Anxious"])
journal = st.text_area("What’s on your mind? (optional)", height=100)

if st.button("Log Mood"):
    with get_connection() as conn:
        if conn:
            try:
                embedding = get_embedding(journal) if journal else get_embedding("No journal entry")
                if embedding:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO mood_logs (timestamp, mood, journal, embedding) VALUES (%s, %s, %s, %s)",
                            (datetime.now(), mood, journal, embedding)
                        )
                        conn.commit()
                    st.success("Mood logged successfully! Your journey is being tracked.")
                else:
                    st.error("Failed to generate embedding.")
            except Exception as e:
                st.error(f"Error logging mood: {str(e)}")

# Personalized suggestions
st.subheader("Personalized Suggestions")
st.markdown("<p style='color: #8BC34A; font-size: 12px;'>Tailored ideas to enhance your mood:</p>", unsafe_allow_html=True)
if mood:
    suggestion = f"Suggestion: {mood.lower() == 'calm' and 'Read a calming book' or 'Try a short walk'}"
    st.write(suggestion)

# Recommended resources
st.subheader("Recommended Resources")
st.markdown("<p style='color: #8BC34A; font-size: 12px;'>Mood-specific activities and links to support you:</p>", unsafe_allow_html=True)
if mood:
    resource = f"Resource: {mood.lower() == 'calm' and 'Do a guided stretching routine' or 'Listen to upbeat music'}"
    st.write(resource)

# Mood trends
st.subheader("Mood Trends")
st.markdown("<p style='color: #8BC34A; font-size: 12px;'>Visualize your mood journey over time:</p>", unsafe_allow_html=True)
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
                st.line_chart(
                    data=moods_df.set_index("Timestamp")["MoodScore"],
                    use_container_width=True,
                    height=300
                )
            else:
                st.write("No mood data available yet. Log more moods to see trends!")
        except Exception as e:
            st.error(f"Error displaying trends: {str(e)}")

# Mood prediction
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
                    confidence = min(100, int(100 / len(last_three) * 3))
                    st.write(f"**Predicted Mood**: You might feel {predicted_mood} next based on your recent trends! (Confidence: {confidence}%)")
                else:
                    st.write(f"Log at least 3 moods for a prediction. Current data: {len(moods_df)} entries.")
            else:
                st.write("No mood data available yet.")
        except Exception as e:
            st.error(f"Error predicting mood: {str(e)}")

# Top 5 tracks
st.subheader("Top 5 Tracks for Your Mood")
top_tracks = get_top_mood_tracks(mood)
if top_tracks:
    for name, artist, url in top_tracks:
        st.write(f"**{name}** by {artist} - [Listen]({url})")
else:
    st.write("No tracks found for this mood.")

if __name__ == "__main__":
    st.markdown("<p style='color: #757575; font-size: 10px; text-align: center;'>Powered by MoodLift | Supporting mental health globally</p>", unsafe_allow_html=True)
    st.write("Running MoodLift... Empowering your mental health journey!")