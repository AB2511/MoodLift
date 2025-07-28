# MoodLift: Holistic Mental Health Companion

## Overview
MoodLift is a personalized mental well-being companion designed to support users in tracking their moods, gaining community insights, receiving tailored suggestions, accessing mood-specific resources, visualizing trends, and predicting future moods. Built for the TiDB AgentX Hackathon 2025, it leverages TiDB Serverless with vector search to create an innovative, multi-step AI agent workflow.

## Features
- **Mood Logging**: Log your current mood (Happy, Sad, Anxious, Calm) with optional journal entries, stored with NLP-generated embeddings.
- **Community Insights**: Discover common moods based on similar journal entries from others.
- **Personalized Suggestions**: Receive tailored ideas to enhance your mood (e.g., journaling for Calm).
- **Recommended Resources**: Access mood-specific activities or articles (e.g., "Listen to soothing classical music" for Calm).
- **Mood Trends**: Visualize your mood journey over time with a line chart.
- **Mood Prediction**: Predict your next mood with confidence based on recent trends.

## Data Flow and Integrations
- **Ingest & Index Data**: User moods and journal entries are ingested into TiDB Serverless, with embeddings generated using the `sentence-transformers/all-MiniLM-L6-v2` model and stored as vectors.
- **Search Data**: Queries TiDB Serverless using vector search to find similar journal entries and full-text search for resource recommendations.
- **Chain LLM Calls**: Analyzes embeddings to provide personalized suggestions and predictions.
- **Multi-Step Flow**: Automates from mood input to resource recommendation in a single workflow.

## Prerequisites
- Python 3.8+
- TiDB Cloud account (free tier or trial)
- Required libraries: `streamlit`, `pandas`, `mysql-connector-python`, `transformers`, `torch`, `numpy`, `python-dotenv`

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AB2511/MoodLift.git
   cd MoodLift
   ```

2. **Install Dependencies**:
   - Create a `requirements.txt` file with:
     ```
     streamlit
     pandas
     mysql-connector-python
     transformers
     torch
     numpy
     python-dotenv
     ```
   - Install them:
     ```bash
     pip install -r requirements.txt
     ```

3. **Configure TiDB Cloud**:
   - Sign up for a free TiDB Cloud account at [https://tidbcloud.com/free-trial/](https://tidbcloud.com/free-trial/).
   - Create a Serverless cluster and note the connection details (host, port, user, password, database name).
   - Set up the `resources` and `mood_logs` tables:
     ```sql
     CREATE TABLE mood_logs (
         id INT AUTO_INCREMENT PRIMARY KEY,
         timestamp DATETIME,
         mood VARCHAR(50),
         journal TEXT,
         embedding BLOB
     );
     CREATE TABLE resources (
         id INT AUTO_INCREMENT PRIMARY KEY,
         type VARCHAR(50),
         content TEXT,
         mood VARCHAR(50) DEFAULT NULL
     );
     ```
   - Populate `resources` with sample data (e.g., from your JSON).

4. **Set Environment Variables**:
   - Create a `.env` file in the project directory with:
     ```
     TIDB_HOST=your_host
     TIDB_PORT=your_port
     TIDB_USER=your_user
     TIDB_PASSWORD=your_password
     TIDB_DB_NAME=your_database
     CA_PATH=path_to_ca_cert (if using SSL)
     ```
   - Keep `.env` out of Git by adding it to `.gitignore`.

5. **Run the App**:
   ```bash
   streamlit run app.py
   ```
   - Open the URL (e.g., `http://localhost:8501`) in your browser.

## Troubleshooting
- **Connection Errors**: Verify `.env` values and TiDB Cloud connection.
- **No Resources**: Ensure `resources` table has data.
- **Embedding Issues**: Check internet connectivity for model download.

## License
This project is released under the MIT License (for Best Open Source Award eligibility).
