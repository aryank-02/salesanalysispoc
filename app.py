import streamlit as st
import google.generativeai as genai
import os
import tempfile

# --- System Prompt for the LLM (Gemini) ---
SYSTEM_PROMPT = """
You are an expert sales call analyst. Your task is to analyze the transcription of a sales call and provide a detailed evaluation.

You must rate the call on a scale of 1 to 10 for each of the five key parameters listed below. For each parameter, provide a score and a brief, constructive justification for that score. Finally, provide an overall call score and a summary of the call's strengths and areas for improvement.

Structure your output in Markdown format.

## Sales Call Analysis

**Overall Call Score:** [Provide a single score from 1-10]

**Summary:**
*   **Strengths:** [List 2-3 key strengths of the call]
*   **Areas for Improvement:** [List 2-3 specific areas for improvement]

---

### Detailed Parameter Analysis

**1. Clarity of Pitch:**
*   **Score:** [Score from 1-10]
*   **Justification:** [Explain why you gave this score. Was the value proposition clear? Was the language concise and easy to understand?]

**2. Objection Handling:**
*   **Score:** [Score from 1-10]
*   **Justification:** [Explain why you gave this score. Did the sales rep effectively address customer concerns? Did they acknowledge the objection before responding?]

**3. Engagement Level:**
*   **Score:** [Score from 1-10]
*   **Justification:** [Explain why you gave this score. Did the rep build rapport? Was it a two-way conversation, or a monologue? Did they ask good questions?]

**4. Closing Technique:**
*   **Score:** [Score from 1-10]
*   **Justification:** [Explain why you gave this score. Did the rep attempt to close or define next steps? Was the closing natural and confident?]

**5. Use of Product USPs (Unique Selling Propositions):**
*   **Score:** [Score from 1-10]
*   **Justification:** [Explain why you gave this score. Were the product's key benefits mentioned and linked to the customer's needs?]
"""

# --- Streamlit App Interface ---

st.set_page_config(page_title="AI Sales Call Analyzer", layout="wide")

st.title("🤖 AI Sales Call Analysis (Powered by Gemini 2.5 Flash)")
st.markdown("Upload a recorded sales call to receive a full transcription and a detailed performance analysis.")

# --- API Key Handling for Deployment ---
# Try to get the key from Vercel's environment variables
google_api_key = os.environ.get("GOOGLE_AI_API_KEY")

# If not found (e.g., running locally), use Streamlit's sidebar input
if not google_api_key:
    st.sidebar.header("Configuration")
    google_api_key = st.sidebar.text_input("Enter your Google AI API Key", type="password")

# Configure Google AI client if the key is available
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
    except Exception as e:
        st.error(f"Failed to configure Google AI client: {e}")
        st.stop()
else:
    st.warning("Please provide your Google AI API Key to use the app.")
    st.stop()


# Function to transcribe audio using Gemini's multimodal capabilities
def transcribe_audio_gemini(uploaded_audio_file):
    with st.spinner('Uploading and transcribing with Gemini 2.5 Flash...'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_audio_file.getvalue())
                tmp_file_path = tmp_file.name
            
            audio_file = genai.upload_file(
                path=tmp_file_path,
                display_name=uploaded_audio_file.name
            )
            
            os.remove(tmp_file_path)
            
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            response = model.generate_content(
                ["Please provide a clean and accurate transcription of this audio file.", audio_file],
                request_options={'timeout': 600}
            )
            return response.text
        except Exception as e:
            st.error(f"Error during transcription: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            return None

# Function to analyze transcript using Google Gemini
def analyze_transcript_gemini(transcript):
    with st.spinner('Analyzing transcript with Gemini 2.5 Flash...'):
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            full_prompt = f"{SYSTEM_PROMPT}\n\nPlease analyze the following sales call transcript:\n\n---\n\n{transcript}"
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            return None

# Main App Body
st.divider()
uploaded_file = st.file_uploader("Upload an audio file (e.g., MP3, WAV, M4A)", type=['mp3', 'wav', 'm4a', 'mpeg'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/mp3')
    if st.button("Analyze Call", type="primary"):
        transcript_text = transcribe_audio_gemini(uploaded_file)
        if transcript_text:
            with st.expander("View Full Call Transcription"):
                st.text_area("Transcription", transcript_text, height=250)
            analysis_result = analyze_transcript_gemini(transcript_text)
            if analysis_result:
                st.subheader("📊 Analysis Report (Generated by Gemini 2.5 Flash)")
                st.markdown(analysis_result)

