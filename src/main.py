import streamlit as st
import whisper
import numpy as np
from datetime import datetime, timedelta
import os
from collections import defaultdict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pydub import AudioSegment
import tempfile

def clean_llm_response(response):
    """Clean LLM response by removing metadata and response formatting"""
    if not response:
        return "", 0
        
    text = str(response)
    confidence = 0
    
    # Clean up formatting and metadata
    unwanted_patterns = [
        'additional_kwargs=',
        'response_metadata=',
        'usage_metadata=',
        'content=',
        'Confidence:',
        '\\n',
        '\\r',
        '\n\n'
    ]
    
    for pattern in unwanted_patterns:
        if pattern in text:
            text = text.split(pattern)[0].strip()
    
    # Remove quotes and clean up escaped characters
    text = text.replace("'", "").replace('"', "").strip()
    text = text.replace("\\n", "\n").replace("\\t", "  ")
    
    # Fix bullet points and formatting
    text = text.replace("\\- ", "• ").replace("-", "•")
    text = text.replace("\n\n", "\n").replace("  ", " ")
    
    # Remove any leftover metadata patterns
    text = text.split('token_usage')[0].strip()
    text = text.split('completion_tokens')[0].strip()
    text = text.split('prompt_tokens')[0].strip()
    
    # Final cleanup
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text, confidence



def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        state_vars = {
            'accessibility_settings': {
                'font_size': 'Medium',
                'contrast_mode': 'Standard',
                'show_visual_aids': True,
                'highlight_key_terms': True
            },
            'term_definitions': {},
            'summary_generated': False,
            'current_summary': "",
            'qa_history': [],
            'processed_files': set(),
            'current_file_processed': False,
            'rag_system': None,
            'api_key': None,
            'file_content_processed': False,
            'current_file_name': None,
            'definitions_generated': False,
            'summary_with_confidence': None,
            'messages': [],
            'term_definitions': {},
            'summary_with_confidence': None
        }
        
        for var, value in state_vars.items():
            if var not in st.session_state:
                st.session_state[var] = value
        
        st.session_state.initialized = True

class EnhancedAudioRAG:
    def __init__(self, openai_api_key=None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            self.embeddings = OpenAIEmbeddings()
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        # Initialize Whisper model with appropriate settings for CPU
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
            self.whisper_model = whisper.load_model("base", device="cpu")
        
        self.transcribed_segments = []
        self.current_speaker = None
        self.terminology_bank = defaultdict(list)
        self.key_terms = set()
        self.visual_aids = defaultdict(dict)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vector_store = None

    def process_mp3_file(self, file_path):
        """Process an uploaded audio/video file"""
        try:
            # Check if it's an MP4 file
            is_video = file_path.name.lower().endswith('.mp4')
            
            # Convert MP4 to WAV for transcription if it's a video file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                if is_video:
                    # Extract audio from video
                    audio = AudioSegment.from_file(file_path, format="mp4")
                else:
                    # Handle MP3 as before
                    audio = AudioSegment.from_mp3(file_path)
                    
                audio.export(temp_wav.name, format='wav')
                    
                # Transcribe audio file
                result = self.whisper_model.transcribe(temp_wav.name)
                
                # Process segments with timestamps
                segments = result.get('segments', [])
                for segment in segments:
                    timestamp = str(timedelta(seconds=int(segment['start'])))
                    text = segment['text']
                    
                    # Extract key terms with confidence scores
                    key_terms = self._extract_key_terms(text)
                    
                    # Generate visual aid with video player if it's a video file
                    visual_aid = self._generate_visual_aid(text, file_path if is_video else None)
                    
                    segment_data = {
                        "timestamp": timestamp,
                        "text": text,
                        "speaker": self.current_speaker,
                        "confidence": segment.get('confidence', 0),
                        "key_terms": key_terms,
                        "visual_aid": visual_aid
                    }
                    self.transcribed_segments.append(segment_data)
                    self._update_key_terms([term["term"] for term in key_terms])
                
                # Update vector store
                self.update_vector_store()
                
                os.unlink(temp_wav.name)
                return True
                
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    

    def ask_question(self, question):
        """Ask a question about the transcribed content with confidence score"""
        if not self.vector_store:
            return {"answer": "No content available to answer questions.", "confidence": 0}
            
        try:
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful AI assistant answering questions about audio content.
                Base your answers only on the provided context. If you're uncertain, indicate that in your response.
                Include a confidence score (0-100) at the end of your response in [confidence: X] format.
                If the context doesn't contain relevant information, say so and give a low confidence score."""),
                ("human", "{question}"),
                ("human", "Context: {context}")
            ])
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
                chain_type_kwargs={"prompt": qa_prompt}
            )
            
            response = qa_chain.invoke(question)
            answer = response['result']
            
            confidence_score = 0
            if '[confidence:' in answer:
                try:
                    confidence_str = answer.split('[confidence:')[1].split(']')[0].strip()
                    confidence_score = float(confidence_str)
                    answer = answer.split('[confidence:')[0].strip()
                except:
                    confidence_score = 0
            
            return {
                "answer": answer,
                "confidence": confidence_score,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
    # Continuing the EnhancedAudioRAG class...

    def _extract_key_terms(self, text):
        """Identify key academic terms from text with confidence scores"""
        if not text.strip():
            return []
            
        try:
            response = self.llm.invoke(
                f"""Identify important academic or technical terms from this text.
                For each term, provide a confidence score (0-100) indicating how certain you are it's a key term.
                Return in format: term1 [score], term2 [score], etc.
                
                Text: {text}
                
                Terms:"""
            )
            
            # Parse terms and scores
            terms_with_scores = []
            parts = str(response).split(',')
            for part in parts:
                if '[' in part and ']' in part:
                    term = part.split('[')[0].strip()
                    score = float(part.split('[')[1].split(']')[0].strip())
                    if term and score >= 50:  # Only keep terms with confidence >= 50
                        terms_with_scores.append({"term": term, "confidence": score})
            
            return terms_with_scores
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []

    def _generate_visual_aid(self, text, video_file=None):
        """Display video player for the corresponding segment"""
        if not video_file:
            return None
        
        try:
            # Calculate timestamp in seconds from the text segment
            # Assuming segment timestamps are in format HH:MM:SS
            timestamp = None
            for segment in self.transcribed_segments:
                if text in segment["text"]:
                    time_parts = segment["timestamp"].split(":")
                    timestamp = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                    break
            
            if timestamp is None:
                return None
                
            return {
                "video_file": video_file,
                "start_time": timestamp,
                "confidence": segment.get('confidence', 0) if segment else 0
            }
                
        except Exception as e:
            print(f"Error setting up video player: {e}")
            return None

    def _update_key_terms(self, new_terms):
        """Update the key terms set with new terms"""
        self.key_terms.update(new_terms)

    def update_vector_store(self):
        """Update the vector store with new transcribed content"""
        if not self.transcribed_segments:
            return
        
        text_chunks = []
        for segment in self.transcribed_segments:
            text_chunks.extend(self.text_splitter.split_text(segment["text"]))
        
        if not text_chunks:
            return
            
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(text_chunks, self.embeddings)
        else:
            self.vector_store.add_texts(text_chunks)

    def generate_summary(self):
        """Generate a summary of the content with confidence score"""
        if not self.transcribed_segments:
            return None
            
        try:
            full_text = " ".join([seg["text"] for seg in self.transcribed_segments])
            if not full_text.strip():
                return None
                
            summary_prompt = """Create a clear, structured summary of this content with the following format:
            Main topics:
            
            
            Key points:
            -
            if topic is history then put 
            Important dates:
            -
            if topic is math or science then put
            Formulas:
            -
       
            Include a confidence score (0-100) at the end of your response as [confidence: X].
            
            Content: {text}""".format(text=full_text)
            
            raw_response = self.llm.invoke(summary_prompt)
            summary_text = str(raw_response)
            
            # Remove content= prefix if present
            if 'content=' in summary_text:
                summary_text = summary_text.split('content=')[1]
            
            # Remove metadata
            if 'additional_kwargs=' in summary_text:
                summary_text = summary_text.split('additional_kwargs=')[0]
            
            # Extract confidence score
            confidence = 0
            if '[confidence:' in summary_text:
                try:
                    confidence_str = summary_text.split('[confidence:')[1].split(']')[0].strip()
                    confidence = float(confidence_str)
                    summary_text = summary_text.split('[confidence:')[0].strip()
                except:
                    confidence = 0
            
            # Clean up formatting
            summary_text = summary_text.replace("'", "").replace('"', "")
            summary_text = summary_text.replace("\\n", "\n")
            summary_text = summary_text.strip()
            
            return {
                "summary": summary_text,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return None

    def _extract_key_terms(self, text):
        """Identify key academic terms from text with confidence scores"""
        if not text.strip():
            return []
            
        try:
            term_prompt = f"""Identify important academic or technical terms from this text.
            Return only the terms themselves, one per line.
            Include a confidence score (0-100) after each term in square brackets.
            
            Text: {text}
            """
            
            raw_response = self.llm.invoke(term_prompt)
            terms_text = str(raw_response)
            
            # Remove metadata and formatting
            if 'content=' in terms_text:
                terms_text = terms_text.split('content=')[1]
            if 'additional_kwargs=' in terms_text:
                terms_text = terms_text.split('additional_kwargs=')[0]
            
            terms_text = terms_text.replace("'", "").replace('"', "")
            terms_text = terms_text.replace("\\n", "\n").strip()
            
            # Parse terms and scores
            terms_with_scores = []
            for line in terms_text.split('\n'):
                line = line.strip()
                if '[' in line and ']' in line:
                    term = line.split('[')[0].strip()
                    score = float(line.split('[')[1].split(']')[0].strip())
                    if term and score >= 50:  # Only keep terms with confidence >= 50
                        terms_with_scores.append({"term": term, "confidence": score})
            
            return terms_with_scores
        except Exception as e:
            print(f"Error extracting key terms: {e}")
            return []

    def generate_term_definition(self, term):
        """Generate definition for a term with confidence score"""
        try:
            full_text = " ".join([seg["text"] for seg in self.transcribed_segments])
            if not full_text.strip():
                return None
                
            definition_prompt = f"""Define this specific term from the content below:
            Term to define: {term}

            Provide a clear, concise definition for how this specific term is used in the given context.
            Focus only on information directly related to this term.
            Include a confidence score (0-100) at the end of your response in [confidence: X].
            
            Content: {full_text}"""
            
            raw_response = self.llm.invoke(definition_prompt)
            definition_text = str(raw_response)
            
            # Remove metadata
            if 'content=' in definition_text:
                definition_text = definition_text.split('content=')[1]
            if 'additional_kwargs=' in definition_text:
                definition_text = definition_text.split('additional_kwargs=')[0]
            
            # Extract confidence
            confidence = 0
            if '[confidence:' in definition_text:
                try:
                    confidence_str = definition_text.split('[confidence:')[1].split(']')[0].strip()
                    confidence = float(confidence_str)
                    definition_text = definition_text.split('[confidence:')[0].strip()
                except:
                    confidence = 0
            
            # Clean up formatting
            definition_text = definition_text.replace("'", "").replace('"', "")
            definition_text = definition_text.replace("\\n", "\n")
            definition_text = definition_text.strip()
            
            return {
                "term": term,
                "definition": definition_text,
                "confidence": confidence,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            print(f"Error generating definition: {e}")
            return None
        
    def export_content(self, content_type="transcript"):
        """Export content in various formats"""
        if not self.transcribed_segments:
            return None
            
        if content_type == "transcript":
            return "\n\n".join([
                f"[{seg['timestamp']}]\n{seg['text']}"
                for seg in self.transcribed_segments
            ])
        
        elif content_type == "study_guide":
            if not hasattr(self, 'current_summary'):
                self.current_summary = self.generate_summary()
            
            return f"""AUDIO CONTENT STUDY GUIDE
            
            SUMMARY:
            {self.current_summary['summary'] if self.current_summary else 'No summary available.'}
            
            KEY TERMS:
            {chr(10).join([f'- {term}' for term in sorted(self.key_terms)])}
            """
        
        elif content_type == "glossary":
            glossary_items = []
            for term in sorted(self.key_terms):
                definition = self.generate_term_definition(term)
                if definition:
                    glossary_items.append(
                        f"Term: {term}\n"
                        f"Definition: {definition['definition']}\n"
                        f"Confidence: {definition['confidence']}%"
                    )
            return "\n\n".join(glossary_items)
        
        return None

    # Add these methods inside your EnhancedAudioRAG class


def set_custom_style():
    st.markdown("""
        <style>
        /* Overall app styling */
        .stApp {
            background-color: #161B22;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1B2028;
            padding: 2rem 1rem;
        }

        [data-testid="stSidebar"] .block-container {
            padding: 0 !important;
        }

        /* Title styling */
        h1, h2, h3, h4, h5, h6 {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 500;
            color: rgb(250, 250, 250);
        }

        /* Main title */
        [data-testid="stSidebar"] h1 {
            font-size: 1.5rem !important;
            margin-bottom: 2rem;
        }

        /* Section headers */
        [data-testid="stSidebar"] h3 {
            font-size: 0.875rem !important;
            color: rgb(200, 200, 200);
            margin: 1.5rem 0 0.5rem 0;
        }

        /* API Key input styling */
        .stTextInput input {
            background-color: #0D1117 !important;
            border: 1px solid #30363D !important;
            border-radius: 6px;
            color: white;
            padding: 0.5rem;
            font-size: 0.875rem;
        }

        /* File uploader styling */
        [data-testid="stFileUploader"] {
            background-color: #0D1117;
            border: 1px dashed #30363D;
            border-radius: 6px;
            padding: 1.5rem;
        }

        .uploadedFile {
            background-color: #0D1117;
            border-radius: 6px;
            padding: 0.75rem;
        }

        /* Main content area styling */
        .main .block-container {
            padding: 2rem 1rem !important;
            max-width: none !important;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.25rem;
            padding: 0.5rem;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 2rem;
            padding: 0 1rem;
            color: rgb(148, 148, 148);
            background-color: transparent;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            font-weight: 400;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: rgb(250, 250, 250);
            background-color: rgba(255, 255, 255, 0.1);
        }

        .stTabs [aria-selected="true"] {
            color: rgb(250, 250, 250) !important;
            background-color: rgba(255, 255, 255, 0.1) !important;
            border: none !important;
        }

        /* Remove tab highlight */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* Content cards */
        .stAlert {
            background-color: #0D1117;
            border: 1px solid #30363D;
            border-radius: 6px;
            padding: 1rem;
        }

        /* Slider styling */
        [data-testid="stSlider"] {
            padding: 1rem 0;
        }

        .stSlider [data-baseweb="slider"] {
            margin-top: 1rem;
        }

        /* Transcript styling */
        .transcript-segment {
            padding: 0.5rem;
            border-bottom: none;
        }

        .timestamp {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.5);
        }

        .text {
            color: rgb(250, 250, 250);
            line-height: 1.5;
        }

        /* Chat container */
        .stChatMessage {
            background-color: #0D1117;
            border-radius: 6px;
            padding: 1rem;
            margin: 0.5rem 0;
        }

        /* Chat input */
        .stChatInput {
            border-color: #30363D !important;
        }

        .stChatInput input {
            background-color: #0D1117 !important;
            border-color: #30363D !important;
        }

        /* Button styling */
        .stButton button {
            background-color: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
        }

        .stButton button:hover {
            background-color: #2EA043;
        }
        </style>
    """, unsafe_allow_html=True)

def format_transcript_line(timestamp, text):
    return f"""
        <div class="transcript-line" style="display: flex; align-items: flex-start; padding: 0.5rem 0; gap: 1rem;">
            <div class="timestamp" style="flex-shrink: 0; min-width: 60px; color: rgba(255, 255, 255, 0.5); font-size: 0.75rem;">
                {timestamp}
            </div>
            <div class="transcript-text" style="flex: 1; color: rgb(250, 250, 250); line-height: 1.5;">
                {text}
            </div>
        </div>
    """

def create_streamlit_ui():
    st.set_page_config(page_title="Content Analysis", layout="wide")
    set_custom_style()
    init_session_state()

    # Left sidebar content
    with st.sidebar:
        st.markdown('# Earsforeveryone')
        st.markdown('---')
        
        st.markdown('### OpenAI API Key')
        openai_api_key = st.text_input("", type="password", placeholder="Enter your API key", label_visibility="collapsed")
        if openai_api_key:
            if not st.session_state.rag_system or st.session_state.api_key != openai_api_key:
                st.session_state.rag_system = EnhancedAudioRAG(openai_api_key)
                st.session_state.api_key = openai_api_key
        
        st.markdown('### Upload Content')
        st.markdown('Choose an audio/video file')
        uploaded_file = st.file_uploader("", type=['mp3', 'mp4', 'mpeg4'], label_visibility="collapsed")
        
        # Process uploaded file
        if uploaded_file and not st.session_state.current_file_processed:
            if st.session_state.rag_system:
                with st.spinner("Processing file..."):
                    if st.session_state.rag_system.process_mp3_file(uploaded_file):
                        st.session_state.current_file_processed = True
                        st.session_state.file_content_processed = True
                        st.success("File processed successfully!")
                    else:
                        st.error("Error processing file")
        
        # Layout settings
        st.markdown("### Layout Settings")
        column_width = st.slider(
            "Adjust column width",
            min_value=30,
            max_value=70,
            value=48,
            help="Drag to adjust the width of the main content column"
        )

    # Calculate column ratios based on slider value
    left_ratio = column_width / 100
    right_ratio = 1 - left_ratio

    # Create columns with dynamic ratio
    col1, col2 = st.columns([left_ratio, right_ratio])

    # Main column content
    with col1:
        tab1, tab2 = st.tabs(["Transcript", "Visual aid"])
        

        with tab1:
    if st.session_state.file_content_processed:
        for segment in st.session_state.rag_system.transcribed_segments:
            st.markdown(f"""
                <div class="transcript-line" style="display: flex; align-items: flex-start; padding: 0.5rem 0; gap: 1rem;">
                    <div class="timestamp" style="flex-shrink: 0; min-width: 60px; color: rgba(255, 255, 255, 0.5); font-size: 0.75rem;">
                        {segment['timestamp']}
                    </div>
                    <div class="transcript-text" style="flex: 1; color: rgb(250, 250, 250); line-height: 1.5;">
                        {segment['text']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Upload a file to see transcription")
        
        with tab2:
            if uploaded_file and uploaded_file.name.lower().endswith('.mp4'):
                st.video(uploaded_file)
            else:
                st.info("Upload a video file to view")

    # Features column content
    with col2:
        st.markdown("### Content Analysis")
        feature_tabs = st.tabs(["Chat", "Flashcards", "Summary"])
        
        # Chat tab
        with feature_tabs[0]:
            st.markdown("#### Chat with your content")
            if "messages" not in st.session_state:
                st.session_state.messages = []
                
            # Display chat interface if file is processed
            if st.session_state.file_content_processed:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message["role"] == "assistant" and "confidence" in message:
                            st.progress(message["confidence"] / 100)
                            st.caption(f"Confidence: {message['confidence']}%")
                
                if prompt := st.chat_input("Ask anything about the content..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    if st.session_state.rag_system:
                        response = st.session_state.rag_system.ask_question(prompt)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "confidence": response["confidence"]
                        })
                        st.rerun()
            else:
                st.info("Upload and process a file to start chatting")

        # Flashcards tab
        with feature_tabs[1]:
            st.markdown("#### Key Terms & Definitions")
            if st.session_state.file_content_processed and st.session_state.rag_system:
                if not st.session_state.definitions_generated:
                    with st.spinner("Generating flashcards..."):
                        terms = list(st.session_state.rag_system.key_terms)
                        for term in terms:
                            if term not in st.session_state.term_definitions:
                                definition = st.session_state.rag_system.generate_term_definition(term)
                                if definition:
                                    st.session_state.term_definitions[term] = definition
                        st.session_state.definitions_generated = True

                for term in st.session_state.term_definitions:
                    with st.expander(f"📝 {term}"):
                        definition = st.session_state.term_definitions[term]
                        st.markdown(f"**Definition:** {definition['definition']}")
                        st.progress(definition['confidence'] / 100)
                        st.caption(f"Confidence: {definition['confidence']}%")
            else:
                st.info("Upload a file to generate flashcards")

        # Summary tab
        with feature_tabs[2]:
            st.markdown("#### Content Summary")
            if st.session_state.file_content_processed and st.session_state.rag_system:
                if not st.session_state.summary_generated:
                    with st.spinner("Generating summary..."):
                        summary = st.session_state.rag_system.generate_summary()
                        if summary:
                            st.session_state.summary_with_confidence = summary
                            st.session_state.summary_generated = True

                if st.session_state.summary_with_confidence:
                    st.markdown(st.session_state.summary_with_confidence["summary"])
                    st.progress(st.session_state.summary_with_confidence["confidence"] / 100)
                    st.caption(f"Confidence: {st.session_state.summary_with_confidence['confidence']}%")
            else:
                st.info("Upload a file to generate summary")

def process_upload(uploaded_file):
    if st.session_state.rag_system:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_id != st.session_state.current_file_name:
            with st.spinner("Processing file..."):
                try:
                    if st.session_state.rag_system.process_mp3_file(uploaded_file):
                        st.session_state.current_file_name = file_id
                        st.session_state.file_content_processed = True
                        reset_state()
                        return True
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.file_content_processed = False
    return False

def reset_state():
    st.session_state.summary_generated = False
    st.session_state.current_summary = ""
    st.session_state.qa_history = []
    st.session_state.term_definitions = {}

def handle_chat_question(question):
    with st.spinner("Generating response..."):
        response = st.session_state.rag_system.ask_question(question)
        st.session_state.qa_history.append({
            "question": question,
            "response": response
        })

def render_glossary():
    st.markdown("### Key Terms")
    if st.session_state.rag_system and st.session_state.rag_system.key_terms:
        for term in sorted(st.session_state.rag_system.key_terms):
            with st.expander(term):
                if term in st.session_state.term_definitions:
                    render_term_definition(
                        term,
                        st.session_state.term_definitions[term],
                        st.session_state.accessibility_settings
                    )
                else:
                    st.info("Definition pending")

def generate_summary():
    with st.spinner("Generating summary..."):
        summary_data = st.session_state.rag_system.generate_summary()
        if summary_data:
            st.session_state.current_summary = summary_data
            st.session_state.summary_generated = True

def display_summary():
    st.markdown(f"""
        <div class="card">
            {st.session_state.current_summary['summary']}
            <div class="confidence">
                Confidence: {st.session_state.current_summary['confidence']}%
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    create_streamlit_ui()
