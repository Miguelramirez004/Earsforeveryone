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
    def generate_transcript_timeline(self):
        """Generate an interactive timeline of the transcript"""
        if not self.transcribed_segments:
            return None
            
        try:
            # Create timeline data structure
            timeline_data = []
            for segment in self.transcribed_segments:
                # Convert timestamp to seconds for sorting
                time_parts = segment["timestamp"].split(":")
                seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                
                # Extract key terms if they exist
                key_terms = ", ".join([term["term"] for term in segment.get("key_terms", [])])
                
                timeline_data.append({
                    "timestamp": segment["timestamp"],
                    "seconds": seconds,
                    "text": segment["text"],
                    "key_terms": key_terms,
                    "confidence": segment.get("confidence", 0)
                })

            def render_timeline():
                """Render the timeline visualization"""
                st.markdown("### Transcript Timeline")
                
                # Create timeline settings
                col1, col2 = st.columns([3, 1])
                with col2:
                    timeline_view = st.radio(
                        "View Mode",
                        ["Compact", "Detailed"],
                        key="timeline_view"
                    )
                    show_confidence = st.checkbox("Show Confidence Scores", value=True)
                    show_terms = st.checkbox("Show Key Terms", value=True)

                # Display timeline
                for i, entry in enumerate(timeline_data):
                    # Create a card for each timeline entry
                    with st.container():
                        cols = st.columns([1, 4])
                        
                        # Timestamp column
                        with cols[0]:
                            st.markdown(f"**{entry['timestamp']}**")
                            if show_confidence:
                                confidence = entry["confidence"]
                                st.markdown(
                                    f"<span style='color: {'red' if confidence < 50 else 'orange' if confidence < 75 else 'green'}'>"
                                    f"Confidence: {confidence:.1f}%</span>",
                                    unsafe_allow_html=True
                                )
                        
                        # Content column
                        with cols[1]:
                            if timeline_view == "Detailed":
                                st.markdown(entry["text"])
                                if show_terms and entry["key_terms"]:
                                    st.markdown(f"*Key Terms: {entry['key_terms']}*")
                            else:
                                # Show truncated text for compact view
                                truncated_text = entry["text"][:100] + "..." if len(entry["text"]) > 100 else entry["text"]
                                st.markdown(truncated_text)
                        
                        # Add visual separator between entries
                        if i < len(timeline_data) - 1:
                            st.markdown("<hr style='margin: 5px 0; opacity: 0.3'>", unsafe_allow_html=True)

            return render_timeline
            
        except Exception as e:
            print(f"Error generating timeline: {e}")
            return None

    def process_transcript_timeline(self):
        """Process transcript and create timeline"""
        try:
            timeline_func = self.generate_transcript_timeline()
            if timeline_func:
                return timeline_func
            return None
        except Exception as e:
            print(f"Error processing transcript timeline: {e}")
            return None

def set_custom_style():
    """Set custom CSS styling for the application"""
    st.markdown("""
        <style>
        /* Base colors */
        :root {
            --primary-blue: #1e3a8a;
            --accent-blue: #2563eb;
            --bg-dark: #0f172a;
            --bg-light: #ffffff;
            --text-dark: #000000;
            --text-light: #f8fafc;
            --border-color: #e2e8f0;
        }
        
        /* Content text */
        .content-text {
            color: #000000 !important;
            line-height: 1.5;
            margin-top: 0.5rem;
        }
        
        /* Containers */
        .container {
            background-color: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Visual aid */
        .visual-aid {
            background-color: #f8fafc;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #000000 !important;
        }
        
        /* Timestamp */
        .timestamp {
            color: #64748b;
            font-size: 0.875rem;
        }
        
        /* Key terms */
        .key-term {
            background-color: #fef9c3;
            border-bottom: 2px solid #ca8a04;
            padding: 0 0.25rem;
            border-radius: 2px;
            color: #000000 !important;
        }
        
        /* Font sizes */
        .font-size-small { font-size: 0.875rem; }
        .font-size-medium { font-size: 1rem; }
        .font-size-large { font-size: 1.25rem; }
        .font-size-xlarge { font-size: 1.5rem; }
        
        /* Confidence indicators */
        .confidence-low {
            color: #ef4444;
            font-weight: 500;
        }
        
        .confidence-medium {
            color: #f59e0b;
            font-weight: 500;
        }
        
        .confidence-high {
            color: #10b981;
            font-weight: 500;
        }
        
        /* Card styles */
        .card {
            background-color: #ffffff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        /* Button styles */
        .stButton button {
            width: 100%;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            background-color: var(--accent-blue);
            color: white;
            border: none;
        }
        
        /* High contrast mode */
        .high-contrast {
            background-color: #000000;
            color: #ffffff !important;
        }
        
        .high-contrast .content-text {
            color: #ffffff !important;
        }
        
        /* Dark mode */
        .dark-mode {
            background-color: #1a1a1a;
            color: #ffffff !important;
        }
        
        /* Light mode */
        .light-mode {
            background-color: #ffffff;
            color: #000000 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_transcript_segment(segment, accessibility_settings):
    """Render a single transcript segment with accessibility features"""
    text = segment['text']
    
    # Apply key term highlighting
    if accessibility_settings['highlight_key_terms'] and 'key_terms' in segment:
        for term_data in segment['key_terms']:
            term = term_data['term']
            confidence = term_data['confidence']
            if term and term.strip():
                text = text.replace(
                    term, 
                    f'<span class="key-term" title="Confidence: {confidence}%">{term}</span>'
                )
    
    # Apply font size
    font_size_class = {
        'Small': 'font-size-small',
        'Medium': 'font-size-medium',
        'Large': 'font-size-large',
        'Extra Large': 'font-size-xlarge'
    }[accessibility_settings['font_size']]
    
    # Apply contrast mode
    contrast_class = {
        'Standard': '',
        'High Contrast': 'high-contrast',
        'Dark Mode': 'dark-mode',
        'Light Mode': 'light-mode'
    }[accessibility_settings['contrast_mode']]
    
    st.markdown(f"""
        <div class="container {font_size_class} {contrast_class}">
            <div class="timestamp">{segment['timestamp']}</div>
            <div class="content-text">{text}</div>
            <div class="confidence" style="font-size: 0.8em; color: #666;">
                Transcription Confidence: {segment['confidence']:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_visual_aid(visual_aid_data, accessibility_settings):
    """Render video player or visual aid with confidence score"""
    if not visual_aid_data:
        return
    
    if "video_file" in visual_aid_data:
        # Display video player
        st.video(visual_aid_data["video_file"], start_time=visual_aid_data["start_time"])
        
        # Show confidence score
        confidence = visual_aid_data['confidence']
        confidence_class = (
            'confidence-low' if confidence < 50 else
            'confidence-medium' if confidence < 75 else
            'confidence-high'
        )
        
        st.markdown(f"""
            <div class="{confidence_class}">
                Confidence: {confidence}%
            </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback to original visual aid display if no video
        st.info("No video content available for this segment.")
    
    font_size_class = {
        'Small': 'font-size-small',
        'Medium': 'font-size-medium',
        'Large': 'font-size-large',
        'Extra Large': 'font-size-xlarge'
    }[accessibility_settings['font_size']]
    
    contrast_class = {
        'Standard': '',
        'High Contrast': 'high-contrast',
        'Dark Mode': 'dark-mode',
        'Light Mode': 'light-mode'
    }[accessibility_settings['contrast_mode']]
    
    confidence = visual_aid_data['confidence']
    confidence_class = (
        'confidence-low' if confidence < 50 else
        'confidence-medium' if confidence < 75 else
        'confidence-high'
    )
    
    st.markdown(f"""
        <div class="visual-aid {font_size_class} {contrast_class}">
            <h4>Visual Summary</h4>
            <p class="content-text">{visual_aid_data['description']}</p>
            <div class="{confidence_class}">
                Confidence: {confidence}%
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_qa_response(qa_item, accessibility_settings):
    """Render a Q&A response with confidence score"""
    confidence = qa_item['response']['confidence']
    confidence_class = (
        'confidence-low' if confidence < 50 else
        'confidence-medium' if confidence < 75 else
        'confidence-high'
    )
    
    font_size_class = {
        'Small': 'font-size-small',
        'Medium': 'font-size-medium',
        'Large': 'font-size-large',
        'Extra Large': 'font-size-xlarge'
    }[accessibility_settings['font_size']]
    
    st.markdown(f"""
        <div class="card {font_size_class}">
            <div class="content-text">
                <strong>Q: {qa_item['question']}</strong><br>
                A: {qa_item['response']['answer']}
            </div>
            <div class="{confidence_class}" style="margin-top: 0.5rem;">
                Confidence: {confidence}%
            </div>
            <div class="timestamp">
                {qa_item['response']['timestamp']}
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_term_definition(term, definition_data, accessibility_settings):
    """Render a term definition with confidence score"""
    confidence = definition_data['confidence']
    confidence_class = (
        'confidence-low' if confidence < 50 else
        'confidence-medium' if confidence < 75 else
        'confidence-high'
    )
    
    font_size_class = {
        'Small': 'font-size-small',
        'Medium': 'font-size-medium',
        'Large': 'font-size-large',
        'Extra Large': 'font-size-xlarge'
    }[accessibility_settings['font_size']]
    
    st.markdown(f"""
        <div class="card {font_size_class}">
            <strong class="key-term">{term}</strong>
            <p class="content-text">{definition_data['definition']}</p>
            <div class="{confidence_class}">
                Confidence: {confidence}%
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_streamlit_ui():
    """Create the main Streamlit user interface"""
    st.set_page_config(page_title="Audio Content Analysis", layout="wide")
    set_custom_style()
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown('# Audio Content Analysis')
        st.markdown('---')
        
        # API Key Input
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if openai_api_key:
            if not st.session_state.rag_system or st.session_state.api_key != openai_api_key:
                st.session_state.rag_system = EnhancedAudioRAG(openai_api_key)
                st.session_state.api_key = openai_api_key
        
        # MP3 File Upload
        st.markdown('### Upload Audio')
        uploaded_file = st.file_uploader("Choose an audio/video file", type=['mp3', 'mp4'])
        
        if uploaded_file:
            if st.session_state.rag_system:
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                
                # Only process if it's a new file or hasn't been processed
                if file_id != st.session_state.current_file_name:
                    with st.spinner("Processing audio file..."):
                        try:
                            if st.session_state.rag_system.process_mp3_file(uploaded_file):
                                st.session_state.current_file_name = file_id
                                st.session_state.file_content_processed = True
                                st.success("Audio file processed successfully!")
                                # Reset state for new file
                                st.session_state.summary_generated = False
                                st.session_state.current_summary = ""
                                st.session_state.qa_history = []
                                st.session_state.term_definitions = {}
                        except Exception as e:
                            st.error(f"Error processing file: {str(e)}")
                            st.session_state.file_content_processed = False
            else:
                st.warning("Please enter your API key first")
        
        # Accessibility Settings
        st.markdown('### Accessibility Settings')
        st.session_state.accessibility_settings['font_size'] = st.select_slider(
            'Text Size',
            options=['Small', 'Medium', 'Large', 'Extra Large']
        )
        
        st.session_state.accessibility_settings['contrast_mode'] = st.selectbox(
            'Display Mode',
            ['Standard', 'High Contrast', 'Dark Mode', 'Light Mode']
        )
        
        st.session_state.accessibility_settings['show_visual_aids'] = st.checkbox(
            'Show Visual Aids',
            value=True
        )
        
        st.session_state.accessibility_settings['highlight_key_terms'] = st.checkbox(
            'Highlight Key Terms',
            value=True
        )
    
    # Main content tabs
    tabs = st.tabs([
        "📝 Transcript",
        "❓ Q&A Assistant",
        "📚 Study Materials",
        "⚙️ Accessibility Tools"
    ])
    
    # Transcript tab
    with tabs[0]:
        st.markdown("### Audio Transcript")
        if not st.session_state.rag_system:
            st.info("Enter your API key to begin")
            return
            
        if not st.session_state.file_content_processed:
            st.info("Upload an MP3 file to see transcription")
            return
        if uploaded_file and uploaded_file.name.lower().endswith('.mp4'):
            st.markdown("#### visual aid")
            st.video(uploaded_file) 
         
        cols = st.columns([2, 1])
        with cols[0]:
            for segment in st.session_state.rag_system.transcribed_segments:
                render_transcript_segment(segment, st.session_state.accessibility_settings)
        
        with cols[1]:
            if st.session_state.accessibility_settings['show_visual_aids']:
                if st.session_state.rag_system.transcribed_segments:
                    last_segment = st.session_state.rag_system.transcribed_segments[-1]
                    if 'visual_aid' in last_segment:
                        render_visual_aid(
                            last_segment['visual_aid'],
                            st.session_state.accessibility_settings
                        )
    
    # Q&A Assistant tab
    with tabs[1]:
        st.markdown("### Ask Questions About the Content")
        if not st.session_state.rag_system:
            st.info("Enter your API key to begin")
            return
            
        if not st.session_state.file_content_processed:
            st.info("Upload an MP3 file to ask questions")
            return
            
        # Question input and submission
        question = st.text_input(
            "Ask a question about the content",
            placeholder="What are the main points discussed?",
            key="qa_input"
        )
        
        if st.button("Ask Question", type="primary", key="ask_button"):
            if question:
                with st.spinner("Generating answer..."):
                    response = st.session_state.rag_system.ask_question(question)
                    st.session_state.qa_history.append({
                        "question": question,
                        "response": response
                    })
        
        # Display QA history
        if st.session_state.qa_history:
            st.markdown("### Previous Questions")
            for qa in reversed(st.session_state.qa_history):
                render_qa_response(qa, st.session_state.accessibility_settings)
    
    # Study Materials tab
    # Study Materials tab
    with tabs[2]:
        st.markdown("### Study Materials & Resources")
        if not st.session_state.rag_system:
            st.info("Enter your API key to begin")
            return

        if not st.session_state.file_content_processed:
            st.info("Upload an audio/video file to generate study materials")
            return
            
        # Create subtabs
        timeline_tab, summary_tab, glossary_tab = st.tabs([
            "📅 Timeline", 
            "📝 Summary", 
            "📚 Glossary"
        ])
        
        # Timeline tab
        with timeline_tab:
            timeline_func = st.session_state.rag_system.process_transcript_timeline()
            if timeline_func:
                timeline_func()
        
        # Summary tab
        with summary_tab:
            st.markdown("#### Content Summary")
            if not st.session_state.summary_generated:
                if st.button("Generate Summary", type="primary"):
                    with st.spinner("Generating summary..."):
                        try:
                            summary_data = st.session_state.rag_system.generate_summary()
                            if summary_data:
                                st.session_state.current_summary = summary_data
                                st.session_state.summary_generated = True
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
            
            if st.session_state.summary_generated and st.session_state.current_summary:
                confidence = st.session_state.current_summary['confidence']
                confidence_class = (
                    'confidence-low' if confidence < 50 else
                    'confidence-medium' if confidence < 75 else
                    'confidence-high'
                )
                
                st.markdown(f"""
                    <div class="card">
                        <div class="content-text">
                            {st.session_state.current_summary['summary']}
                        </div>
                        <div class="{confidence_class}" style="margin-top: 0.5rem;">
                            Confidence: {confidence}%
                        </div>
                        <div class="timestamp">
                            Generated at: {st.session_state.current_summary['timestamp']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Glossary tab
        with glossary_tab:
            st.markdown("#### Key Terms Glossary")
            if st.session_state.rag_system.key_terms:
                undefined_terms = [term for term in st.session_state.rag_system.key_terms 
                                if term not in st.session_state.term_definitions]
                
                if undefined_terms and st.button("Generate Definitions"):
                    with st.spinner("Generating definitions..."):
                        for term in undefined_terms:
                            try:
                                definition = st.session_state.rag_system.generate_term_definition(term)
                                if definition:
                                    st.session_state.term_definitions[term] = definition
                            except Exception as e:
                                st.error(f"Error generating definition for {term}: {str(e)}")
                
                # Display glossary
                for term in sorted(st.session_state.rag_system.key_terms):
                    with st.expander(term):
                        if term in st.session_state.term_definitions:
                            definition_data = st.session_state.term_definitions[term]
                            render_term_definition(
                                term,
                                definition_data,
                                st.session_state.accessibility_settings
                            )
                        else:
                            st.info("Click 'Generate Definitions' to get the definition.")
            else:
                st.info("No key terms identified in the content yet")
        
        # Export options
        st.markdown("### Export Options")
        export_cols = st.columns(3)
        
        with export_cols[0]:
            if st.button("📝 Export Transcript"):
                transcript = st.session_state.rag_system.export_content("transcript")
                if transcript:
                    st.download_button(
                        "Download Transcript",
                        transcript,
                        "transcript.txt",
                        "text/plain",
                        key="download_transcript"
                    )
        
        with export_cols[1]:
            if st.button("📖 Export Glossary"):
                glossary = st.session_state.rag_system.export_content("glossary")
                if glossary:
                    st.download_button(
                        "Download Glossary",
                        glossary,
                        "glossary.txt",
                        "text/plain",
                        key="download_glossary"
                    )
        
        with export_cols[2]:
            if st.button("📚 Export Study Guide"):
                study_guide = st.session_state.rag_system.export_content("study_guide")
                if study_guide:
                    st.download_button(
                        "Download Study Guide",
                        study_guide,
                        "study_guide.txt",
                        "text/plain",
                        key="download_study_guide"
                    )
    with tabs[3]:
        st.markdown("### Accessibility Tools & Settings")
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("#### Current Settings")
            st.markdown(f"""
                - Font Size: {st.session_state.accessibility_settings['font_size']}
                - Display Mode: {st.session_state.accessibility_settings['contrast_mode']}
                - Visual Aids: {'Enabled' if st.session_state.accessibility_settings['show_visual_aids'] else 'Disabled'}
                - Key Terms Highlighting: {'Enabled' if st.session_state.accessibility_settings['highlight_key_terms'] else 'Disabled'}
            """)
            
            if st.button("Reset to Defaults"):
                st.session_state.accessibility_settings = {
                    'font_size': 'Medium',
                    'contrast_mode': 'Standard',
                    'show_visual_aids': True,
                    'highlight_key_terms': True
                }
                st.success("Settings reset to defaults!")
                st.rerun()
        
        with cols[1]:
            st.markdown("#### Keyboard Shortcuts")
            st.markdown("""
                - **Spacebar**: Play/Pause audio
                - **Up/Down**: Adjust font size
                - **Ctrl + F**: Search content
                - **Ctrl + H**: Toggle high contrast
            """)

if __name__ == "__main__":
    create_streamlit_ui()
