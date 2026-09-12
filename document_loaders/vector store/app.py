import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="DeepLearning RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for advanced styling - IMPROVED CONTRAST
st.markdown("""
    <style>
        /* Color scheme - can be customized */
        :root {
            --primary-color: #2196F3;
            --secondary-color: #9c27b0;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --danger-color: #f44336;
        }
        
        /* Main background */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Chat container */
        .chat-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* User message - IMPROVED TEXT COLOR */
        .user-message {
            background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
            border-left: 4px solid #0d47a1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            animation: slideIn 0.3s ease-in-out;
            color: #ffffff;
        }
        
        .user-message strong {
            color: #ffffff;
            font-weight: 700;
        }
        
        /* Assistant message - IMPROVED TEXT COLOR */
        .assistant-message {
            background: linear-gradient(135deg, #7b1fa2 0%, #6a1b9a 100%);
            border-left: 4px solid #4a148c;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            animation: slideIn 0.3s ease-in-out;
            color: #ffffff;
        }
        
        .assistant-message strong {
            color: #ffffff;
            font-weight: 700;
        }
        
        /* Source documents - IMPROVED TEXT COLOR */
        .source-document {
            background: linear-gradient(135deg, #ffd54f 0%, #ffca28 100%);
            border-left: 4px solid #f9a825;
            padding: 12px;
            border-radius: 6px;
            margin: 8px 0;
            font-size: 0.85rem;
            max-height: 200px;
            overflow-y: auto;
            color: #333333;
        }
        
        .source-title {
            font-weight: bold;
            color: #d68910;
            margin-bottom: 8px;
            font-size: 0.9rem;
        }
        
        /* Metric cards */
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-top: 3px solid #2196F3;
        }
        
        /* Animations */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Info boxes */
        .info-box {
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            color: #1565c0;
        }
        
        /* Success boxes */
        .success-box {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 12px;
            border-radius: 4px;
            margin: 10px 0;
            color: #2e7d32;
        }
        
        /* Divider */
        .divider {
            border-top: 2px solid #ddd;
            margin: 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_metadata" not in st.session_state:
    st.session_state.conversation_metadata = {
        "start_time": datetime.now().isoformat(),
        "message_count": 0,
        "total_tokens_used": 0
    }

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Advanced Settings")
    
    # Tabs in sidebar
    sidebar_tabs = st.tabs(["Settings", "Statistics", "Export", "About"])
    
    # Settings Tab
    with sidebar_tabs[0]:
        st.subheader("🧠 Model Configuration")
        model_name = st.selectbox(
            "Select LLM Model",
            ["ministral-3b-2512", "mistral-7b-latest"],
            help="Choose the Mistral AI model to use"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        st.markdown("---")
        st.subheader("🔍 Retrieval Configuration")
        
        k = st.slider(
            "Documents to retrieve (k)",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of most relevant documents"
        )
        
        fetch_k = st.slider(
            "Fetch K (for MMR)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="Documents to fetch for MMR algorithm"
        )
        
        lambda_mult = st.slider(
            "Lambda Multiplier",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0=diverse, 1=relevant"
        )
        
        st.markdown("---")
        st.subheader("🎨 Display Options")
        
        show_sources = st.checkbox("Show source documents by default", value=True)
        show_metadata = st.checkbox("Show response metadata", value=False)
    
    # Statistics Tab
    with sidebar_tabs[1]:
        st.subheader("📊 Conversation Statistics")
        
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👤 User Messages", user_messages)
        with col2:
            st.metric("🤖 AI Responses", assistant_messages)
        
        if user_messages > 0:
            st.info(f"Avg sources per response: {(sum([len(m.get('sources', [])) for m in st.session_state.messages if m['role'] == 'assistant']) / max(assistant_messages, 1)):.1f}")
        
        start_time = datetime.fromisoformat(st.session_state.conversation_metadata["start_time"])
        duration = datetime.now() - start_time
        st.metric("⏱️ Session Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")
    
    # Export Tab
    with sidebar_tabs[2]:
        st.subheader("📥 Export Conversation")
        
        if len(st.session_state.messages) > 0:
            # JSON Export
            json_data = json.dumps(st.session_state.messages, indent=2, default=str)
            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Markdown Export
            markdown_content = "# RAG Conversation Export\n\n"
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    markdown_content += f"## User Message {i//2 + 1}\n{msg['content']}\n\n"
                else:
                    markdown_content += f"## AI Response {i//2}\n{msg['content']}\n\n"
                    if "sources" in msg:
                        markdown_content += "### Sources\n"
                        for j, source in enumerate(msg["sources"], 1):
                            markdown_content += f"- **Source {j}**: {source[:200]}...\n"
                        markdown_content += "\n"
            
            st.download_button(
                label="📝 Download as Markdown",
                data=markdown_content,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("No conversation to export yet.")
        
        st.markdown("---")
        if st.button("🗑️ Clear All History"):
            st.session_state.messages = []
            st.session_state.conversation_metadata["start_time"] = datetime.now().isoformat()
            st.success("Chat history cleared!")
            st.rerun()
    
    # About Tab
    with sidebar_tabs[3]:
        st.subheader("ℹ️ About")
        st.markdown("""
        ### Technology Stack
        - **LangChain**: Orchestration framework
        - **Mistral AI**: LLM & Embeddings
        - **Chroma**: Vector database
        - **Streamlit**: UI framework
        
        ### Features
        - 💬 Interactive chat interface
        - 🔍 Smart document retrieval (MMR)
        - 📚 Source document viewing
        - 📊 Conversation statistics
        - 📥 Export conversations
        - ⚙️ Advanced configuration
        
        ### Version
        v2.0 Advanced Edition
        """)

# Main Content Area
st.title("🤖 DeepLearning RAG Assistant")
st.markdown("*Advanced Edition - Powered by Mistral AI & LangChain*")

st.markdown("---")

# Initialize RAG Components
@st.cache_resource
def load_rag_components():
    """Load and cache RAG components"""
    with st.spinner("Loading embeddings and vector store..."):
        try:
            embedding_model = MistralAIEmbeddings(
                model="mistral-embed"
            )
            
            vectorstore = Chroma(
                persist_directory="chroma_db",
                embedding_function=embedding_model
            )
            
            llm = ChatMistralAI(
                model=model_name,
                temperature=temperature
            )
            
            return vectorstore, llm
        except Exception as e:
            st.error(f"Error loading components: {e}")
            return None, None

# Load components
vectorstore, llm = load_rag_components()

if vectorstore is None or llm is None:
    st.stop()

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant specializing in Deep Learning.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the provided documents."

Provide clear, concise, and well-structured answers.
Format your response with proper paragraphs and bullet points where appropriate.
Be educational and explain concepts clearly."""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

# Display chat history
st.subheader("💬 Conversation")

if len(st.session_state.messages) == 0:
    st.info("👋 Welcome! Ask me any questions about Deep Learning. Use the settings on the left to customize the retrieval and model behavior.")
else:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong style="color: #ffffff;">👤 You:</strong><br>
                <span style="color: #ffffff;">{message["content"]}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <strong style="color: #ffffff;">🤖 AI Assistant:</strong><br>
                <span style="color: #ffffff;">{message["content"]}</span>
            </div>
            """, unsafe_allow_html=True)
            
            if "sources" in message and show_sources:
                with st.expander(f"📚 View {len(message['sources'])} Source Document(s)"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-document">
                            <div class="source-title">📄 Source {i}</div>
                            <span style="color: #333333;">{doc[:500]}...</span>
                        </div>
                        """, unsafe_allow_html=True)

# Input section
st.markdown("---")
st.subheader("❓ Ask Your Question")

col1, col2 = st.columns([0.85, 0.15])

with col1:
    user_input = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are neural networks? Explain backpropagation...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("🚀 Send", use_container_width=True)

# Process user input
if send_button and user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    st.session_state.conversation_metadata["message_count"] += 1
    
    # Process with loading animation
    with st.spinner("🔍 Retrieving documents & generating response..."):
        try:
            # Create retriever with current settings
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                }
            )
            
            # Retrieve documents
            docs = retriever.invoke(user_input)
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate response
            final_prompt = prompt.invoke({
                "context": context,
                "question": user_input
            })
            
            response = llm.invoke(final_prompt)
            response_text = response.content
            
            # Add assistant message
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "sources": [doc.page_content for doc in docs],
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "model": model_name,
                    "documents_retrieved": len(docs),
                    "temperature": temperature
                }
            }
            
            st.session_state.messages.append(assistant_message)
            
            st.success("Response generated successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error: {e}")

# Footer with metrics
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📝 Total Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
with col2:
    st.metric("🔍 Retrieved Docs", k)
with col3:
    st.metric("🧠 Model", model_name.split("-")[0].capitalize())
with col4:
    st.metric("🌡️ Temperature", temperature)

# Footer
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8rem; margin-top: 30px;'>
    <p>🚀 DeepLearning RAG Assistant | Advanced Edition v2.0</p>
    <p>Built with ❤️ using Streamlit, LangChain & Mistral AI</p>
    </div>
    """,
    unsafe_allow_html=True
)