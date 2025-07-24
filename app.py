# app.py
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from agent_factory import create_agent, SYSTEM_PROMPT
import PyPDF2
import os
import openai

st.title("General PDF Q&A Chatbot")
st.markdown("""
Welcome! Upload any PDF documents and ask questions about their content.
""")

# PDF type selection
pdf_type = st.radio(
    "Select PDF type:",
    ("Text-based PDF", "Scanned PDF (images, not selectable text)"),
    index=0
)

# PDF upload
uploaded_files = st.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)
pdf_text = ""
if uploaded_files:
    st.session_state["pdfs"] = uploaded_files
    st.success(f"{len(uploaded_files)} PDF(s) uploaded.")

    if pdf_type == "Text-based PDF":
        # Extract text from all uploaded PDFs
        for pdf_file in uploaded_files:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text() or ""
        st.session_state["pdf_text"] = pdf_text

        # Show PDF content in an expander
        with st.expander("Show extracted PDF content"):
            st.text_area("PDF Content", pdf_text, height=300)

        # Show the system prompt for debugging
        debug_prompt = SYSTEM_PROMPT.format(context=pdf_text)
        with st.expander("Show system prompt (debug)"):
            st.text_area("System Prompt", debug_prompt, height=200)

        # Create and store agent with current context
        if os.environ.get("OPENAI_API_KEY"):
            st.session_state["agent"] = create_agent(context=pdf_text)
        else:
            # Dummy agent for testing without API key
            class DummyAgent:
                def invoke(self, *args, **kwargs):
                    return {"messages": [AIMessage(content="(Dummy agent: No API key set)\n\n" + debug_prompt)]}
            st.session_state["agent"] = DummyAgent()
    else:
        # Scanned PDF: upload to OpenAI Files API
        st.info("Scanned PDF selected. The PDF(s) will be sent directly to the LLM for understanding.")
        with st.expander("Show uploaded PDF files"):
            st.write([f.name for f in uploaded_files])

        # Upload PDFs to OpenAI and store file_ids
        file_ids = []
        if os.environ.get("OPENAI_API_KEY"):
            openai.api_key = os.environ["OPENAI_API_KEY"]
            for pdf_file in uploaded_files:
                # Save to a temp file because openai.Files.create needs a file path
                with open(pdf_file.name, "wb") as temp_out:
                    temp_out.write(pdf_file.read())
                file_obj = openai.files.create(
                    file=open(pdf_file.name, "rb"),
                    purpose="user_data"
                )
                file_ids.append(file_obj.id)
            st.session_state["file_ids"] = file_ids
        else:
            st.session_state["file_ids"] = []

        # Show the system prompt for debugging (context is file list)
        debug_prompt = SYSTEM_PROMPT.format(context="PDF(s) attached: " + ", ".join([f.name for f in uploaded_files]))
        with st.expander("Show system prompt (debug)"):
            st.text_area("System Prompt", debug_prompt, height=200)

        # Create and store agent with file_ids as context
        if os.environ.get("OPENAI_API_KEY"):
            class ScannedPDFAgent:
                def invoke(self, *args, **kwargs):
                    messages = args[0].get("messages", [])
                    user_question = ""
                    for m in messages[::-1]:
                        if isinstance(m, HumanMessage):
                            user_question = m.content
                            break
                    # Use the first file_id for demo; extend as needed
                    file_ids = st.session_state.get("file_ids", [])
                    if not file_ids:
                        return {"messages": [AIMessage(content="No file uploaded to OpenAI.")]}

                    resp = openai.responses.create(
                        model="gpt-4.1-nano",
                        input=[
                            {"type": "input_file", "file_id": file_ids[0]},
                            {"type": "input_text", "text": user_question}
                        ]
                    )
                    return {"messages": [AIMessage(content=resp.choices[0].message.content)]}
            st.session_state["agent"] = ScannedPDFAgent()
        else:
            class DummyAgent:
                def invoke(self, *args, **kwargs):
                    return {"messages": [AIMessage(content="(Dummy agent: No API key set)\n\n" + debug_prompt)]}
            st.session_state["agent"] = DummyAgent()
elif "agent" not in st.session_state:
    # No PDFs uploaded yet, create agent with empty context
    if os.environ.get("OPENAI_API_KEY"):
        st.session_state["agent"] = create_agent(context="")
    else:
        class DummyAgent:
            def invoke(self, *args, **kwargs):
                prompt = SYSTEM_PROMPT.format(context="")
                return {"messages": [AIMessage(content="(Dummy agent: No API key set)\n\n" + prompt)]}
        st.session_state["agent"] = DummyAgent()

agent = st.session_state["agent"]

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Render past messages
for entry in st.session_state.history:
    role, text = entry
    if role == "user":
        st.chat_message("user").write(text)
    else:
        st.chat_message("assistant").write(text)

# Input box
if prompt := st.chat_input("Your message"):
    # Record user turn
    st.session_state.history.append(("user", prompt))
    with st.spinner("Thinking…"):
        # Build full chat history for agent
        messages = []
        for role, text in st.session_state.history:
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "bot":
                messages.append(AIMessage(content=text))
        messages.append(HumanMessage(content=prompt))

        resp = agent.invoke(
                {"messages": messages},
                config={"configurable": {"thread_id": "user_thread"}}
            )
        # Extract the final AIMessage text
        ai_msg = None
        if isinstance(resp, dict) and "messages" in resp:
            # Filter out AIMessage objects
            ai_msgs = [m for m in resp["messages"] if isinstance(m, AIMessage)]
            if ai_msgs:
                ai_msg = ai_msgs[-1].content
            else:
                ai_msg = ""
        else:
            ai_msg = str(resp)
    # Record bot turn
    st.session_state.history.append(("bot", ai_msg))
    # Rerun to display
    st.rerun()