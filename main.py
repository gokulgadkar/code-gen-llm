import streamlit as st
from styles.cssTemplate import css
from controllers import processPdf
from controllers import ragApproach1, ragApproach2
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def main():
    st.header("Sensor data sheet summerizer")
    st.title("Pdf analysizer :books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize conversation history in session state if not already done
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    
    

    # Initialize the vector database in session state if not already done
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    # Initialize the vector store status in session state if not already done
    if "vector_initialized" not in st.session_state:
        st.session_state.vector_initialized = False

    # Input for user to type their question
    question = st.chat_input("Ask the question here ")

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your documents and click 'Process'", type=["pdf"], accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing"):
                PDFprocessor = processPdf.PDFprocessor(
                    pdf_docs=pdf_docs, pdf_names=[pdf_doc.name for pdf_doc in pdf_docs])
                chunks = PDFprocessor.parse_pdf()
                csv_chunks = PDFprocessor.get_tables()

                # Display the name of the uploaded PDF
                for pdf in pdf_docs:
                    st.write(f"Uploaded PDF: {pdf.name}")

                    # Initialize the vector database with the processed PDF and CSV chunks
                    st.session_state.vector_db = ragApproach1.ragApproach1(
                        pdf_docs=pdf_docs, chunks=chunks, pdf_names=[
                            pdf_doc.name for pdf_doc in pdf_docs],
                        csv_chunks=csv_chunks
                    )
                    # Create the vector store
                    st.session_state.vector_initialized = st.session_state.vector_db.create_vector_store()

    if question:
        if st.session_state.vector_initialized:
            # Update the conversation with the user's question
            st.session_state.conversation.append({
                "role": "user",
                "parts": [question]
            })
            
            # Display the user's question in the chat
            with st.chat_message("user"):
                st.write(question)

            # Get the answer from the model
            answer = st.session_state.vector_db.get_answer_for_text_query(
                question=question, chat_history=st.session_state.conversation
            )
            
            # Update the conversation with the bot's response
            st.session_state.conversation.append({
                "role": "model",
                "parts": [answer]
            })
            
            # Display the model's response in the chat
            with st.chat_message("model"):
                st.write(answer)
            
            question = ""
        else:
            st.error("Please upload a PDF document and enter a question.", icon="🚨")
            st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        st.stop()
