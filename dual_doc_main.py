import streamlit as st
from styles.cssTemplate import css
from controllers import processPdf
from controllers import ragApproach1, ragApproach2
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def main():
    st.header("Sensor data sheet summarizer")
    st.title("PDF Analyzer :books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize conversation history in session state if not already done
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Initialize the vector databases in session state if not already done
    if "vector_db_1" not in st.session_state:
        st.session_state.vector_db_1 = None

    if "vector_db_2" not in st.session_state:
        st.session_state.vector_db_2 = None

    # Initialize the vector store status in session state if not already done
    if "vector_initialized_1" not in st.session_state:
        st.session_state.vector_initialized_1 = False

    if "vector_initialized_2" not in st.session_state:
        st.session_state.vector_initialized_2 = False

    # Input for user to type their question
    question = st.chat_input("Ask the question here ")

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.subheader("Your documents")
        st.write("Upload first set of documents:")

        # Upload first set of PDFs
        pdf_docs_1 = st.file_uploader(
            "Upload first set of PDF documents", type=["pdf"], accept_multiple_files=True, key="pdf_docs_1"
        )

        st.write("Upload second set of documents:")

        # Upload second set of PDFs
        pdf_docs_2 = st.file_uploader(
            "Upload second set of PDF documents", type=["pdf"], accept_multiple_files=True, key="pdf_docs_2"
        )

        # Button to process both PDF sets
        if st.button("Process"):
            with st.spinner("Processing"):
                
                # Process first set of PDFs
                if pdf_docs_1:
                    st.write("Processing first set of PDFs...")
                    PDFprocessor_1 = processPdf.PDFprocessor(
                        pdf_docs=pdf_docs_1, pdf_names=[pdf_doc.name for pdf_doc in pdf_docs_1]
                    )
                    chunks_1 = PDFprocessor_1.parse_pdf()
                    csv_chunks_1 = PDFprocessor_1.get_tables()

                    st.session_state.vector_db_1 = ragApproach1.ragApproach1(
                        pdf_docs=pdf_docs_1, chunks=chunks_1, pdf_names=[pdf_doc.name for pdf_doc in pdf_docs_1], 
                        csv_chunks=csv_chunks_1
                    )

                    st.session_state.vector_initialized_1 = st.session_state.vector_db_1.create_vector_store()

                    for pdf in pdf_docs_1:
                        st.write(f"Uploaded PDF: {pdf.name} (Set 1)")

                # Process second set of PDFs
                if pdf_docs_2:
                    st.write("Processing second set of PDFs...")
                    PDFprocessor_2 = processPdf.PDFprocessor(
                        pdf_docs=pdf_docs_2, pdf_names=[pdf_doc.name for pdf_doc in pdf_docs_2]
                    )
                    
                    chunks_2 = PDFprocessor_2.parse_pdf()
                   
                   
                    csv_chunks_2 = PDFprocessor_2.get_tables()
                    
                    st.session_state.vector_db_2 = ragApproach1.ragApproach1(
                        pdf_docs=pdf_docs_2, chunks=chunks_2, pdf_names=[pdf_doc.name for pdf_doc in pdf_docs_2], 
                        csv_chunks=csv_chunks_2
                    )

                    st.session_state.vector_initialized_2 = st.session_state.vector_db_2.append_to_vector_store()

                    for pdf in pdf_docs_2:
                        st.write(f"Uploaded PDF: {pdf.name} (Set 2)")

    # Handle question input
    if question:
        if st.session_state.vector_initialized_1 or st.session_state.vector_initialized_2:
            # Update the conversation with the user's question
            st.session_state.conversation.append({
                "role": "user",
                "parts": [question]
            })

            # Display the user's question in the chat
            with st.chat_message("user"):
                st.write(question)

            # Get answers from both vector databases if initialized
            answers = []

            if st.session_state.vector_initialized_1:
                answer_1 = st.session_state.vector_db_1.get_answer_for_text_query(
                    question=question, chat_history=st.session_state.conversation
                )
                answers.append(f"**Answer from first set of documents:**\n{answer_1}")

            if st.session_state.vector_initialized_2:
                answer_2 = st.session_state.vector_db_2.get_answer_for_text_query(
                    question=question, chat_history=st.session_state.conversation
                )
                answers.append(f"**Answer from second set of documents:**\n{answer_2}")

            # Update the conversation with the bot's response
            st.session_state.conversation.append({
                "role": "model",
                "parts": answers
            })

            # Display the model's responses in the chat
            with st.chat_message("model"):
                for answer in answers:
                    st.write(answer)

            question = ""
        else:
            st.error("Please upload PDF documents and enter a question.", icon="🚨")
            st.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        st.stop()
