import streamlit as st
from styles.cssTemplate import css
from controllers import processPdf
from controllers import ragApproach1, ragApproach2
from dotenv import load_dotenv
from gemini_utils import upload_to_gemini, wait_for_files_active, query_gemini, code_convert_gemini
import os

# Load environment variables from a .env file
load_dotenv()



def checkpdf(pdfname, filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read() == pdfname
    return False


def savepdf(pdfname, filepath):
    with open(filepath, "w") as f:
        f.write(pdfname)

def main():
    st.set_page_config(layout="wide")
    
    # st.write(css, unsafe_allow_html=True)

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




    # Box 1: Query and Compare
    st.subheader("Step 1: Query and Compare Results")

    col1, col2 = st.columns(2)

    with col1:
        pdf_docs_1 = st.file_uploader(
            "Upload first set of PDF documents", type=["pdf"], accept_multiple_files=True, key="pdf_docs_1"
        )
        
    with col2:
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

                if checkpdf( pdf_docs_1[0].name, "processed_pdf/1.txt"):
                    st.session_state.vector_initialized_1  = True
                else:
                    st.session_state.vector_initialized_1 = st.session_state.vector_db_1.create_vector_store()
                    savepdf(pdf_docs_1[0].name, "processed_pdf/1.txt")

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
                if checkpdf( pdf_docs_2[0].name, "processed_pdf/2.txt"):
                    st.session_state.vector_initialized_2  = True
                else:
                    st.session_state.vector_initialized_2 = st.session_state.vector_db_2.append_to_vector_store()
                    savepdf(pdf_docs_2[0].name, "processed_pdf/2.txt")

                

                for pdf in pdf_docs_2:
                    st.write(f"Uploaded PDF: {pdf.name} (Set 2)")




    questions = [
        "Make table from register address, power on and power off and other features",
        "Communication (I2C, SPI, MIPI, or serial), register values",
        "Electrical parameters, information related voltages, power supplies and current"
    ]
    code_questions = [
        "Extract all the definitions, macros (whole) and function (whole) related to setting up the device, initializing modes, or enabling features into a code block, output: strictly code only",
        "Extract all the definitions, macros (whole) and function (whole) related to Communication (I2C, SPI, MIPI, or serial) into a code block , output: strictly code only",
        "Extract all the definitions, macros (whole) and function (whole) related to Communication (I2C, SPI, MIPI, or serial) into a code block , output: strictly code only",

    ]

    answers = []

    # Query Vector DBs
    if st.session_state.vector_initialized_1:
        answer_1 = st.session_state.vector_db_1.get_answer_for_text_query(question=questions)
        answers.append(answer_1)

    if st.session_state.vector_initialized_2:
        answer_2 = st.session_state.vector_db_2.get_answer_for_text_query2(question=questions)
        answers.append(answer_2)

    # Display results in an expander for a cleaner UI
    
    for idx, question in enumerate(questions):
        with st.expander("Compare Results by Category"):
            st.write("**First Document Set Answer**")
            st.markdown(answers[0][idx] if len(answers) > 0 else "No data available.")
            st.write("**Second Document Set Answer**")
            st.markdown(answers[1][idx] if len(answers) > 1 else "No data available.")


    # Box 2: Upload and Process File
    st.subheader("Step 2: Upload and Process File")
    uploaded_file = st.file_uploader("Upload a .c file", type=["c"])

    code_snips = []

    if uploaded_file:
        # Convert to .txt file
        txt_file_path = f"{uploaded_file.name}.txt"
        with open(txt_file_path, "w") as f:
            f.write(uploaded_file.read().decode("utf-8"))

        st.success(f"File converted to: {txt_file_path}")

        # Upload to Gemini and process file
        with st.spinner("Uploading file and processing..."):
            file = upload_to_gemini(txt_file_path, mime_type="text/plain")
            wait_for_files_active([file])

        st.success("File uploaded successfully!")

        with st.expander("Extract codes"):
        # Extract category-wise data
            for code_question in code_questions:
                response = query_gemini([file], f"""{code_question}""")
                st.markdown(f"####Category: {code_question}")
                st.markdown(response)
                code_snips.append(response)

            

    if st.button("Generate Code conversion"):
        with st.spinner("Processing"):
            for idx, code_snip in enumerate(code_snips):
                response = code_convert_gemini(f"""
                Code snippet for doc 1 -> {pdf_docs_2[0].name} :
                {code_snip}

                Information for doc 1-> {pdf_docs_2[0].name} :
                {answers[0][idx]}

                Information for doc 2-> {pdf_docs_1[0].name} :
                {answers[1][idx]}

                Output: Strictly code in code block (After conversion)


                """)
                st.markdown(f"####Category: {code_question}")
                st.markdown(response)

    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}", icon="🚨")
        st.stop()
