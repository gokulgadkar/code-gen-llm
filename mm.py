import os
import shutil


def clear_directory(directory):
    print(directory)
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f'Successfully deleted file: {file_path}')
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f'Successfully deleted directory: {file_path}')
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


directories = [
    'boschHackathon/faiss_db/',
    'boschHackathon/context_docs/',
    'boschHackathon/assets/Images/'
    # Add more directories as needed
]

for directory in directories:
    if os.path.exists(directory):
        clear_directory(directory)
        print(f'Cleared directory: {directory}')
    else:
        print(f'Directory does not exist: {directory}')



    # # Sidebar for uploading and processing documents
    # with st.sidebar:
    #     st.subheader("Your documents")






    # # Handle question input
    # if question:
    #     if st.session_state.vector_initialized_1 and st.session_state.vector_initialized_2:

    #         with st.chat_message("user"):
    #             st.write(question)

    #         questions = [
    #             "Make a table for registers and addresses"
    #             "Extract technical details related to Communication",
    #             "Extract technical details related to Voltage"

    #         ]

    #         # Get answers from both vector databases if initialized
    #         answers = []

    #         if st.session_state.vector_initialized_1:
    #             answer_1 = st.session_state.vector_db_1.get_answer_for_text_query(
    #                 question=questions
    #             )
    #             answers.append(f"**Answer from first set of documents:**\n{answer_1}")

    #         if st.session_state.vector_initialized_2:
    #             answer_2 = st.session_state.vector_db_2.get_answer_for_text_query2(
    #                 question=questions
    #             )
    #             answers.append(f"**Answer from second set of documents:**\n{answer_2}")

    #         with st.chat_message("model"):
    #             for answer in answers:
    #                 st.write(answer)

    #         question = ""
    #     else:
    #         st.error("Please upload PDF documents and enter a question.", icon="🚨")
    #         st.stop()
    
