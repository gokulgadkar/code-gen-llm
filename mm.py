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
