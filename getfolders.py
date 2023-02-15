import os


# get current directory
def get_current_dir():
    return os.path.dirname(os.path.abspath(__file__))


# get training directory
def get_training_dir():
    return os.path.normpath(os.path.join(get_current_dir() + '/traincropped'))


# get testing directory
def get_testing_dir():
    return os.path.normpath(os.path.join(get_current_dir() + '/testdata'))


# get output directory
def get_output_dir():
    return os.path.normpath(os.path.join(get_current_dir() + '/output'))


# get negative images directory
def get_negative_dir():
    return os.path.normpath(os.path.join(get_current_dir() + '/negative'))


# util method for folder creation
def create_folder_for_file(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_path = os.path.join(directory, filename)
            folder_name = os.path.splitext(filename)[0]
            os.makedirs(os.path.join(directory, folder_name), exist_ok=True)
