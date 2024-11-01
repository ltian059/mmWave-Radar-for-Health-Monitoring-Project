import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Return only necessary values
    model_path = config['DEFAULT'].get('model_path', 'default_model_path')
    model_data_input_folder = config['DEFAULT'].get('model_data_input_folder', 'default_input_folder')
    model_output_folder = config['DEFAULT'].get('model_output_folder', 'default_output_folder')

    return {
        'MODEL_PATH': model_path,
        'MODEL_DATA_INPUT_FOLDER': model_data_input_folder,
        'MODEL_OUTPUT_FOLDER': model_output_folder
    }


# Usage:
config_values = load_config()
MODEL_PATH = config_values['MODEL_PATH']
MODEL_DATA_INPUT_FOLDER = config_values['MODEL_DATA_INPUT_FOLDER']
MODEL_OUTPUT_FOLDER = config_values['MODEL_OUTPUT_FOLDER']





