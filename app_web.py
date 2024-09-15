import streamlit as st
import subprocess
import tempfile
import os
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load available workflows
def load_workflows():
    workflows = []
    config_dir = os.path.join(current_dir, 'config')
    for file in os.listdir(config_dir):
        if file.endswith('_config.yaml'):
            workflows.append(file.replace('_config.yaml', ''))
    return workflows

# Function to load config for a specific workflow
def load_config(workflow):
    config_path = os.path.join(current_dir, 'config', f'{workflow}_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Streamlit app
st.title('Text Processing Workflow')

# Workflow selection
workflows = load_workflows()
selected_workflow = st.selectbox('Select Workflow', workflows)

# Load config for selected workflow
config = load_config(selected_workflow)

# Text input
input_text = st.text_area('Enter text to process')

# Process button
if st.button('Process'):
    if input_text:
        # Save input text to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
            temp_file.write(input_text)
            temp_file_path = temp_file.name

        # Prepare command with full path to app.py
        app_path = os.path.join(current_dir, 'app.py')
        cmd = ['poetry', '--directory', current_dir, 'run', 'python', app_path, temp_file_path, '--workflow', selected_workflow]

        # Run the command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Display output and provide download link
            output_file = os.path.join(os.path.dirname(temp_file_path), f'{selected_workflow}-output.md')
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_content = f.read()
                st.text_area('Output', output_content, height=300)
                st.download_button('Download Output', output_content, file_name=f'{selected_workflow}-output.md')
            else:
                st.warning('Output file not found. Displaying standard output and error for debugging:')
                st.text_area('Standard Output', result.stdout, height=300)
                if result.stderr:
                    st.text_area('Standard Error', result.stderr, height=300)

        except subprocess.CalledProcessError as e:
            st.error(f'Error occurred. Displaying standard output and error for debugging:')
            st.text_area('Standard Output', e.stdout, height=300)
            st.text_area('Standard Error', e.stderr, height=300)

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    else:
        st.warning('Please enter some text to process.')