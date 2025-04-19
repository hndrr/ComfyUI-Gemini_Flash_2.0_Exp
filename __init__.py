import os
import json
import importlib.util

current_path = os.path.dirname(os.path.realpath(__file__))

def load_python_file(filepath):
    try:
        spec = importlib.util.spec_from_file_location("module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

# Try to find the node files (case-insensitive on Linux)
nodes_dir = os.path.join(current_path, "nodes")
gemini_filename = "Gemini_Flash_Node.py"
audio_recorder_filename = "nodes_audio_recorder.py"

gemini_path = None
audio_recorder_path = None

# First try the exact case
exact_gemini_path = os.path.join(nodes_dir, gemini_filename)
exact_audio_path = os.path.join(nodes_dir, audio_recorder_filename)

# Then try lowercase if needed
lower_gemini_path = os.path.join(nodes_dir, gemini_filename.lower())
lower_audio_path = os.path.join(nodes_dir, audio_recorder_filename.lower())

# Check which paths exist and use them
if os.path.exists(exact_gemini_path):
    gemini_path = exact_gemini_path
    print(f"Found Gemini node at: {gemini_path}")
elif os.path.exists(lower_gemini_path):
    gemini_path = lower_gemini_path
    print(f"Found Gemini node at lowercase path: {gemini_path}")

if os.path.exists(exact_audio_path):
    audio_recorder_path = exact_audio_path
    print(f"Found Audio recorder at: {audio_recorder_path}")
elif os.path.exists(lower_audio_path):
    audio_recorder_path = lower_audio_path
    print(f"Found Audio recorder at lowercase path: {audio_recorder_path}")

# Try to create config.json if it doesn't exist
try:
    config_path = os.path.join(current_path, "nodes", "config.json")
    if not os.path.exists(config_path):
        print(f"Creating config.json at {config_path}")
        with open(config_path, 'w') as f:
            json.dump({"GEMINI_API_KEY": ""}, f, indent=4)
except Exception as e:
    print(f"Could not create config.json: {str(e)}")

# Load modules if found
gemini_module = None
audio_recorder_module = None

if gemini_path:
    gemini_module = load_python_file(gemini_path)
else:
    print(f"Could not find Gemini node file (tried {gemini_filename} and {gemini_filename.lower()})")

if audio_recorder_path:
    audio_recorder_module = load_python_file(audio_recorder_path)
else:
    print(f"Could not find Audio recorder node file (tried {audio_recorder_filename} and {audio_recorder_filename.lower()})")

# Initialize mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Merge mappings
if gemini_module and hasattr(gemini_module, 'NODE_CLASS_MAPPINGS'):
    NODE_CLASS_MAPPINGS.update(gemini_module.NODE_CLASS_MAPPINGS)
    print("Added Gemini node to mappings")

if gemini_module and hasattr(gemini_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
    NODE_DISPLAY_NAME_MAPPINGS.update(gemini_module.NODE_DISPLAY_NAME_MAPPINGS)

if audio_recorder_module and hasattr(audio_recorder_module, 'NODE_CLASS_MAPPINGS'):
    NODE_CLASS_MAPPINGS.update(audio_recorder_module.NODE_CLASS_MAPPINGS)
    print("Added Audio recorder node to mappings")

if audio_recorder_module and hasattr(audio_recorder_module, 'NODE_DISPLAY_NAME_MAPPINGS'):
    NODE_DISPLAY_NAME_MAPPINGS.update(audio_recorder_module.NODE_DISPLAY_NAME_MAPPINGS)

# Define web directory
WEB_DIRECTORY = os.path.join(current_path, "web")