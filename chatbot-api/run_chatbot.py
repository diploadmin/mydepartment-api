import subprocess
import sys

def main():
    # Config is loaded from .env by default (app/core/config.py)
    try:
        # Import the specific variables needed, not the config object
        from app.core.config import HOST, PORT, RELOAD
    except ImportError as e:
        print(f"Error importing config variables: {e}", file=sys.stderr)
        print("Ensure that app/core/config.py defines HOST, PORT, RELOAD and is accessible.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during config import: {e}", file=sys.stderr)
        sys.exit(1)

    # Construct the uvicorn command using the imported variables
    command = [
        sys.executable,  # Use the same python interpreter running this script
        "-m", "uvicorn",
        "main:app",
        "--host", HOST,
        "--port", str(PORT), # Port needs to be a string for subprocess
    ]
    # Conditionally add --reload based on the RELOAD variable
    if RELOAD:
        command.append("--reload")

    print("Starting Chatbot API...")
    print(f"Running command: {' '.join(command)}")

    try:
        # Execute the command
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("Error: 'uvicorn' command not found.", file=sys.stderr)
        print("Please ensure uvicorn is installed in your environment (`pip install uvicorn`).", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Uvicorn process failed with error code {e.returncode}.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main() 