import streamlit.web.cli as stcli
import sys
import os

def main():
    # Add the src directory to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    sys.argv = ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()