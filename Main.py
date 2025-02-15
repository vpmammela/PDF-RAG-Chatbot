import logging
from src.ui import create_ui

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()