import os
import logging
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_embeddings(text: str, retries: int = MAX_RETRIES) -> List[float]:
    """Get embeddings from OpenAI API with retry logic"""
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=1536
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to get embeddings after {retries} attempts: {str(e)}")
                raise
            logger.warning(f"Embedding attempt {attempt + 1} failed, retrying...") 