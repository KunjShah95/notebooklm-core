import logging 
import os 
import tempfile 
from pathlib import Path
from typing import List , Optional
import assemblyai as aai
import yt_dlp

from src.document_processing.doc_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YoutubeTranscriber:
    def __init__(self, api_key: str):
        self.assemblyai_api_key = assemblyai_api_key
        self.temp_dir = Path(tempfile.gettempdir()) / "youtube_transcriber"
        self.temp_dir.mkdir(exist_ok=True)
        
        aai.settings.api_key = assemblyai_api_key
        
        logger.info("YouTubeTranscriber initialized")