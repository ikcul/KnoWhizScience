import os
import logging
import re
import json
import requests
import http.client
import wikipediaapi
from arxiv import Search
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, VideoUnavailable, TranscriptsDisabled

logger = logging.getLogger("kzpipeline.science.links_handler")

class Page:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}
        self.page_text_content = content

class ExternalLinksHandler:
    def __init__(self, url, file_path=''):
        self.user_agent = 'KnowWhiz'
        self.not_found_message = "Error: link content not found."
        self.file_path = file_path
        self.url = url
        self.chunk_size = 4000

    def _split_doc_into_chunks(self, document):
        """
        Splits a document into chunks of specified size with no overlap.

        Parameters:
        document (str): The text of the document to be split.
        chunk_size (int): The size of each chunk.

        Returns:
        list: A list of text chunks.
        """
        chunk_size = self.chunk_size
        chunks = []
        for i in range(0, len(document), chunk_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
        logger.info(f'Now you have {len(chunks)} chunks from the document.')
        return chunks

    def load(self):
        """
        Determines the type of URL and calls the appropriate function to load the content.

        Parameters:
        url (str): The URL to load the content from.

        Returns:
        str: The loaded content, or an error message if the link type is not recognized or content fetching fails.
        """
        loaded_content = []
        if 'wikipedia.org' in self.url:
            docs = self.get_wikipedia_content(self.url)
            if(docs == self.not_found_message):
                return self.not_found_message
        elif 'youtube.com' in self.url:
            docs = self.get_youtube_content(self.url)
            if(docs == self.not_found_message):
                logger.info(f"Error not found message: {self.not_found_message}")
                return self.not_found_message
        elif 'arxiv.org' in self.url:
            docs = self.get_arxiv_content(self.url)
            if(docs == self.not_found_message):
                return self.not_found_message
        else:
            logger.info("Error: Unsupported URL type.")
            return self.not_found_message

        docs = self._split_doc_into_chunks(str(docs))
        # logger.info("Content loaded successfully. Docs: ", docs)
        logger.info(f"Content loaded successfully. Docs: {len(docs)}")

        for doc in docs:
            loaded_content.append(Page(doc))

        # logger.info("Content loaded successfully. Docs: ", docs)

        return loaded_content

    def _get_raw_wikipedia_tex(self, title):
        """
        Fetches the raw MediaWiki format content from a Wikipedia page and converts LaTeX formulas.

        Parameters:
        title (str): The title of the Wikipedia page.

        Returns:
        str: The raw MediaWiki content with LaTeX formulas formatted, or an error message if the page is not found.
        """
        api_url = f"https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'parse',
            'page': title,
            'prop': 'wikitext',
            'format': 'json'
        }
        response = requests.get(api_url, params=params)
        if response.status_code != 200:
            logger.info(f"Failed to fetch page content. Status code: {response.status_code}")
            return self.not_found_message
        data = response.json()
        if 'error' in data:
            logger.info(f"Error fetching page content: {data['error']['info']}")
            return self.not_found_message
        raw_content = data['parse']['wikitext']['*']
        return self._format_latex(raw_content)

    def _get_clean_wikipedia_text(self, title):
        """
        Fetches the formatted content from a Wikipedia page and converts LaTeX formulas.

        Parameters:
        title (str): The title of the Wikipedia page.

        Returns:
        str: The formatted content with LaTeX formulas converted, or an error message if the page is not found.
        """
        wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=self.user_agent)
        page = wiki_wiki.page(title)
        if not page.exists():
            return self.not_found_message
        content = page.text
        return self._format_latex(content)
        # return content

    def _format_latex(self, content):
        """
        Replaces LaTeX formulas in the content with formatted versions.

        Parameters:
        content (str): The content to format.

        Returns:
        str: The content with LaTeX formulas formatted.
        """
        # Identify and format LaTeX formulas (example pattern for LaTeX in Wikipedia)
        def replace_latex(match):
            return f"$$ {match.group(1)} $$"
        latex_pattern = re.compile(r'\{\{math\|(.+?)\}\}')
        return latex_pattern.sub(replace_latex, content)

    def get_wikipedia_content(self, url, is_raw_tex=True):
        """
        Fetches content from a Wikipedia page and processes it.

        Parameters:
        url (str): The URL of the Wikipedia page.
        is_raw_tex (bool): If True, fetches the raw MediaWiki format content and converts LaTeX formulas. 
                           If False, fetches the formatted content with LaTeX formulas converted (clean text only).

        Returns:
        str: The content of the Wikipedia page with LaTeX formulas formatted, or an error message if the page is not found.
        """
        title = url.split('/')[-1]
        if is_raw_tex:
            return self._get_raw_wikipedia_tex(title)
        else:
            return self._get_clean_wikipedia_text(title)

    def get_youtube_content(self, url):
        """
        Fetches the transcript of a YouTube video.

        Parameters:
        url (str): The URL of the YouTube video.

        Returns:
        str: The transcript of the YouTube video, or an error message if fetching fails.
        """
        logger.info(f"Fetching content from YouTube URL: {url}")
        try:
            # Extract the video ID from the URL using regex
            video_id = re.search(r'(?<=v=)[^&]+', url)
            if not video_id:
                logger.info("Error: Invalid YouTube URL.")
                return self.not_found_message
            video_id = video_id.group(0)
            logger.info(f"video_id: {video_id}")
            # Fetch the transcript
            # transcript = YouTubeTranscriptApi.get_transcript(video_id)
            conn = http.client.HTTPSConnection("youtube-transcripts.p.rapidapi.com")
            headers = {
                'x-rapidapi-key': "4993c13a1emshc228b092e6a0cd0p1b7d73jsn122f34065a4d",
                'x-rapidapi-host': "youtube-transcripts.p.rapidapi.com"
            }
            conn.request("GET", "/youtube/transcript?url=" + url + "chunkSize=500", headers=headers)
            res = conn.getresponse()
            data = res.read()
            # print(data.decode("utf-8"))
            
            # Combine the transcript segments into a single string
            # transcript_text = ' '.join([item['text'] for item in transcript])
            transcript_text = str([item["text"] for item in json.loads(data.decode("utf-8"))["content"]])
            logger.info(f"Transcript text: {transcript_text}")
            return transcript_text

        except VideoUnavailable as e:
            logger.exception(f"Error: The video is unavailable as {str(e)}.")
            return self.not_found_message
        except NoTranscriptFound as e:
            logger.exception(f"Error: No transcript found for this video as {str(e)}.")
            return self.not_found_message
        except TranscriptsDisabled as e:
            logger.exception(f"Error: Transcripts are disabled for this video as {str(e)}.")
            return self.not_found_message
        except Exception as e:
            logger.exception(f"An error occurred: {str(e)}")
            return self.not_found_message

    def get_arxiv_content(self, url, dir=""):
        """
        Fetches the content of a PDF from an arXiv link.

        Parameters:
        url (str): The URL of the arXiv PDF.
        dir (str): The directory to save the downloaded PDF file (temporarily).

        Returns:
        str: The content of the arXiv PDF, or an error message if fetching fails.
        """
        dir = self.file_path
        if not os.path.exists(dir):
            os.makedirs(dir)
        try:
            # Extract the arXiv ID from the URL
            arxiv_id = url.split('/')[-1]
            # Search for the paper on arXiv
            search = Search(id_list=[arxiv_id])
            paper = next(search.results())
            # Construct the file path
            pdf_path = os.path.join(dir, f"{arxiv_id}.pdf")
            # Download the PDF
            paper.download_pdf(filename=pdf_path)
            # Extract text using PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_path)
            extracted_text = loader.load()
            # Delete the PDF file
            os.remove(pdf_path)
            return extracted_text
        except Exception as e:
            logger.exception(f"An error occurred: {str(e)}")
            return self.not_found_message

# # Example usage:
# url = "https://arxiv.org/pdf/2405.10501"
# url = "https://www.youtube.com/watch?v=sOF0SsddQ_s"
# url = "https://en.wikipedia.org/wiki/Rabi_cycle"
# handler = ExternalLinksHandler(url)
# content = handler.load()
# logger.info(content)