import os
import json
import logging
import hashlib
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import ChatPromptTemplate

from pipeline.science.api_handler import ApiHandler
from pipeline.science.doc_handler import DocHandler
from pipeline.science.prompts.zeroshot_flashcards_prompts import ZeroshotPrompts

logger = logging.getLogger("kzpipeline.science.meta_creater")

class Meta_Creater:
    def __init__(self, para):
        self.options_list = para["options_list"]
        self.rich_content = para["rich_content"]    # True: rich content, False: only use one time prompt
        self.regions = para["regions"]
        self.subject_options = ZeroshotPrompts.subject_options()
        self.results_dir = para['results_dir']
        self.number_of_keywords = 10    # Default number of keywords in each chapter when converting anki flashcards

        # load LLMs
        self.api = ApiHandler(para)
        self.llm_advance = self.api.models['advance']['instance']
        self.llm_basic = self.api.models['basic']['instance']
        self.llm_stable = self.api.models['stable']['instance']
        self.llm_creative = self.api.models['creative']['instance']
        self.llm_basic_backup_1 = self.api.models['basic_backup_1']['instance']
        self.llm_basic_backup_2 = self.api.models['basic_backup_2']['instance']

        self.llm_basic_context_window = self.api.models['basic']['context_window']

        if(para['zero_shot']):
            self.course_info = para["course_info"]
            self._hash_course_info()
            self.flashcard_dir = self.results_dir + "flashcards/" + self.course_id + "/"
            self.quiz_dir = self.results_dir + "quiz/" + self.course_id + "/"
            self.test_dir = self.results_dir + "test/" + self.course_id + "/"
            self.course_meta_dir = self.results_dir + "course_meta/" + self.course_id + "/"
            os.makedirs(self.flashcard_dir, exist_ok=True)
            os.makedirs(self.quiz_dir, exist_ok=True)
            os.makedirs(self.test_dir, exist_ok=True)
            os.makedirs(self.course_meta_dir, exist_ok=True)
            logger.info(f"\nself.course_meta_dir: {str(self.course_meta_dir)}")
            self._extract_course_name_domain()
            self.course_name_domain = self.course_name_domain
        else:
            self.docs = DocHandler(para)
            self.flashcard_dir = self.docs.flashcard_dir
            self.quiz_dir = self.docs.quiz_dir
            self.test_dir = self.docs.test_dir
            self.course_meta_dir = self.docs.course_meta_dir
            self.book_embedding_dir = self.docs.book_embedding_dir
            self.course_name_domain = self.docs.course_name_domain

    def _hash_course_info(self):
        """
        Hash the course description.
        """
        # Initialize a hashlib object for SHA-224
        logger.info("Hashing course information.")
        sha224_hash = hashlib.sha224()
        sha224_hash.update(self.course_info.encode("utf-8"))

        # Calculate the final hash
        self.course_id = sha224_hash.hexdigest()
        logger.info(f"Course information hashed: {self.course_id}")

    def _extract_course_name_domain(self):
        """
        Get the course_name_domain based on self.course_info = para["course_info"].
        """
        llm = self.llm_advance
        llm = self.llm_basic
        if(os.path.exists(self.course_meta_dir + "course_name_domain.json")):
            with open(self.course_meta_dir + "course_name_domain.json", 'r') as file:
                self.course_name_domain = json.load(file)
        else:
            # Support complex input formats
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt = ZeroshotPrompts.topic_extraction_prompt()
            prompt = ChatPromptTemplate.from_template(prompt)
            chain = prompt | llm | error_parser
            response = chain.invoke({'course_info': self.course_info, 'subject_options': self.subject_options})
            self.course_name_domain = response
            self.level = response["level"]
            with open(self.course_meta_dir + "course_name_domain.json", 'w') as file:
                json.dump(self.course_name_domain, file, indent=2)