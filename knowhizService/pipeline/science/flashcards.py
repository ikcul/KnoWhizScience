import os
import re
import json
import math
import logging
import shutil
import openai
from openai import RateLimitError

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.helper.azure_blob import AzureBlobHelper
from pipeline.science.prompt_handler import PromptHandler
from pipeline.science.utils.cards import CardsUtil    # CardsUtil.combine_cards, CardsUtil.find_indices_to_remove, CardsUtil.divide_into_groups, CardsUtil.locate_indices_to_sets
from pipeline.science.utils.string import StringUtil    # StringUtil.ensure_utf8_encoding
from pipeline.science.meta_creater import Meta_Creater

from pipeline.science.prompts.flashcards_prompts import FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.STEM import STEM_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.social_sciences import SocialSciences_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.health_education import HealthEducation_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.business_economics import BusinessEconomics_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.humanities import Humanities_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.law import Law_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.CSDS import CSDS_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.Math import Math_FlashcardsPrompts
from pipeline.science.prompts.subjects_flashcards_prompts.language import Language_FlashcardsPrompts

logger = logging.getLogger("kzpipeline.science.flashcards")

class Flashcards(Meta_Creater):
    def __init__(self, para):
        super().__init__(para)
        self.definition_detail_level = para["definition_detail_level"]
        self.expansion_detail_level = para["expansion_detail_level"]
        self.max_flashcard_definition_words = int(para["definition_detail_level"] * 30 + 20)
        self.max_flashcard_expansion_words = int(para["expansion_detail_level"] * 100 + 100)

        self.main_filenames = para.get('main_filenames', [])
        logger.info(f"Main filenames: {self.main_filenames}")
        if len(self.main_filenames) != 1:
            logger.info("Failed. Please only upload one main file.")
            raise Exception("Multiple main file failed")
        self.flashcards_set_size = int(para['flashcards_set_size'])
        self.link_flashcards_size = int(para['link_flashcards_size'])   # Size of flashcards generated from links
        self.max_flashcards_size = int(para['max_flashcards_size'])

        self.similarity_score_thresh = float(para['similarity_score_thresh'])
        self.keywords_per_page = para["keywords_per_page"]
        # book with no index
        # self.num_context_pages = para["num_context_pages"]
        self.num_context_pages = min(self.docs.main_docs_pages[0], para["num_context_pages"])
        self.page_set_size = para["page_set_size"]
        self.overlapping = para["overlapping"]

        self.prompt = PromptHandler(self.api)
        self.prompt_class = self._determine_prompt_class()
        self.nMain_doc_pages = self.docs.main_docs_pages[0]
        self.main_doc = self.docs.main_docs[0]
        self.nkeywords = int(min(self.keywords_per_page * self.nMain_doc_pages, self.max_flashcards_size))
        self.keywords = []
        if(self.docs.main_file_types[0] != 'apkg'):
            self.main_embedding = self.docs.main_embedding[0]

    def _determine_prompt_class(self):
        """
        Determines the prompt class to use based on the subject.
        """
        subject = self.course_name_domain["subject"]
        if subject in ['Physics', 'Chemistry', 'Biology', 'Engineering']:
            return STEM_FlashcardsPrompts()
        elif subject in ['Mathematics']:
            return Math_FlashcardsPrompts()
        elif subject in ['Literature', 'Philosophy', 'Art']:
            return Humanities_FlashcardsPrompts()
        elif subject in ['Economics', 'History', 'Geography', 'Political Science', 'Sociology']:
            return SocialSciences_FlashcardsPrompts()
        elif subject in ['Health', 'Physical Education']:
            return HealthEducation_FlashcardsPrompts()
        elif subject in ['Business', 'Finance', 'Accounting', 'Marketing']:
            return BusinessEconomics_FlashcardsPrompts()
        elif subject in ['Law']:
            return Law_FlashcardsPrompts()
        elif subject in ['Computer Science', 'Data Science']:
            return CSDS_FlashcardsPrompts()
        elif subject in ['Language']:
            return Language_FlashcardsPrompts()
        else:
            return FlashcardsPrompts()

    def create_keywords(self):
        """
        Generates keywords from the document content. If a previously generated keywords file exists,
        it loads the keywords from that file. Otherwise, it determines whether the document has an index
        section and selects the appropriate method to generate keywords. Finally, it refines the generated
        keywords.
        """
        if(self.docs.main_file_types[0] != 'apkg' and self.docs.main_file_types[0] != 'link'):
            file_path = os.path.join(self.docs.flashcard_dir,  "sorted_keywords_docs.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    keywords_docs = json.load(file)
                    self.keywords = list(keywords_docs.keys())
                    self.nkeywords = len(self.keywords)
                    self.keywords_qdocs = list(keywords_docs.values())
                return  # Exit the method after loading the keywords

            logger.info("\nCreating keywords...")
            # Check if the document has an index section and call the appropriate method
            if self.docs.indx_page_docs[0].loc[0, 'index_pages'] is not None:
                self._create_keywords_with_index()
            else:
                self._create_keywords_without_index()

            logger.info("\nRefining keywords...")
            self._refine_keywords()
        elif(self.docs.main_file_types[0] == 'apkg'):
            self.keywords = [page.anki_content["Question"] for page in self.docs.main_docs[0]]
            self.nkeywords = len(self.keywords)
            self.number_of_keywords = int(math.sqrt(self.nkeywords)) * 2
            logger.info((self.keywords))
            logger.info(f"\nLength of keywords: {self.nkeywords}")
        elif(self.docs.main_file_types[0] == 'link'):
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt = self.prompt_class.keywords_extraction_links_prompt()
            prompt = ChatPromptTemplate.from_template(prompt)
            chain = prompt | self.llm_advance | error_parser
            self.keywords = chain.invoke({'text': self.docs.textbook_content_pages})["concepts"]
            logger.info(f"\nKeywords for links: {self.keywords}")
            logger.info(f"\nLength of keywords: {len(self.keywords)}")
            logger.info(f"\nType of keywords: {type(self.keywords)}")
            
            # Ensure we have at least 5 keywords per chapter
            min_required_keywords = max(5, self.link_flashcards_size)
            
            if len(self.keywords) < min_required_keywords:
                # If we have fewer keywords than the minimum required, generate more
                logger.info(f"\nNot enough keywords generated (only {len(self.keywords)}). Regenerating to ensure at least {min_required_keywords} keywords...")
                prompt = self.prompt_class.keywords_extraction_links_prompt() + self.prompt_class.keywords_extraction_fix_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | self.llm_advance | error_parser
                self.keywords = chain.invoke({'text': self.docs.textbook_content_pages, 'nkeys': min_required_keywords})["concepts"]
            elif(len(self.keywords) > self.link_flashcards_size):
                # If we have more keywords than the desired number of flashcards, we need to re-generate the keywords
                logger.info("\nRe-generating keywords for links with smaller amount of keywords...")
                prompt = self.prompt_class.keywords_extraction_links_prompt() + self.prompt_class.keywords_extraction_fix_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | self.llm_advance | error_parser
                self.keywords = chain.invoke({'text': self.docs.textbook_content_pages[:self.link_flashcards_size], 'nkeys': self.link_flashcards_size})["concepts"]
                
            # Final check to ensure we have at least 5 keywords
            if len(self.keywords) < 5:
                logger.info("\nStill not enough keywords. Forcing generation of at least 5 keywords...")
                prompt = self.prompt_class.keywords_extraction_links_prompt() + "\nImportant: You MUST generate at least 5 distinct concepts, even if you need to be creative."
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | self.llm_advance | error_parser
                additional_keywords = chain.invoke({'text': self.docs.textbook_content_pages, 'nkeys': 5})["concepts"]
                
                # Add new keywords without duplicates
                self.keywords = list(set(self.keywords + additional_keywords))
                
                # Ensure at least 5 keywords
                while len(self.keywords) < 5:
                    self.keywords.append(f"Additional Topic {len(self.keywords) + 1}")

            file_path = os.path.join(self.docs.flashcard_dir, f'raw_keywords{0}.json')
            with open(file_path, 'w') as file:
                json.dump(self.keywords, file, indent=2)

            # logger.info(f"\nlink_flashcards_size: {self.link_flashcards_size}")
            # logger.info(f"\nKeywords for links: {self.keywords}")

    def _create_keywords_with_index(self, max_token=2048):
        """
        Creates keywords from the index section of documents using a language model.

        Parameters:
        - max_token (int): The maximum token limit for processing chunks of the index.
        """
        # TEST
        print("Creating keywords with index")

        docs = self.docs.indx_page_docs[0]['index_docs']
        index_pages = "".join([docs[i] for i in range(len(docs))])
        index_chunks = self.prompt.split_prompt(str(index_pages), 'basic', custom_token_limit=max_token)
        n = len(index_chunks)
        logger.info(f'index_chunks: {n}.')
        nkeywords_in_chunk = CardsUtil.divide_into_groups(self.nkeywords, ngroup=n)
        logger.info(f'nkeywords: {nkeywords_in_chunk}')
        for i in range(n):
            file_path = os.path.join(self.docs.flashcard_dir, f'raw_keywords{i}.json')
            if not os.path.exists(file_path):
                parser = JsonOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = self.prompt_class.keywords_extraction_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | self.llm_advance | error_parser
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'index_docs': index_chunks[i], 'nkey': nkeywords_in_chunk[i]})

                with open(file_path, 'w') as file:
                    json.dump(response['Keywords'], file, indent=2)

    def _refine_keywords(self):
        if os.path.exists(os.path.join(self.docs.flashcard_dir, 'keywords.json')):
            with open(os.path.join(self.docs.flashcard_dir, 'keywords.json'), 'r') as file:
                self.keywords = json.load(file)
            return
        #refine keywords
        # Create a regex pattern for filenames matching 'flashcards_set{integer}.json'
        pattern = re.compile(r'^raw_keywords(\d+)\.json$')
        # Use list comprehension to filter files that match the pattern
        raw_keywords_files = [file for file in os.listdir(self.docs.flashcard_dir) if pattern.match(file)]
        raw_keywords = []
        for raw_keywords_file in raw_keywords_files:
            with open(os.path.join(self.docs.flashcard_dir, raw_keywords_file), 'r') as file:
                keywords = json.load(file)
                raw_keywords.append(keywords)
        # self.raw_keywords = " ".join([raw_keywords[i] for i in range(len(raw_keywords))])
        self.raw_keywords = [item for sublist in raw_keywords for item in sublist]

        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
        prompt = self.prompt_class.keywords_extraction_refine_prompt()
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | self.llm_advance | error_parser
        response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'keywords': self.raw_keywords})
        self.keywords = list(set(response['Keywords']))
        self.nkeywords = len(self.keywords)
        logger.info(f"the number of keywords: {len(self.keywords)} ")
        with open(os.path.join(self.docs.flashcard_dir, 'keywords.json'), 'w') as file:
            json.dump(self.keywords, file, indent=2)

    def _create_keywords_without_index(self):
        """
        Generates keywords for a course based on its document content, excluding an index section.
        This process involves segmenting the document, summarizing segments, and extracting essential keywords.
        """
        # TEST
        print("Creating keywords without index")

        # Determine the number of chunks by dividing the total page count by the desired chunk size.
        chunk_sizes = CardsUtil.divide_into_groups(self.nMain_doc_pages, group_size=self.num_context_pages)
        nchunks = len(chunk_sizes)

        # TEST
        print(f"Number of chunks: {nchunks}")

        nkeywords_in_chunk = CardsUtil.divide_into_groups(self.nkeywords, ngroup=nchunks)

        # TEST
        print(f"Number of keywords in each chunk: {nkeywords_in_chunk}")

        start_page = 0
        # Iterate through each chunk to process its content.
        for i in range(nchunks):
            end_page = start_page + chunk_sizes[i]
            file_path = os.path.join(self.docs.flashcard_dir, f'raw_keywords{i}.json')
            if not os.path.exists(file_path):
                temp_page_content = ""
                for k in range(start_page, end_page):
                    temp_page_content += self.main_doc[k].page_content + " "
                ntempkeys = min(self.keywords_per_page * chunk_sizes[i], nkeywords_in_chunk[i]) # The number of temp keys to extract.
                ntempkeys = int(ntempkeys)

                print("ntempkeys: ", ntempkeys)

                # First step: Use a basic LLM to summarize the chunk's content.
                parser = StrOutputParser()
                prompt = """As as a professor teaching the course: {course_name_domain}.""" + self.prompt_class.summary_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | self.llm_basic | parser
                try:
                    response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'text': temp_page_content})
                    temp_extracted_content = response
                except Exception as e:
                    logger.exception(f"Exception: {e}")
                    # logger.info(f"Failed to summarize the content of chunk {i}: {e}")
                    temp_extracted_content = self.prompt.summarize_prompt(temp_page_content, 'basic', custom_token_limit=int(self.llm_basic_context_window/1.2))

                # Second step: Use an advanced LLM to identify essential keywords from the summarized content.
                parser = JsonOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = self.prompt_class.keywords_extraction_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | self.llm_advance | error_parser
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, 'nkey': ntempkeys, 'index_docs': temp_extracted_content})
                with open(file_path, 'w') as file:
                    json.dump(response['Keywords'], file, indent=2)
            start_page = end_page

    def _find_keywords_docs(self):
        embed_book = self.main_embedding
        self.keywords_qdocs = []
        for i in range(len(self.chapters_list)):
            logger.info(f"\nSearching qdocs for chapter: {i}")
            keywords_temp = self.keywords_list[i]
            file_path = os.path.join(self.docs.flashcard_dir, f'main_qdocs_set{i}.json')
            if not os.path.exists(file_path):
                qdocs_list_temp = []
                for keyword in keywords_temp:
                    docs = embed_book.similarity_search(keyword, k=4)
                    logger.info(f"\nDocs for keyword: {keyword}")
                    # logger.info(f"\nDocs: {docs}")
                    qdocs = "".join([docs[i].page_content for i in range(len(docs))])
                    qdocs = qdocs.replace('\u2022', '').replace('\n', '').replace('\no', '').replace('. .', '')
                    qdocs_list_temp.append(qdocs)
                    self.keywords_qdocs.append(qdocs)
                with open(file_path, 'w') as file:
                    json.dump(dict(zip(keywords_temp, qdocs_list_temp)), file, indent=2)
            else:
                with open(file_path, 'r') as file:
                    qdocs_list_dict_temp = json.load(file)
                    extracted_qdocs = [qdocs_list_dict_temp[key] for key in keywords_temp]
                    self.keywords_qdocs.extend(extracted_qdocs)

        file_path = os.path.join(self.docs.flashcard_dir, 'keywords_docs.json')
        with open(file_path, 'w') as file:
            json.dump(self.keywords_qdocs, file, indent=2)

    def _create_flashcards_anki(self, course_id: str):
        '''
        Create flashcards for Anki
        '''
        self.anki_dict = {}
        for i in range(len(self.docs.main_docs[0])):
            self.anki_dict[self.docs.main_docs[0][i].anki_content["Question"]] = self.docs.main_docs[0][i].anki_content["Answer"]
        # logger.info(f"\n anki_dict: {(self.anki_dict)}")
        self.full_flashcards_set = []

        for i in range(len(self.chapters_list)):
            keywords = self.keywords_list[i]
            definitions = []
            expansions = []
            contexts = []
            for keyword in keywords:
                definition = self.anki_dict[keyword]
                definitions.append(definition)
                contexts.append('')
            definitions = dict(zip(keywords, definitions))
            # expansions = dict(zip(keywords, expansions))
            # expansions = self.generate_expansions(self.llm_basic, keywords, contexts, definitions, self.docs.course_name_domain, self.max_flashcard_definition_words, self.max_flashcard_expansion_words, 3, regions=self.regions)
            expansions = {keyword: '' for keyword in keywords}
            cards = CardsUtil.combine_cards(definitions, expansions)
            self.full_flashcards_set.append(cards)
            file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set{i}.json')
            with open(file_path, 'w') as file:
                json.dump(cards, file, indent=2)
            file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set_def{i}.json')
            with open(file_path, 'w') as file:
                json.dump(definitions, file, indent=2)
            file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set_exp{i}.json')
            with open(file_path, 'w') as file:
                json.dump(expansions, file, indent=2)
        file_path = os.path.join(self.docs.flashcard_dir, 'full_flashcards_set.json')
        with open(file_path, 'w') as file:
            json.dump(self.full_flashcards_set, file, indent=2)

        # Adding images to the flashcards folder
        def copy_image_files(src_dir, dest_dir):
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            for item in os.listdir(src_dir):
                if item.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg')):
                    src_file = os.path.join(src_dir, item)
                    dest_file = os.path.join(dest_dir, item)
                    shutil.copy2(src_file, dest_file)

        src_dir = os.path.join(self.docs.book_dir, self.docs.main_filenames[0].split('.')[0]) + ".apkg_extracted"
        logger.info(f"\nSource directory: {src_dir}")
        dest_dir = self.docs.flashcard_dir
        logger.info(f"\nDestination directory: {dest_dir}")
        copy_image_files(src_dir, dest_dir)
        media_dir = src_dir
        container_name = "knowhizmedia"

        image_url_mapping = self.upload_media_files(media_dir, container_name, course_id)
        full_flashcards_json = json.dumps(self.full_flashcards_set)
        def replace_local_paths(match):
            filename = os.path.basename(match.group(0))
            if filename in image_url_mapping:
                public_url = image_url_mapping[filename]
                return public_url
            return match.group(0)

        for ext in ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.tiff', '.svg']:
            pattern = r'(?!https?://)[\w/]*' + re.escape(ext)
            full_flashcards_json = re.sub(pattern, replace_local_paths, full_flashcards_json)

        self.full_flashcards_set = json.loads(full_flashcards_json)
        for i, cards in enumerate(self.full_flashcards_set):
            file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set{i}.json')
            with open(file_path, 'w') as file:
                json.dump(cards, file, indent=2)
        file_path = os.path.join(self.docs.flashcard_dir, 'full_flashcards_set.json')
        with open(file_path, 'w') as file:
            json.dump(self.full_flashcards_set, file, indent=2)

    # Definition generation
    def generate_definitions_async(self, llm, keywords, texts, max_words_flashcards, max_words_expansion):
        inputs = [{
            "max_words_flashcards": max_words_flashcards,
            "text": text,
            "keyword": keyword,
        } for text, keyword in zip(texts, keywords)]
        # parser = JsonOutputParser()
        parser = StrOutputParser()
        prompt = self.prompt_class.flashcards_definition_prompt()
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | llm | parser
        # results = await chain.abatch(inputs)
        results = chain.batch(inputs)
        return dict(zip(keywords, results))

    # Definition generation with given number of attempts
    def generate_definitions(self, llm, keywords, texts, max_words_flashcards, max_words_expansion, max_attempts=3):
        attempt = 0
        llm_sequence = [llm, self.llm_basic_backup_1, self.llm_basic_backup_2]
        llm_index = 0

        while attempt < max_attempts:
            current_llm = llm_sequence[llm_index]
            try:
                return self.generate_definitions_async(current_llm, keywords, texts, max_words_flashcards, max_words_expansion)
            except RateLimitError as e:
                logger.exception(f"RateLimitError encountered on attempt {attempt + 1} with model {current_llm}: {e}")
                if llm_index < len(llm_sequence) - 1:
                    llm_index += 1
                    logger.info(f"Switching to backup model: {llm_sequence[llm_index]}")
                else:
                    attempt += 1
                    llm_index = 0  # Reset to the first model
                    if attempt == max_attempts:
                        logger.info(f"Failed to generate definitions after {max_attempts} attempts.")
                        raise Exception(f"Definitions generation failed after {max_attempts} attempts.")
            except Exception as e:
                logger.exception(f"Attempt {attempt + 1} failed for generating definitions: {e}")
                attempt += 1
                llm_index = 0  # Reset to the first model
                if attempt == max_attempts:
                    logger.info(f"Failed to generate definitions after {max_attempts} attempts.")
                    raise Exception(f"Definitions generation failed after {max_attempts} attempts.")

    # Expansion generation
    def generate_expansions_async(self, llm, keywords, texts, defs, course_name_domain, max_words_flashcards, max_words_expansion, regions = ["Example"]):
        def format_string(regions):
            markdown_content = "\n".join([f'## {region}\n\nExample content for {region}.\n' for region in regions])
            markdown_format_string = f"""
            {markdown_content}
            """
            return markdown_format_string
        markdown_format_string = format_string(regions)

        inputs = [{
            "max_words_expansion": max_words_expansion,
            "text": text,
            "definition": definition,
            "keyword": keyword,
            "course_name_domain": course_name_domain,
            "markdown_format_string": markdown_format_string,
        } for text, keyword, definition in zip(texts, keywords, defs)]
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = self.prompt_class.flashcards_expansion_prompt()
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | llm | error_parser
        # results = await chain.abatch(inputs)
        results = chain.batch(inputs)
        return dict(zip(keywords, results))

    # Expansion generation with given number of attempts
    def generate_expansions(self, llm, keywords, texts, defs, course_name_domain, max_words_flashcards, max_words_expansion, max_attempts=3, regions=["Example"]):
        attempt = 0
        llm_sequence = [llm, self.llm_basic_backup_1, self.llm_basic_backup_2]
        llm_index = 0

        while attempt < max_attempts:
            current_llm = llm_sequence[llm_index]
            try:
                return self.generate_expansions_async(current_llm, keywords, texts, defs, course_name_domain, max_words_flashcards, max_words_expansion, regions)
            except RateLimitError as e:
                logger.exception(f"RateLimitError encountered on attempt {attempt + 1} with model {current_llm}: {e}")
                if llm_index < len(llm_sequence) - 1:
                    llm_index += 1
                    logger.info(f"Switching to backup model: {llm_sequence[llm_index]}")
                else:
                    attempt += 1
                    llm_index = 0  # Reset to the first model
                    if attempt == max_attempts:
                        logger.info(f"Failed to generate expansions after {max_attempts} attempts.")
                        raise Exception(f"Expansions generation failed after {max_attempts} attempts.")
            except Exception as e:
                logger.exception(f"Attempt {attempt + 1} failed for generating expansions: {e}")
                attempt += 1
                llm_index = 0  # Reset to the first model
                if attempt == max_attempts:
                    logger.info(f"Failed to generate expansions after {max_attempts} attempts.")
                    raise Exception(f"Expansions generation failed after {max_attempts} attempts.")

    # Rich content options generation
    def generate_rich_content_options(self, llm, keywords, content_list, chapter_name, course_name, options_list = ["Mindmap", "Table", "Formula", "Code", "Image"]):
        """
        Generate rich content format options for the given keywords
        """
        options_map = self.prompt_class.rich_content_generation_prompts_map()
        inputs = [{
            "course_name": course_name,
            "chapter_name": chapter_name,
            "keyword": keyword,
            "content": content_list[keyword],
            "option": options_list,
        } for keyword in keywords]
        parser = StrOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = self.prompt_class.rich_content_generation_options_prompt()
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | llm | error_parser
        # options = await chain.abatch(inputs)
        options = chain.batch(inputs)
        for i in range(len(options)):
            options[i] = re.sub(r'[^a-zA-Z0-9 ]', '', options[i])

        # logger.info(f"Options: {options}")

        formats = []
        for i in range(len(options)):
            if options[i] in options_list:
                formats.append(options_map[options[i]])
            else:
                formats.append("Sentence")

        # logger.info(f"Formats: {formats}")

        formats = dict(zip(keywords, formats))
        options = dict(zip(keywords, options))
        return formats, options

    # Rich content generation
    def generate_rich_content(self, llm, keywords, content_list, chapter_name, course_name, formats, options):
        """
        Generate rich content for the given keywords with the given format options
        """
        inputs = [{
            "course_name": course_name,
            "chapter_name": chapter_name,
            "keyword": keyword,
            "content": content_list[keyword],
            "format": formats[keyword],
            "option": options[keyword],
        } for keyword in keywords]
        parser = StrOutputParser()
        # parser = XMLOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = self.prompt_class.rich_content_generation_prompt()
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | llm | error_parser
        # rich_contents = await chain.abatch(inputs)
        rich_contents = chain.batch(inputs)

        # final_roots = nest_dict_to_xml(rich_contents)

        return dict(zip(keywords, rich_contents))

    # Rich content generation with given number of attempts
    def robust_generate_rich_content(self, llm, keywords, content_list, chapter_name, course_name, options_list=["Mindmap", "Table", "Formula", "Code", "Image"], max_attempts=3, if_parallel=True):
        attempt = 0
        llm_sequence = [llm, self.llm_basic_backup_1, self.llm_basic_backup_2]
        llm_index = 0

        while attempt < max_attempts:
            current_llm = llm_sequence[llm_index]
            try:
                if if_parallel:
                    formats, options = self.generate_rich_content_options(current_llm, keywords, content_list, chapter_name, course_name, options_list=options_list)
                    return self.generate_rich_content(current_llm, keywords, content_list, chapter_name, course_name, formats, options)
                else:
                    results = {}
                    for keyword in keywords:
                        formats, options = self.generate_rich_content_options(current_llm, [keyword], content_list, chapter_name, course_name, options_list=options_list)
                        result = self.generate_rich_content(current_llm, [keyword], content_list, chapter_name, course_name, formats, options)
                        results.update(result)
                    return results
            except RateLimitError as e:
                logger.exception(f"RateLimitError encountered on attempt {attempt + 1} with model {current_llm}: {e}")
                if llm_index < len(llm_sequence) - 1:
                    llm_index += 1
                    logger.info(f"Switching to backup model: {llm_sequence[llm_index]}")
                else:
                    attempt += 1
                    llm_index = 0  # Reset to the first model
                    if attempt == max_attempts:
                        logger.info(f"Failed to generate rich content after {max_attempts} attempts due to RateLimitError.")
                        raise Exception(f"Rich content generation failed after {max_attempts} attempts.")
            except Exception as e:
                logger.exception(f"Attempt {attempt + 1} failed for generating rich content: {e}")
                attempt += 1
                llm_index = 0  # Reset to the first model
                if attempt == max_attempts:
                    logger.info(f"Failed to generate rich content after {max_attempts} attempts.")
                    raise Exception(f"Rich content generation failed after {max_attempts} attempts.")


    def upload_media_files(self, media_dir, container_name, course_id):
        azure_blob_helper = AzureBlobHelper()
        image_url_mapping = {}
        for filename in os.listdir(media_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                local_image_path = os.path.join(media_dir, filename)
                print(local_image_path)
                if os.path.exists(local_image_path):
                    print("exist")
                    blob_name = f"image/{course_id}/{filename}"
                    public_url = azure_blob_helper.upload(local_image_path, blob_name, container_name)
                    image_url_mapping[filename] = public_url
                else:
                    logger.info(f"Image not found at path: {local_image_path}. Skipping this upload.")
        return image_url_mapping

    def create_flashcards(self, user_id="", course_id="", flashcards_write_cb=None):
        # If the main file type is Anki, create flashcards directly
        if self.docs.main_file_types[0] == 'apkg':
            logger.info(f"\nCourse ID is: {self.docs.course_id}")
            logger.info("\nCreating chapters and assigning keywords...")
            self._create_chapters()
            self._asign_keywords_direct()
            self._create_flashcards_anki(course_id)

            if flashcards_write_cb is not None:
                for i, cards in enumerate(self.full_flashcards_set):
                    flashcards_write_cb(user_id, course_id, i, cards, self.chapters_list)
            return

        # If the main file type is Link, create flashcards directly
        elif self.docs.main_file_types[0] == 'link':
            logger.info(f"\nCourse ID is: {self.docs.course_id}")
            logger.info("\nCreating chapters and assigning keywords...")
            self._create_chapters()
            self._asign_keywords_k2c()
            logger.info("\nSearching keywords in documents...")
            self._find_keywords_docs()

        # If the main file type is not Anki or Link, proceed with creating flashcards
        else:
            logger.info(f"\nCourse ID is: {self.docs.course_id}")
            logger.info("\nCreating chapters and assigning keywords...")
            self._create_chapters()
            try:
                self._asign_keywords_c2k()
            except Exception as e:
                logger.exception(f"Exception: {e}")
                self._asign_keywords_k2c()
            logger.info("\nSearching keywords in documents...")
            self._find_keywords_docs()

        file_path = os.path.join(self.docs.flashcard_dir, "keywords_docs.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                self.keywords_qdocs = json.load(file)
        else:
            logger.info("keywords_docs.json file not found. Please check the file path.")
        start = 0
        self.full_flashcards_set = []
        for i in range(len(self.chapters_list)):
            cards = {}
            keywords_temp = self.keywords_list[i]
            file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set{i}.json')
            end = start + len(keywords_temp)
            if not os.path.exists(file_path):
                keywords = []
                texts = []
                # We can further reduce cost here
                for j in range(start, end):
                    keyword = keywords_temp[j-start]
                    keywords.append(keyword)
                    logger.info(f'keyword: {keyword}')

                    qdocs = self.keywords_qdocs[j]
                    # logger.info("\nLength of qdocs: ", len(self.keywords_qdocs))
                    if self.docs.nSupp > 0:
                        qdocs_supps = self._match_docs_in_supp(keyword)
                        # logger.info("\nContent of qdocs_supps: ", qdocs_supps)
                        qdocs = qdocs + qdocs_supps
                    qdocs_summary = self.prompt.summarize_prompt(qdocs, 'basic', custom_token_limit=int(self.llm_basic_context_window/1.2))
                    texts.append(qdocs_summary)

                file_path = os.path.join(self.docs.flashcard_dir, f'qdocs_set{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(dict(zip(keywords, texts)), file, indent=2)

                file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set_def{i}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        cards_def = json.load(file)
                else:
                    cards_def = self.generate_definitions(self.llm_basic, keywords, texts, self.max_flashcard_definition_words, self.max_flashcard_expansion_words)
                file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set_def{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(cards_def, file, indent=2)

                definitions_list = [item for item in cards_def.values()]
                keywords = list(cards_def.keys())

                file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set_exp{i}.json')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        cards_exp = json.load(file)
                else:
                    try:
                        cards_exp = self.generate_expansions(self.llm_basic, keywords, texts, definitions_list, self.docs.course_name_domain, self.max_flashcard_definition_words, self.max_flashcard_expansion_words, 3, regions=self.regions)
                    except Exception as e:
                        logger.exception(f"Error generating expansions for chapter {i}: {e}")
                        # continue  # Skip this iteration and proceed with the next chapter

                chapters_name_temp = self.chapters_list[i]
                keywords_list_temp = self.keywords_list[i]

                if(self.rich_content == True):
                    # Generate rich content for the definitions
                    # llm = self.llm_advance
                    logger.info("Generating rich content for the definitions...")
                    llm = self.llm_basic
                    rich_content = self.robust_generate_rich_content(llm, keywords_list_temp, cards_exp, chapters_name_temp, self.course_name_textbook_chapters["Course name"], options_list=self.options_list)
                    cards_exp = rich_content

                cards = CardsUtil.combine_cards(cards_def, cards_exp)

                self.full_flashcards_set.append(cards)

                file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set_exp{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(cards_exp, file, indent=2)
                file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set{i}.json')
                with open(file_path, 'w') as file:
                    json.dump(cards, file, indent=2)

            else:
                try:
                    with open(file_path, 'r') as file:
                        cards = json.load(file)
                except json.JSONDecodeError as e:
                    logger.info(f"JSONDecodeError: {e}")
                except FileNotFoundError:
                    logger.info("FileNotFoundError: Please check the file path.")
                self.full_flashcards_set.append(cards)
            start = end

            if(flashcards_write_cb is not None):
                flashcards_write_cb(user_id, course_id, i, cards, self.chapters_list)

        print("Removing duplicated flashcards...")
        logger.info("Removing duplicated flashcards...")
        self._remove_duplicated_flashcards()

    def _match_docs_in_supp(self, keyword):
        qdocs_supps = ""
        for ell in range(self.docs.nSupp):
            docs_supp = self.docs.supp_embedding[ell].similarity_search(keyword, k=4)
            qdocs_supp =  "".join([docs_supp[m].page_content for m in range(len(docs_supp))])
            qdocs_supp = qdocs_supp.replace('\u2022', '').replace('\n', '').replace('\no', '').replace('. .', '')
            qdocs_supps += qdocs_supp
        qdocs_supps_summary = self.prompt.summarize_prompt(qdocs_supps, 'basic', custom_token_limit= int(self.llm_basic_context_window/1.2))
        return qdocs_supps_summary

    def _remove_duplicated_flashcards(self):
        # logger.info("\nRemoving duplicated flashcards...")
        all_flashcards = {k: v for d in self.full_flashcards_set for k, v in d.items()}
        keywords = list(all_flashcards.keys())
        answers = list(all_flashcards.values())
        # Convert the answers in dict format to JSON strings for similarity comparison
        answers = [json.dumps(d) for d in answers]
        indices = list(CardsUtil.find_indices_to_remove(keywords, texts=[keywords, answers], thresh=self.similarity_score_thresh))

        # Calculate the sets size for each chapter
        self.sets_size = []
        pattern = re.compile(r'^flashcards_set(\d+)\.json$')
        # List and sort the files based on the number in the file name
        raw_flashcards_files = sorted(
            [file for file in os.listdir(self.docs.flashcard_dir) if pattern.match(file)],
            key=lambda x: int(pattern.match(x).group(1))
        )
        logger.info(f"\nFile list: {raw_flashcards_files}")
        for temp in raw_flashcards_files:
            with open(os.path.join(self.docs.flashcard_dir, temp), 'r') as file:
                flashcards_temp = json.load(file)
                self.sets_size.append(len(flashcards_temp))
        logger.info(f"\nSets size: {self.sets_size}")

        # logger.info(f"Number of duplicates found: {len(indices)}")
        if indices:
            mapped_indices = CardsUtil.locate_indices_to_sets(indices, self.sets_size)
            for i in range(len(indices)):
                set_label = mapped_indices[i][0]
                logger.info("Keywords to remove: " + keywords[indices[i]])
                self.full_flashcards_set[set_label].pop(keywords[indices[i]], None)
                file_path = os.path.join(self.docs.flashcard_dir, f'flashcards_set{set_label}.json')
                with open(file_path, 'w') as file:
                    json.dump(self.full_flashcards_set[set_label], file, indent=2)
        else:
            logger.info("No duplicates found based on the threshold.")
        return

    def _create_chapters(self):
        llm = self.llm_advance
        path = os.path.join(self.docs.course_meta_dir, "course_name_textbook_chapters.json")
        # Check if the course_name_textbook_chapters.json file exists and load the chapters
        if os.path.exists(path):
            with open(path, 'r') as file:
                self.course_name_textbook_chapters = json.load(file)
                self.chapters_list = self.course_name_textbook_chapters["Chapters"]

        # If the file does not exist, generate the chapters
        else:
            # Check if the main file type is a link and generate chapters accordingly
            if(self.docs.main_file_types[0] == 'link'):
                parser = StrOutputParser()
                error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
                prompt = ChatPromptTemplate.from_template(
                    """
                    For the given text ```{text}```, generate a concise title for it within 10 words.
                    """)
                chain = prompt | self.llm_basic | error_parser
                try:
                    response = chain.invoke({'text': self.docs.textbook_content_pages})

                except Exception as e:
                    logger.exception(f"Exception: {e}")
                    textbook_content_summary = self.prompt.summarize_prompt(self.docs.textbook_content_pages, 'basic', custom_token_limit=int(self.llm_basic_context_window/1.2))
                    response = chain.invoke({"text": textbook_content_summary})

                self.chapters_list = [response]
                self.course_name_textbook_chapters = {
                    "Course name": self.docs.course_name_domain,
                    "Textbooks": [self.docs.main_filenames[0]],
                    "Chapters": self.chapters_list
                }
                path = os.path.join(self.docs.course_meta_dir, "course_name_textbook_chapters.json")
                with open(path, 'w') as file:
                    json.dump(self.course_name_textbook_chapters, file, indent=2)

                return
            elif(self.docs.main_file_types[0] == 'apkg'):
                # Proceed to assign keywords to chapters
                number_of_keywords = self.number_of_keywords
                total_keywords = len(self.keywords)
                number_of_chapters = math.ceil(total_keywords / number_of_keywords)
                self.chapters_list = [f"Chapter {i+1}" for i in range(number_of_chapters)]

                # Split keywords into chunks of size number_of_keywords
                self.keywords_list = [self.keywords[i:i + number_of_keywords] for i in range(0, total_keywords, number_of_keywords)]
                self.course_name_textbook_chapters = {
                    "Course name": self.docs.course_name_domain,
                    "Textbooks": [self.docs.main_filenames[0]],
                    "Chapters": self.chapters_list
                }
                path = os.path.join(self.docs.course_meta_dir, "course_name_textbook_chapters.json")
                with open(path, 'w') as file:
                    json.dump(self.course_name_textbook_chapters, file, indent=2)

                return

            # If the main file type is not a link, generate the chapters using the LLM
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            # error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
            prompt = self.prompt_class.chapters_creation_with_content_prompt()
            prompt = ChatPromptTemplate.from_template(prompt)
            chain = prompt | self.llm_basic | error_parser

            # # TEST
            # print("The content pages are: ", self.docs.textbook_content_pages)

            try:
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, "textbook_content_pages": self.docs.textbook_content_pages})
                # logger.info("\n\nThe response is: ", response)
                # print(f"\n\nThe course_name_textbook_chapters self.docs.textbook_content_pages is: {type(self.docs.textbook_content_pages)}")
                self.course_name_textbook_chapters = response
                self.chapters_list = self.course_name_textbook_chapters["Chapters"]
            except Exception as e:
                logger.exception(f"Exception: {e}")
                # Sometimes the API fails to generate the chapters. In such cases, we regenerate the chapters with summarized content.
                chain = prompt | self.llm_stable | error_parser
                textbook_content_summary = self.prompt.summarize_prompt(self.docs.textbook_content_pages, 'basic', custom_token_limit=int(self.llm_basic_context_window/1.2))
                response = chain.invoke({'course_name_domain': self.docs.course_name_domain, "textbook_content_pages": textbook_content_summary})
                logger.info(f"\n\nThe retried with stable course_name_domain response is: {response}")
                self.course_name_textbook_chapters = response
                self.chapters_list = self.course_name_textbook_chapters["Chapters"]

            # Check if the number of chapters is less than 5 and regenerate the chapters if so.
            # logger.info("\nThe list of chapters is: ", self.course_name_textbook_chapters["Chapters"])
            if(len(self.course_name_textbook_chapters["Chapters"]) <= 5 or len(self.course_name_textbook_chapters["Chapters"]) > 15):
                logger.info("\n\nThe number of chapters is less than 5. Please check the chapters.")
                # logger.info(f"\n\nThe chapters are: {(self.course_name_textbook_chapters["Chapters"])}")
                prompt = self.prompt_class.chapters_creation_no_content_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | self.llm_basic | error_parser
                try:
                    response = chain.invoke({'course_name_domain': self.docs.course_name_domain})
                    self.course_name_textbook_chapters = response
                    self.chapters_list = self.course_name_textbook_chapters["Chapters"]
                except Exception as e:
                    logger.exception(f"Exception: {e}")
                    chain = prompt | self.llm_stable | error_parser
                    response = chain.invoke({'course_name_domain': self.docs.course_name_domain})
                    self.course_name_textbook_chapters = response
                    self.chapters_list = self.course_name_textbook_chapters["Chapters"]

                logger.info(f"\n\nThe refined chapters are: {response}")

            path = os.path.join(self.docs.course_meta_dir, "course_name_textbook_chapters.json")
            with open(path, 'w') as file:
                json.dump(self.course_name_textbook_chapters, file, indent=2)

    # Assign keywords to each chapters, from chapters to keywords (c2k) or from keywords to chapters (k2c)
    def _asign_keywords_c2k(self):
        """
        Asigning keywords to each chapter based on the content of the chapters. Review the content of the chapters and assign keywords to each chapter all together.
        Go from chapter to keywords.

        Key variables changed:
        - self.keywords_list
        - self.chapters_list
        - self.keywords_text_in_chapters

        Key files changed/created:
        - path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json") - most important
        - path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        - path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        - path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        """
        llm = self.llm_advance

        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        if os.path.exists(path):
            logger.info("File exists. Loading data from file.")
            with open(path, 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.keywords_list = data_temp["keywords_list"]
            return

        self.keywords_min_num = max(int(len(self.keywords) / len(self.chapters_list) / 2), 7)
        self.keywords_max_num = max(int(len(self.keywords) / len(self.chapters_list) / 1.2), 7) + 5
        # logger.info("\nmin_num is: ", self.keywords_min_num)
        # logger.info("max_num is: ", self.keywords_max_num)

        path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        # Check if the "keywords_text_in_chapters.txt" file exists. If it does, load and extract the chapters and keywords from the file.
        if os.path.exists(path):
            with open(path, 'r') as file:
                self.keywords_text_in_chapters = file.read()
            logger.info("\n")

            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt_2 = self.prompt_class.keywords_assignment_c2k_prompt()
            prompt_2 = ChatPromptTemplate.from_template(prompt_2)
            chain_2 = prompt_2 | self.llm_advance | error_parser
            response = chain_2.invoke({'keywords_text_in_chapters': self.keywords_text_in_chapters, 'chapters_list': self.chapters_list})
            self.keywords_list = response["keywords_list"]

        # If the file does not exist, create the chapters and assign keywords to each chapter.
        else:
            logger.info("\n")
            # 3. Make sure each chapter has at least {min_num} keywords and no more than {max_num} keywords.
            parser = StrOutputParser()
            prompt_1 = self.prompt_class.keywords_assignment_refinement_prompt()
            prompt_1 = ChatPromptTemplate.from_template(prompt_1)
            chain_1 = prompt_1 | self.llm_advance | parser
            response = chain_1.invoke({'course_name_textbook_chapters': self.course_name_textbook_chapters, 'sorted_keywords': self.keywords, 'min_num': self.keywords_min_num, 'max_num': self.keywords_max_num})
            self.keywords_text_in_chapters = response
            
            logger.info("\n")
            parser = JsonOutputParser()
            error_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_basic)
            prompt_3 = self.prompt_class.keywords_assignment_refinement_format_prompt()
            prompt_3 = ChatPromptTemplate.from_template(prompt_3)
            chain_3 = prompt_3 | self.llm_basic | error_parser
            response = chain_3.invoke({'keywords_text_in_chapters': self.keywords_text_in_chapters, 'chapters_list': self.chapters_list})
            self.keywords_list = response["keywords_list"]

        # If the keywords in some chapters are less than the minimum number of keywords, reassign keywords to the chapters.
        if(any(len(keyword_group) < 5 for keyword_group in self.keywords_list)):
            logger.info("\nThe number of keywords in some chapters is less than 5. Reassigning keywords to the chapters...")
            parser = StrOutputParser()
            prompt_1 = self.prompt_class.keywords_assignment_refinement_fix_prompt()
            prompt_1 = ChatPromptTemplate.from_template(prompt_1)
            chain_1 = prompt_1 | self.llm_advance | parser
            response = chain_1.invoke({'course_name_textbook_chapters': self.course_name_textbook_chapters, 'sorted_keywords': self.keywords, 'min_num': self.keywords_min_num, 'max_num': self.keywords_max_num, 'keywords_list_original': self.keywords_list})
            self.keywords_text_in_chapters = response

            logger.info("\n")
            parser = JsonOutputParser()
            prompt_3 = self.prompt_class.keywords_assignment_refinement_fix_format_prompt()
            prompt_3 = ChatPromptTemplate.from_template(prompt_3)
            chain_3 = prompt_3 | self.llm_basic | parser
            response = chain_3.invoke({'keywords_text_in_chapters': self.keywords_text_in_chapters, 'chapters_list': self.chapters_list})
            self.keywords_list = response["keywords_list"]

        path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        with open(path, 'w') as file:
            file.write(self.keywords_text_in_chapters)
        path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        with open(path, 'w') as file:
            json.dump(self.chapters_list, file, indent=2)
        path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        with open(path, 'w') as file:
            json.dump(self.keywords_list, file, indent=2)

        logger.info(f"\n\nself.chapters_list are:\n\n{self.chapters_list}")
        logger.info(f"\n\nself.keywords_list are:\n\n{self.keywords_list}")

        data_temp = {
            "chapters_list": self.chapters_list,
            "keywords_list": self.keywords_list
        }

        if(len(self.chapters_list) != len(self.keywords_list)):
            raise ValueError("The number of chapters and keywords do not match.")

        # Save to JSON file
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        with open(path, 'w') as json_file:
            json.dump(data_temp, json_file, indent=4)
        return data_temp

    def _asign_keywords_direct(self):
        """
        Assigning keywords to chapters based on the number of keywords we want to have in each chapter.
        The chapter name is just Chapter 1, Chapter 2, etc.

        Key variables changed:
        - self.keywords_list
        - self.chapters_list
        - self.keywords_text_in_chapters

        Key files changed/created:
        - path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json") - most important
        - path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        - path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        """
        # First, check if the "chapters_and_keywords.json" file exists. If it does, load data from the file.
        number_of_keywords = self.number_of_keywords
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        if os.path.exists(path):
            logger.info("File exists. Loading data from file.")
            with open(path, 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.keywords_list = data_temp["keywords_list"]
            return

        # Proceed to assign keywords to chapters
        total_keywords = len(self.keywords)
        number_of_chapters = math.ceil(total_keywords / number_of_keywords)
        self.chapters_list = [f"Chapter {i+1}" for i in range(number_of_chapters)]

        # Split keywords into chunks of size number_of_keywords
        self.keywords_list = [self.keywords[i:i + number_of_keywords] for i in range(0, total_keywords, number_of_keywords)]

        # Generate keywords_text_in_chapters
        self.keywords_text_in_chapters = ""
        for chapter_name, keywords in zip(self.chapters_list, self.keywords_list):
            self.keywords_text_in_chapters += f"{chapter_name}:\n"
            self.keywords_text_in_chapters += ", ".join(keywords) + "\n\n"

        if len(self.chapters_list) != len(self.keywords_list):
            raise ValueError("The number of chapters and keywords lists do not match.")

        # Save files
        data_temp = {
            "chapters_list": self.chapters_list,
            "keywords_list": self.keywords_list
        }

        # Save to chapters_and_keywords.json
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        with open(path, 'w') as json_file:
            json.dump(data_temp, json_file, indent=4)

        # Save chapters_list.json
        path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        with open(path, 'w') as file:
            json.dump(self.chapters_list, file, indent=2)

        # Save keywords_list.json
        path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        with open(path, 'w') as file:
            json.dump(self.keywords_list, file, indent=2)

        # Save keywords_text_in_chapters.txt
        path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        with open(path, 'w') as file:
            file.write(self.keywords_text_in_chapters)

        logger.info(f"\n\nself.chapters_list are:\n\n{self.chapters_list}")
        logger.info(f"\n\nself.keywords_list are:\n\n{self.keywords_list}")

        return data_temp

    def _asign_keywords_direct(self):
        """
        Assigning keywords to chapters based on the number of keywords we want to have in each chapter.
        The chapter name is just Chapter 1, Chapter 2, etc.

        Key variables changed:
        - self.keywords_list
        - self.chapters_list
        - self.keywords_text_in_chapters

        Key files changed/created:
        - path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json") - most important
        - path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        - path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        """
        # First, check if the "chapters_and_keywords.json" file exists. If it does, load data from the file.
        number_of_keywords = self.number_of_keywords
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        if os.path.exists(path):
            logger.info("File exists. Loading data from file.")
            with open(path, 'r') as json_file:
                data_temp = json.load(json_file)
                self.chapters_list = data_temp["chapters_list"]
                self.keywords_list = data_temp["keywords_list"]
            return

        # Proceed to assign keywords to chapters
        total_keywords = len(self.keywords)
        number_of_chapters = math.ceil(total_keywords / number_of_keywords)
        self.chapters_list = [f"Chapter {i+1}" for i in range(number_of_chapters)]

        # Split keywords into chunks of size number_of_keywords
        self.keywords_list = [self.keywords[i:i + number_of_keywords] for i in range(0, total_keywords, number_of_keywords)]

        # Generate keywords_text_in_chapters
        self.keywords_text_in_chapters = ""
        for chapter_name, keywords in zip(self.chapters_list, self.keywords_list):
            self.keywords_text_in_chapters += f"{chapter_name}:\n"
            self.keywords_text_in_chapters += ", ".join(keywords) + "\n\n"

        if len(self.chapters_list) != len(self.keywords_list):
            raise ValueError("The number of chapters and keywords lists do not match.")

        # Save files
        data_temp = {
            "chapters_list": self.chapters_list,
            "keywords_list": self.keywords_list
        }

        # Save to chapters_and_keywords.json
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        with open(path, 'w') as json_file:
            json.dump(data_temp, json_file, indent=4)

        # Save chapters_list.json
        path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        with open(path, 'w') as file:
            json.dump(self.chapters_list, file, indent=2)

        # Save keywords_list.json
        path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        with open(path, 'w') as file:
            json.dump(self.keywords_list, file, indent=2)

        # Save keywords_text_in_chapters.txt
        path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        with open(path, 'w') as file:
            file.write(self.keywords_text_in_chapters)

        logger.info(f"\n\nself.chapters_list are:\n\n{self.chapters_list}")
        logger.info(f"\n\nself.keywords_list are:\n\n{self.keywords_list}")

        return data_temp

    # Chapter assignment generation
    def assign_chapters_async(self, llm, keywords, chapters, course_name):
        inputs = [{
                "course_name": course_name,
                "chapters": chapters,
                "keyword": keyword,
                } for keyword in keywords]
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        prompt = self.prompt_class.keywords_assignment_k2c_prompt()
        prompt = ChatPromptTemplate.from_template(prompt)
        chain = prompt | llm | error_parser
        # results = await chain.abatch(inputs)
        results = chain.batch(inputs)
        return dict(zip(keywords, results))

    # Chapter assignment generation with given number of attempts
    def assign_chapters(self, llm, keywords, chapters, course_name, max_attempts = 3):
        attempt = 0
        llm_sequence = [llm, self.llm_basic_backup_1, self.llm_basic_backup_2]
        llm_index = 0

        while attempt < max_attempts:
            current_llm = llm_sequence[llm_index]
            try:
                data = self.assign_chapters_async(current_llm, keywords, chapters, course_name)
                ordered_chapters = chapters

                # Initialize an empty dictionary
                chapters_dict = {}

                # Populate the dictionary with chapters and their respective keywords
                for keyword, info in data.items():
                    chapter = info['chapter']
                    if chapter not in chapters_dict:
                        chapters_dict[chapter] = []
                    chapters_dict[chapter].append(keyword)

                # # Filter and sort the chapters_dict based on the ordered_chapters list
                # sorted_chapters_dict = {chapter: chapters_dict[chapter] for chapter in ordered_chapters if chapter in chapters_dict}

                # Extract chapter names and their keywords
                chapter_names = list(chapters_dict.keys())
                chapter_keywords = [' '.join(keywords) for keywords in chapters_dict.values()]
                # Create a CountVectorizer instance
                vectorizer = CountVectorizer().fit_transform(ordered_chapters + chapter_names)
                vectors = vectorizer.toarray()
                # Calculate cosine similarity
                cosine_matrix = cosine_similarity(vectors)
                # Find the best match for each chapter in ordered_chapters
                sorted_chapters_dict = {}
                for idx, ordered_chapter in enumerate(ordered_chapters):
                    # Get the cosine similarity scores for the current chapter
                    similarities = cosine_matrix[idx, len(ordered_chapters):]
                    # Find the index of the best match
                    best_match_idx = similarities.argmax()
                    best_match_chapter = chapter_names[best_match_idx]
                    sorted_chapters_dict[ordered_chapter] = chapters_dict[best_match_chapter]

                # Extract chapters and keywords into separate lists
                chapters_list = list(sorted_chapters_dict.keys())
                keywords_list = list(sorted_chapters_dict.values())

                # Construct the final JSON structure
                final_data = {
                    'chapters_list': chapters_list,
                    'keywords_list': keywords_list
                }
                return final_data

            except RateLimitError as e:
                logger.exception(f"RateLimitError encountered on attempt {attempt + 1} with model {current_llm}: {e}")
                if llm_index < len(llm_sequence) - 1:
                    llm_index += 1
                    logger.info(f"Switching to backup model: {llm_sequence[llm_index]}")
                else:
                    attempt += 1
                    llm_index = 0  # Reset to the first model
                    if attempt == max_attempts:
                        logger.info(f"Failed to assign chapters after {max_attempts} attempts.")
                        raise Exception(f"Chapter assignment failed after {max_attempts} attempts.")
            except Exception as e:
                logger.exception(f"Attempt {attempt + 1} failed for assigning chapters: {e}")
                attempt += 1
                llm_index = 0  # Reset to the first model
                if attempt == max_attempts:
                    logger.info(f"Failed to assign chapters after {max_attempts} attempts.")
                    raise Exception(f"Chapter assignment failed after {max_attempts} attempts.")

    # Assign keywords to each chapters, from chapters to keywords (c2k) or from keywords to chapters (k2c)
    def _asign_keywords_k2c(self):
        """
        Asigning each keyword an index of chapter given the list of chapters.
        Go from keywords to chapters.

        Key variables changed:
        - self.keywords_list
        - self.chapters_list
        - self.keywords_text_in_chapters

        Key files changed/created:
        - path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json") - most important
        - path = os.path.join(self.docs.course_meta_dir, "keywords_text_in_chapters.txt")
        - path = os.path.join(self.docs.course_meta_dir, "chapters_list.json")
        - path = os.path.join(self.docs.course_meta_dir, "keywords_list.json")
        """
        llm = self.llm_basic
        path = os.path.join(self.docs.flashcard_dir, "chapters_and_keywords.json")
        if os.path.exists(path):
            logger.info("File exists. Loading data from file.")
            with open(path, 'r') as json_file:
                data_temp = json.load(json_file)
            self.chapters_list = data_temp["chapters_list"]
            self.keywords_list = data_temp["keywords_list"]
            self.keywords = [keyword for keyword_group in self.keywords_list for keyword in keyword_group]
            self.nkeywords = len(self.keywords)
            logger.info(f"The number of final keywords from loaded file: {self.nkeywords}")
        else:
            # Send the prompt to the API and get response
            data_temp = self.assign_chapters(llm = self.llm_basic, keywords = self.keywords, chapters = self.chapters_list, course_name = self.docs.course_name_domain)
            self.chapters_list = data_temp["chapters_list"]
            self.keywords_list = data_temp["keywords_list"]
            self.keywords = [keyword for keyword_group in self.keywords_list for keyword in keyword_group]
            self.nkeywords = len(self.keywords)
            logger.info(f"The number of final keywords: {self.nkeywords}")
            # Save to JSON file
            with open(path, 'w') as json_file:
                json.dump(data_temp, json_file, indent=4)

    def get_chapters_flashcards_list(self):
        return self.full_flashcards_set

    def get_all_flashcards_list(self):
        all_flashcards = {k: v for d in self.full_flashcards_set for k, v in d.items()}
        return all_flashcards

    def get_chapters_list(self):
        return self.chapters_list

    def get_hash_id(self):
        return self.docs.course_id

    def get_course_name(self):
        if "Course name" in self.course_name_textbook_chapters:
            return self.course_name_textbook_chapters["Course name"]
        else:
            return ""
