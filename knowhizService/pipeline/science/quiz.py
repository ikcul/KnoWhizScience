import os
import json
import numpy as np
import re
import logging
import openai
from openai import RateLimitError

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.science.api_handler import ApiHandler
from pipeline.science.prompt_handler import PromptHandler
from pipeline.science.meta_creater import Meta_Creater

from pipeline.science.prompts.quiz_prompts import QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.STEM import STEM_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.social_sciences import SocialSciences_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.health_education import HealthEducation_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.business_economics import BusinessEconomics_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.humanities import Humanities_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.law import Law_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.CSDS import CSDS_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.Math import Math_QuizPrompts
from pipeline.science.prompts.subjects_quiz_prompts.language import Language_QuizPrompts

logger = logging.getLogger("kzpipeline.science.quiz")

class Quiz(Meta_Creater):
    def __init__(self, para, input_dir, output_dir):
        super().__init__(para)
        self.max_quiz_questions_per_section = int(para["max_quiz_questions_per_section"])
        self.quiz_random_seed = int(para["quiz_random_seed"])
        self.prompt = PromptHandler(self.api)
        self.input_dir = self.flashcard_dir # self.input_dir = input_dir
        self.output_dir = self.quiz_dir     # self.output_dir = output_dir
        self.full_quiz_set = []
        self.prompt_class = self._determine_prompt_class()

    def _determine_prompt_class(self):
        """
        Determines the prompt class to use based on the subject.
        """
        subject = self.course_name_domain["subject"]
        if subject in ['Physics', 'Chemistry', 'Biology', 'Engineering']:
            return STEM_QuizPrompts()
        elif subject in ['Mathematics']:
            return Math_QuizPrompts()
        elif subject in ['Literature', 'Philosophy', 'Art']:
            return Humanities_QuizPrompts()
        elif subject in ['Economics', 'History', 'Geography', 'Political Science', 'Sociology']:
            return SocialSciences_QuizPrompts()
        elif subject in ['Health', 'Physical Education']:
            return HealthEducation_QuizPrompts()
        elif subject in ['Business', 'Finance', 'Accounting', 'Marketing']:
            return BusinessEconomics_QuizPrompts()
        elif subject in ['Law']:
            return Law_QuizPrompts()
        elif subject in ['Computer Science', 'Data Science']:
            return CSDS_QuizPrompts()
        elif subject in ['Language']:
            return Language_QuizPrompts()
        else:
            return QuizPrompts()

    def create_quiz(self):
        # Create a regex pattern for filenames matching 'flashcards_set{integer}.json'
        pattern = re.compile(r'^flashcards_set(\d+)\.json$')
        #quiz_types = ['definition', 'expansion']
        quiz_types = ['definition']
        # Use list comprehension to filter files that match the pattern
        flashcards_files_in_dir = [file for file in os.listdir(self.input_dir) if pattern.match(file)]
        flashcards_files = []
        for i in range(len(flashcards_files_in_dir)):
            flashcards_files.append(f'flashcards_set{i}.json')
        for flashcard_file in flashcards_files:
            logger.info(flashcard_file)
            try:
                with open(os.path.join(self.input_dir, flashcard_file), 'r') as file:
                    cards = json.load(file)
                    # Process the 'cards' as needed
            except json.JSONDecodeError as e:
                logger.info(f"Error decoding JSON: {e}")
            except FileNotFoundError:
                logger.info("File not found. Please check the file path.")
            keys_list = list(cards.keys())
            values_list = list(cards.values())
            ncards = len(cards)
            np.random.seed(self.quiz_random_seed)
            q_num = min(ncards, self.max_quiz_questions_per_section)
            sample_cards_indices = np.random.choice(ncards, q_num, replace=False)
            for quiz_type in quiz_types:
                file_path = os.path.join(self.output_dir, flashcard_file.split('.')[0] + f'_quiz_{quiz_type}.json')
                if not os.path.exists(os.path.join(file_path)):
                    keywords = []
                    values = []
                    responses = []
                    for index in sample_cards_indices:
                        keywords.append(keys_list[index])
                        values.append(values_list[index][quiz_type])
                    responses = self.quiz_generations(self.llm_basic, keywords, values, 'multiple_choice', quiz_type, self.llm_advance)
                    logger.info('\n')
                    quiz = responses
                    self.full_quiz_set.append(quiz)
                    with open(file_path, 'w') as file:
                        json.dump(quiz, file, indent=2)
                else:
                    try:
                        with open(file_path, 'r') as file:
                            quiz = json.load(file)
                            self.full_quiz_set.append(quiz)
                    except json.JSONDecodeError as e:
                        logger.info(f"JSONDecodeError: {e}")
                    except FileNotFoundError:
                        logger.info("FileNotFoundError: Please check the file path.")

    # Quiz generation
    def quiz_generations_async(self, llm, keywords, values, qform, quiz_type, llm_advance):
        inputs = [{
            "text": value,
            "keyword": keyword,
        } for value, keyword in zip(values, keywords)]
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        if qform == 'multiple_choice':
            if quiz_type == 'definition':
                prompt = self.prompt_class.multiple_choice_definition_quiz_generation_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                # results = await chain.abatch(inputs)
                results = chain.batch(inputs)
                prompt = self.prompt_class.review_multiple_choice_quiz_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm_advance | error_parser
                inputs = [{"quiz": qa} for qa in results]
                results = chain.batch(inputs)
            if quiz_type == 'expansion':
                prompt = self.prompt_class.multiple_choice_expansion_quiz_generation_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                # results = await chain.abatch(inputs)
                results = chain.batch(inputs)
        return dict(zip(keywords, results))

    # Generation with given number of attempts
    def quiz_generations(self, llm, keywords, values, qform, quiz_type, llm_advance, max_attempts=3):
        attempt = 0
        # llm_sequence = [llm, self.llm_basic_backup_1, self.llm_basic_backup_2]
        llm_sequence = [llm]
        llm_index = 0

        while attempt < max_attempts:
            current_llm = llm_sequence[llm_index]
            try:
                return self.quiz_generations_async(current_llm, keywords, values, qform, quiz_type, llm_advance)
            except RateLimitError as e:
                logger.exception(f"RateLimitError encountered on attempt {attempt + 1} with model {current_llm}: {e}")
                if llm_index < len(llm_sequence) - 1:
                    llm_index += 1
                    logger.info(f"Switching to backup model: {llm_sequence[llm_index]}")
                else:
                    attempt += 1
                    llm_index = 0  # Reset to the first model
                    if attempt == max_attempts:
                        logger.info(f"Failed to generate quizzes after {max_attempts} attempts.")
                        raise Exception(f"Quizzes generation failed after {max_attempts} attempts.")
            except Exception as e:
                logger.exception(f"Attempt {attempt + 1} failed for generating quizzes: {e}")
                attempt += 1
                llm_index = 0  # Reset to the first model
                if attempt == max_attempts:
                    logger.info(f"Failed to generate quizzes after {max_attempts} attempts.")
                    raise Exception(f"Quizzes generation failed after {max_attempts} attempts.")

    def get_quizzes_list(self):
        return self.full_quiz_set