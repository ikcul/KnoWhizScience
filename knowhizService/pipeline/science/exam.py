import os
import json
import re
import logging
import openai
from openai import RateLimitError

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser

from pipeline.science.api_handler import ApiHandler
from pipeline.science.prompt_handler import PromptHandler
from pipeline.science.utils.cards import CardsUtil  # CardsUtil.divide_corresponding_lists
from pipeline.science.meta_creater import Meta_Creater

from pipeline.science.prompts.exam_prompts import ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.STEM import STEM_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.social_sciences import SocialSciences_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.health_education import HealthEducation_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.business_economics import BusinessEconomics_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.humanities import Humanities_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.law import Law_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.CSDS import CSDS_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.Math import Math_ExamPrompts
from pipeline.science.prompts.subject_exam_prompts.language import Language_ExamPrompts

logger = logging.getLogger("kzpipeline.science.exam")

class Test(Meta_Creater):
    def __init__(self, para, input_dir, output_dir):
        super().__init__(para)
        self.max_test_multiple_choice_questions_per_section = int(para['max_test_multiple_choice_questions_per_section'])
        self.max_test_short_answer_questions_per_section = int(para['max_test_short_answer_questions_per_section'])
        self.prompt = PromptHandler(self.api)
        self.input_dir = self.flashcard_dir # self.input_dir = input_dir
        self.output_dir = self.test_dir     # self.output_dir = output_dir
        self.full_exam_mcq_set = []
        self.full_exam_saq_set = []
        self.prompt_class = self._determine_prompt_class()

    def _determine_prompt_class(self):
        """
        Determines the prompt class to use based on the subject.
        """
        subject = self.course_name_domain["subject"]
        if subject in ['Physics', 'Chemistry', 'Biology', 'Engineering']:
            return STEM_ExamPrompts()
        elif subject in ['Mathematics']:
            return Math_ExamPrompts()
        elif subject in ['Literature', 'Philosophy', 'Art']:
            return Humanities_ExamPrompts()
        elif subject in ['Economics', 'History', 'Geography', 'Political Science', 'Sociology']:
            return SocialSciences_ExamPrompts()
        elif subject in ['Health', 'Physical Education']:
            return HealthEducation_ExamPrompts()
        elif subject in ['Business', 'Finance', 'Accounting', 'Marketing']:
            return BusinessEconomics_ExamPrompts()
        elif subject in ['Law']:
            return Law_ExamPrompts()
        elif subject in ['Computer Science', 'Data Science']:
            return CSDS_ExamPrompts()
        elif subject in ['Language']:
            return Language_ExamPrompts()
        else:
            return ExamPrompts()

    def create_test(self):
        # Create a regex pattern for filenames matching 'flashcards_set{integer}.json'
        pattern = re.compile(r'^flashcards_set(\d+)\.json$')
        #test_types = ['definition', 'expansion']
        test_types = ['definition']
        # Use list comprehension to filter files that match the pattern
        flashcards_files = [file for file in os.listdir(self.input_dir) if pattern.match(file)]
        for flashcard_file in flashcards_files:
            logger.info(flashcard_file)
            try:
                with open(os.path.join(self.input_dir, flashcard_file), 'r') as file:
                    cards = json.load(file)
            except json.JSONDecodeError as e:
                logger.info(f"Error decoding JSON: {e}")
            except FileNotFoundError:
                logger.info("File not found. Please check the file path.")
            keys_list0 = list(cards.keys())
            values_list0 = list(cards.values())
            n = 1
            keys_list, values_list = CardsUtil.divide_corresponding_lists(keys_list0, values_list0, n)
            # keys_list = [keys_list0[:] for _ in range(n)]
            # values_list = [values_list0[:] for _ in range(n)]
            for test_type in test_types:
                file_path_mcq = os.path.join(self.output_dir, flashcard_file.split('.')[0] + f'_test_mcq_{test_type}.json')
                file_path_saq = os.path.join(self.output_dir, flashcard_file.split('.')[0] + f'_test_saq_{test_type}.json')
                if not os.path.exists(file_path_mcq):
                    # response_mcq = self._create_test_query(keys_list, values_list, 'multiple_choice', self.max_test_multiple_choice_questions_per_section, test_type, self.llm_advance)
                    # (llm, keywords, values, qform, test_type, qnum)
                    response_mcq = self.test_generations(self.llm_advance, keys_list, values_list, 'multiple_choice', test_type, self.max_test_multiple_choice_questions_per_section)
                    with open(file_path_mcq, 'w') as file:
                        json.dump(response_mcq, file, indent=2)
                    self.full_exam_mcq_set.append(response_mcq)
                else:
                    try:
                        with open(file_path_mcq, 'r') as file:
                            test_items = json.load(file)
                            self.full_exam_mcq_set.append(test_items)
                    except json.JSONDecodeError as e:
                        logger.info(f"JSONDecodeError: {e}")
                    except FileNotFoundError:
                        logger.info("FileNotFoundError: Please check the file path.")
                if not os.path.exists(file_path_saq):
                    # response_saq = self._create_test_query(keys_list, values_list, 'short_answer_questions', self.max_test_short_answer_questions_per_section, test_type, self.llm_advance)
                    # (llm, keywords, values, qform, test_type, qnum)
                    response_saq = self.test_generations(self.llm_advance, keys_list, values_list, 'short_answer_questions', test_type, self.max_test_short_answer_questions_per_section)
                    with open(file_path_saq, 'w') as file:
                        json.dump(response_saq, file, indent=2)
                    self.full_exam_saq_set.append(response_saq)
                else:
                    try:
                        with open(file_path_saq, 'r') as file:
                            test_items = json.load(file)
                            self.full_exam_saq_set.append(test_items)
                    except json.JSONDecodeError as e:
                        logger.info(f"JSONDecodeError: {e}")
                    except FileNotFoundError:
                        logger.info("FileNotFoundError: Please check the file path.")

    # Test generation
    def test_generations_async(self, llm, keywords, values, qform, test_type, qnum):
        inputs = [{
            "qnum": qnum,
            "text": value,
            "keyword": keyword,
        } for value, keyword in zip(values, keywords)]
        parser = JsonOutputParser()
        error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
        if qform == 'multiple_choice':
            if test_type == 'definition':
                prompt = self.prompt_class.multiple_choice_definition_exam_generation_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                # results = await chain.abatch(inputs)
                results = chain.batch(inputs)
                prompt = self.prompt_class.review_multiple_choice_exam_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                inputs = [{"exam": qa} for qa in results]
                results = chain.batch(inputs)
            if test_type == 'expansion':
                prompt = self.prompt_class.multiple_choice_expansion_exam_generation_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                # results = await chain.abatch(inputs)
                results = chain.batch(inputs)
        if qform == 'short_answer_questions':
            if test_type == 'expansion':
                prompt = self.prompt_class.short_answer_questions_expansion_exam_generation_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                # results = await chain.abatch(inputs)
                results = chain.batch(inputs)
            if test_type == 'definition':
                prompt = self.prompt_class.short_answer_questions_definition_exam_generation_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                # results = await chain.abatch(inputs)
                results = chain.batch(inputs)
                prompt = self.prompt_class.review_short_answer_exam_prompt()
                prompt = ChatPromptTemplate.from_template(prompt)
                chain = prompt | llm | error_parser
                inputs = [{"exam": qa} for qa in results]
                results = chain.batch(inputs)
        return results

    # Generation with given number of attempts
    def test_generations(self, llm, keywords, values, qform, test_type, qnum, max_attempts=3):
        attempt = 0
        llm_sequence = [llm, self.llm_basic_backup_1, self.llm_basic_backup_2]
        llm_index = 0

        while attempt < max_attempts:
            current_llm = llm_sequence[llm_index]
            try:
                return self.test_generations_async(current_llm, keywords, values, qform, test_type, qnum)
            except RateLimitError as e:
                logger.exception(f"RateLimitError encountered on attempt {attempt + 1} with model {current_llm}: {e}")
                if llm_index < len(llm_sequence) - 1:
                    llm_index += 1
                    logger.info(f"Switching to backup model: {llm_sequence[llm_index]}")
                else:
                    attempt += 1
                    llm_index = 0  # Reset to the first model
                    if attempt == max_attempts:
                        logger.info(f"Failed to generate tests after {max_attempts} attempts.")
                        raise Exception(f"Tests generation failed after {max_attempts} attempts.")
            except Exception as e:
                logger.exception(f"Attempt {attempt + 1} failed for generating tests: {e}")
                attempt += 1
                llm_index = 0  # Reset to the first model
                if attempt == max_attempts:
                    logger.info(f"Failed to generate tests after {max_attempts} attempts.")
                    raise Exception(f"Tests generation failed after {max_attempts} attempts.")

    def get_exam_mcq_list(self):
        return self.full_exam_mcq_set

    def get_exam_saq_list(self):
        return self.full_exam_saq_set