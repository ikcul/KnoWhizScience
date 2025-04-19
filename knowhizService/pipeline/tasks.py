#!env python3
# coding: utf-8

import time
import openai
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pipeline.helper.azure_blob import AzureBlobHelper
# from pipeline.helper.mongo_ops_update import MongoOps
from pipeline.helper.mongo_ops_with_backend import MongoOps

from pipeline.science.flashcards import Flashcards
from pipeline.science.zeroshot_flashcards import Zeroshot_Flashcards
from pipeline.science.quiz import Quiz
from pipeline.science.exam import Test
from pipeline.config.config import Config
from pipeline.helper.config_gen import is_valid_url
import json

# Configure logging
logger = logging.getLogger("kzpipeline.task")
# print("start to load config")
config = Config()
# print("start to load azure container name")
AZURE_CONTAINER_NAME = config.get_azure_container_name()
# print("start to load openAI key")
openai.api_key = os.environ.get('OPENAI_API_KEY')
# print("start to load azure blob helper")
azure_blob = AzureBlobHelper()
# print("start to load mongodb")
db_ops = MongoOps()
# print("flashcard generating begin")

def mongo_write_sections_and_flashcards(user_id, course_id, chapter_index, flashcards, chapters):
    db_ops.mongo_write_sections_and_flashcards(user_id, course_id, chapter_index, flashcards, chapters)

@retry(
    stop=stop_after_attempt(4),  # Retry up to 3 times
    wait=wait_exponential(multiplier=20),  # Exponential backoff starting at 60 seconds
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_flashcards(para, context):
    try:
        if context:
            context.thread_local_storage.invocation_id = context.invocation_id
        logger.info("flashcard generation...")
        st = time.time()
        retries = 0
        if para["zero_shot"]:
            myflashcards = Zeroshot_Flashcards(para)
            myflashcards_hash_id = myflashcards.get_hash_id()
            print(f"flashcard generating hash: {myflashcards_hash_id}\nparameters: {para}")

            # Update para setting file with generate_hash
            para['generate_hash'] = myflashcards_hash_id
            para_str = json.dumps(para, indent=2)
            pars_filepath = f"{para['global_file_dir']}/{para['pipeline_id']}.json"

            with open(pars_filepath, 'w') as f:
                f.write(para_str)

            try:
                logger.info("start create flashcards")
                myflashcards.create_chapters()
                logger.info("create_chapters done")
                myflashcards.create_keywords()
                logger.info("create_keywords done")
                myflashcards.create_flashcards(user_id=para['user_id'], course_id=para['course_id'], flashcards_write_cb=mongo_write_sections_and_flashcards)
                logger.info("create_flashcards done")
            except Exception as e:
                logger.exception(f"An error occurred while generating flashcards: {e}")
                db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_GEN_FLASHCARDS")
                db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                raise

            quizzes = []
            tests_mcq = []
            tests_saq = []
            try:
                if para["generate_quiz"]:
                    myquiz = Quiz(para, myflashcards.flashcard_dir, myflashcards.quiz_dir)
                    myquiz.create_quiz()
                    logger.info("create_quiz done")
                    quizzes = myquiz.get_quizzes_list()
            except Exception as e:
                logger.exception(f"An error occurred while generating quizzes: {e}")
                db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_GEN_QUIZ")
                db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                raise

            try:
                if para["generate_test"]:
                    mytest = Test(para, myflashcards.flashcard_dir, myflashcards.test_dir)
                    mytest.create_test()
                    logger.info("create_quiz done")
                    tests_mcq = mytest.get_exam_mcq_list()
                    tests_saq = mytest.get_exam_saq_list()
            except Exception as e:
                logger.exception(f"An error occurred while generating tests: {e}")
                db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_GEN_TEST")
                db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                raise

            course_description = ""
            if myflashcards.course_name_domain and "course_name_domain" in myflashcards.course_name_domain:
                course_description = myflashcards.course_name_domain["course_name_domain"]
            try:
                latest_course = db_ops.mongo_get_course_by_course_id(para['course_id'])
                if "status" in latest_course and latest_course["status"] == "DELETED":
                    print(f"The course is deleted. Stop to update course / pipeline / flashcards. user_id: {para['user_id']}, course_id: {para['course_id']}, pipeline_id: {para['pipeline_id']}")
                else:
                    db_ops.mongo_write_course(para['user_id'], para['course_id'], para['pipeline_id'], chapters=myflashcards.get_chapters_list(), chapter_flashcards=myflashcards.get_chapters_flashcards_list(), quizzes=quizzes, tests_mcq=tests_mcq, tests_saq=tests_saq, zeroshot=True, course_description=course_description, material_url="", supplementary_material_urls=[], course_name=myflashcards.get_course_name(), generate_hash=myflashcards_hash_id, is_shareable=True, course_generation_type=para['course_generation_type'])
            except Exception as e:
                logger.exception(f"An error occurred while writing course data: {e}")
                db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_WRITE_DATA")
                db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                raise

            if para["discord_bot"]:
                logger.exception(f"Sending Discord bot message for pipeline_id: {para['pipeline_id']}")
                db_ops.discord_bot_send_message(channel_id=para["discord_channel_id"], message=f"https://www.knowhiz.us/share/flashcards/{para['course_id']}", mention=para["discord_mention"])
        else:
            myflashcards_hash_id = ""
            try:
                if not os.path.exists(f"{para['book_dir']}/"):
                    os.makedirs(f"{para['book_dir']}")

                if not is_valid_url(para['material_url']):
                    temp_filename = para['book_dir'] + para['main_filenames'][0]
                    azure_blob.download(para['material_url'], temp_filename, AZURE_CONTAINER_NAME)

                for key, supplementary_material_url in enumerate(para['supplementary_material_urls']):
                    if not is_valid_url(supplementary_material_url):
                        if isinstance(supplementary_material_url, str) and supplementary_material_url != "":
                            if not os.path.exists(f"{para['book_dir']}/"):
                                os.makedirs(f"{para['book_dir']}")
                            temp_filename = para['book_dir'] + para['supplementary_filenames'][key]
                            azure_blob.download(supplementary_material_url, temp_filename, AZURE_CONTAINER_NAME)
            except Exception as e:
                logger.exception(f"An error occurred while downloading materials: {e}")
                db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_DOWNLOAD_MATERIAL")
                db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                raise

            myflashcards = Flashcards(para)
            myflashcards_hash_id = myflashcards.get_hash_id()
            print(f"flashcard generating hash: {myflashcards_hash_id}\nparameters: {para}")

            para['generate_hash'] = myflashcards_hash_id
            para_str = json.dumps(para, indent=2)
            pars_filepath = f"{para['global_file_dir']}/{para['pipeline_id']}.json"
            with open(pars_filepath, 'w') as f:
                f.write(para_str)

            try:
                logger.info("start create flashcards")
                myflashcards.create_keywords()
                logger.info("create_keywords done")
                myflashcards.create_flashcards(user_id=para['user_id'], course_id=para['course_id'], flashcards_write_cb=mongo_write_sections_and_flashcards)
                logger.info("create_flashcards done")
            except Exception as e:
                logger.exception(f"An error occurred while generating flashcards: {e}")
                db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_GEN_FLASHCARDS")
                db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                raise

            quizzes = []
            tests_mcq = []
            tests_saq = []
            if not (para['course_generation_type'] == "ANKI"):
                try:
                    if para["generate_quiz"]:
                        myquiz = Quiz(para, myflashcards.docs.flashcard_dir, myflashcards.docs.quiz_dir)
                        myquiz.create_quiz()
                        logger.info("create_quiz done")
                        quizzes = myquiz.get_quizzes_list()
                except Exception as e:
                    logger.exception(f"An error occurred while generating quizzes: {e}")
                    db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_GEN_QUIZ")
                    db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                    raise

                try:
                    if para["generate_test"]:
                        mytest = Test(para, myflashcards.docs.flashcard_dir, myflashcards.docs.test_dir)
                        mytest.create_test()
                        logger.info("create_test done")
                        tests_mcq = mytest.get_exam_mcq_list()
                        tests_saq = mytest.get_exam_saq_list()
                except Exception as e:
                    logger.exception(f"An error occurred while generating tests: {e}")
                    db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_GEN_TEST")
                    db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                    raise

            try:
                latest_course = db_ops.mongo_get_course_by_course_id(para['course_id'])
                if "status" in latest_course and latest_course["status"] == "DELETED":
                    print(f"The course is deleted. Stop to update course / pipeline / flashcards. user_id: {para['user_id']}, course_id: {para['course_id']}, pipeline_id: {para['pipeline_id']}")
                else:
                    is_shareable = False
                    course_name = ""
                    if (para['course_generation_type'] == "ANKI"):
                        is_shareable = True
                    else:
                        course_name = myflashcards.get_course_name()
                    course_description = ""
                    if myflashcards.docs.course_name_domain and "course_name_domain" in myflashcards.docs.course_name_domain:
                        course_description = myflashcards.docs.course_name_domain["course_name_domain"]
                    db_ops.mongo_write_course(para['user_id'], para['course_id'], para['pipeline_id'], chapters=myflashcards.get_chapters_list(), chapter_flashcards=myflashcards.get_chapters_flashcards_list(), quizzes=quizzes, tests_mcq=tests_mcq, tests_saq=tests_saq, zeroshot=False, course_description=course_description, material_url=para['material_url'], supplementary_material_urls=para['supplementary_material_urls'], course_name=course_name, generate_hash=myflashcards_hash_id, is_shareable=is_shareable, course_generation_type=para['course_generation_type'])
            except Exception as e:
                logger.exception(f"An error occurred while writing course data: {e}")
                db_ops.mongo_update_pipeline_status(para['pipeline_id'], myflashcards_hash_id, "FAILED", "FAIL_WRITE_DATA")
                db_ops.mongo_update_course_status(para['course_id'], "PROCESSING_ERROR", para['course_generation_type'])
                raise
    except Exception as e:
        logger.exception(f"Final failure after all retries for pipeline_id: {para['pipeline_id']}: {e}")
        db_ops.mongo_update_pipeline_status(para['pipeline_id'], None, "FINAL_FAIL", "FAIL_AFTER_RETRIES")
        db_ops.mongo_update_course_status(para['course_id'], "FINAL_PROCESSING_ERROR", para['course_generation_type'])
        raise
