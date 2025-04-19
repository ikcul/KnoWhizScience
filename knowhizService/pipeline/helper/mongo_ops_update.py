#!/usr/bin/env python3
# coding: utf-8
import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pipeline.dao_mongo import DaoMongodb
from pipeline.config.config import Config
from bson.objectid import ObjectId
import requests
from datetime import datetime, timezone

# Configure logging
logger = logging.getLogger("kzpipeline.helper.mongo_ops")

class MongoOps(object):
    def __init__(self):
        self.db = DaoMongodb()
        self.config = Config()
        self.discord_bot_url = self.config.get_discord_bot_url()

    def discord_bot_send_message(self, channel_id:int, message:str, mention=""):
        json = {"channel_id": channel_id, "message": message, "mention": mention}
        response = requests.post(f"{self.discord_bot_url}/v1/discord/send", json=json)
        if response.status_code == 200:
            logger.info(f"Data retrieved successfully (discord bot send message): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (discord bot send message): {response.status_code}")

    def mongo_write_discord_zeroshot_course(self, user_id):
        try:
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)

            current_time = datetime.now(tz=timezone.utc)
            course_data = {
                "userId": user_id,
                "type": "ZERO_SHOT",
                "status": "PROCESSING",
                "statusTimeline": [
                    {
                        "effectiveTime": current_time,
                        "status": "PROCESSING"
                    }
                ],
                "creationTime": current_time,
                "lastUpdatedTime": current_time,
            }

            course_id = str(self.db.add_update_course(course_data))
            logger.info(f"Zero-shot course created successfully with ID: {course_id}")

            return course_id

        except Exception as e:
            logger.exception(f"Failed to create zero-shot course: {str(e)}")
            return None

    def mongo_write_pipeline(self, user_id, course_id, zeroshot=False, course_description="", full_material_url="", full_supplementary_material_urls=[]):
        try:
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            if isinstance(course_id, str):
                course_id = ObjectId(course_id)

            pipeline_data = {
                "userId": user_id,
                "courseId": course_id,
                "isZeroshot": zeroshot,
                "materialUrl": full_material_url,
                "supplementaryUrls": full_supplementary_material_urls,
                "courseDescription": course_description,
                "status": "STARTED",
                "creationTime": datetime.now(tz=timezone.utc),
                "lastUpdatedTime": datetime.now(tz=timezone.utc)
            }

            pipeline_id = str(self.db.add_update_pipeline(pipeline_data))
            logger.info(f"Pipeline created successfully with ID: {pipeline_id}")
            return pipeline_id

        except Exception as e:
            logger.exception(f"Failed to create pipeline: {str(e)}")
            return None

    def mongo_update_pipeline_status(self, pipeline_id, generate_hash, status, reason):
        try:
            if isinstance(pipeline_id, str):
                pipeline_id = ObjectId(pipeline_id)
            pipeline_data = self.db.get_pipeline(pipeline_id)

            if pipeline_data:
                update_data = {
                    "generateHash": generate_hash,
                    "status": status,
                    "failReason": reason,
                    "lastUpdatedTime": datetime.now(tz=timezone.utc)
                }
                pipeline_data.update(update_data)

                self.db.add_update_pipeline(pipeline_data)
                logger.info(f"Pipeline status updated successfully for pipeline ID: {pipeline_id}")

            else:
                logger.info(f"No pipeline found with ID: {pipeline_id}")
                return None

        except Exception as e:
            logger.exception(f"Failed to update pipeline status: {str(e)}")
            return None

    def mongo_update_course_shareable(self, course_id, is_shareable: bool):
        try:
            logger.info(f"Entering mongo_update_course_shareable with course_id: {course_id} and is_shareable: {is_shareable}")

            if isinstance(course_id, str):
                course_id = ObjectId(course_id)
            course_data = self.db.get_course(course_id)
            logger.info(f"Fetched course_data: {course_data}")

            if course_data:
                course_data["isShareable"] = is_shareable
                result = self.db.add_update_course(course_data)
                logger.info(f"Updated course_data with isShareable. Result: {result}")
            else:
                logger.info(f"No course found with course_id: {course_id}")

        except Exception as e:
            logger.exception(f"Failed to update course shareable status: {str(e)}")
            return None

    def mongo_update_course_uploaded_docs(self, course_id, uploaded_docs: list):
        try:
            logger.info(f"Entering mongo_update_course_uploaded_docs with course_id: {course_id} and uploaded_docs: {uploaded_docs}")

            if isinstance(course_id, str):
                course_id = ObjectId(course_id)

            course_data = self.db.get_course(course_id)
            logger.info(f"Fetched course_data: {course_data}")

            if course_data:
                course_data["uploadDocs"] = uploaded_docs
                result = self.db.add_update_course(course_data)
                logger.info(f"Updated course_data with uploadDocs. Result: {result}")
            else:
                logger.info(f"No course found with course_id: {course_id}")

        except Exception as e:
            logger.exception(f"Failed to update course uploaded docs: {str(e)}")
            return None

    def mongo_update_course_name(self, course_id, name):
        try:
            logger.info(f"Entering mongo_update_course_name with course_id: {course_id} and name: {name}")

            if isinstance(course_id, str):
                course_id = ObjectId(course_id)

            course_data = self.db.get_course(course_id)
            logger.info(f"Fetched course_data: {course_data}")

            if course_data:
                course_data["courseName"] = name
                logger.info(f"course_data after adding/updating courseName: {course_data}")
                result = self.db.add_update_course(course_data)
                logger.info(f"Updated course_data with courseName. Result: {result}")
                return result
            else:
                logger.info(f"No course found with course_id: {course_id}")
                return None

        except Exception as e:
            logger.exception(f"Failed to update course name: {str(e)}")
            return None

    def mongo_update_course_status(self, course_id, status):
        try:
            logger.info(f"Entering mongo_update_course_status with course_id: {course_id} and status: {status}")

            if isinstance(course_id, str):
                course_id = ObjectId(course_id)

            course_data = self.db.get_course(course_id)
            logger.info(f"Fetched course_data: {course_data}")

            if course_data:
                course_data["status"] = status
                result = self.db.add_update_course(course_data)
                logger.info(f"Updated course_data with status. Result: {result}")

                if status == "READY":
                    logger.info(f"Course {course_id} is now READY.")

                return result
            else:
                logger.info(f"No course found with course_id: {course_id}")
                return None

        except Exception as e:
            logger.exception(f"Failed to update course status: {str(e)}")
            return None

    def mongo_write_sections(self, course_id, sections):
        logger.info(f"Entering mongo_write_sections with course_id: {course_id} and sections: {sections}")

        if isinstance(course_id, str):
            course_id = ObjectId(course_id)

        course_data = self.db.get_course(course_id)
        logger.info(f"Fetched course_data: {course_data}")

        if course_data:
            course_data["sections"] = sections
            result = self.db.add_update_course(course_data)
            logger.info(f"Updated course_data with sections. Result: {result}")
            return result
        else:
            logger.info(f"No course found with course_id: {course_id}")
            return None

    def mongo_write_flashcard(self, user_id, course_id, section_id, question, answer):
        logger.info(f"Entering mongo_write_flashcard with user_id: {user_id}, course_id: {course_id}, section_id: {section_id}, question: {question}")

        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        if isinstance(course_id, str):
            course_id = ObjectId(course_id)

        if not isinstance(answer, dict) or "definition" not in answer or "expansion" not in answer:
            logger.info(f"Invalid answer format: {answer}. Expected a dictionary with 'definition' and 'expansion' keys.")
            return None

        flashcard_data = {
            "question": question,
            "mdQuestion": question,
            "answer": answer["definition"],
            "expandedAnswer": {"mdContent": answer["expansion"]},
            "status": "SHOW",
            "courseId": course_id,
            "sectionId": section_id,
            "userId": user_id,
            "creationTime": datetime.now(tz=timezone.utc),
            "lastUpdatedTime": datetime.now(tz=timezone.utc)
        }

        try:
            flashcard_id = self.db.add_update_flashcard(flashcard_data)
            # logger.info(f"Flashcard created successfully with ID: {flashcard_id}")
            return str(flashcard_id)
        except Exception as e:
            logger.exception(f"Failed to create flashcard: {str(e)}")
            return None

    def mongo_write_flashcards(self, flashcards):
        updated_flashcards = []

        for flashcard in flashcards:
            try:
                # logger.info(f"Processing flashcard: {flashcard}")

                if isinstance(flashcard["userId"], str):
                    flashcard["userId"] = ObjectId(flashcard["userId"])
                if isinstance(flashcard["courseId"], str):
                    flashcard["courseId"] = ObjectId(flashcard["courseId"])

                flashcard_id = self.db.add_update_flashcard(flashcard)
                # logger.info(f"Flashcard created successfully with ID: {flashcard_id}")
                flashcard["_id"] = str(flashcard_id)
                updated_flashcards.append(flashcard)

            except Exception as e:
                logger.exception(f"Failed to create flashcard: {flashcard}. Exception: {str(e)}")
                updated_flashcards.append(None)

        logger.info(f"All flashcards processed, updated flashcards.")
        # logger.info(f"All flashcards processed, updated flashcards: {updated_flashcards}")

        return updated_flashcards

    def mongo_write_quizzes(self, quizzes):
        quiz_ids = []

        for quiz in quizzes:
            try:
                # logger.info(f"Processing quiz: {quiz}")

                quiz_id = self.db.add_update_quiz(quiz)

                if quiz_id:
                    quiz_ids.append(quiz_id)
                else:
                    logger.info(f"Failed to create quiz: {quiz}")
                    quiz_ids.append(None)

            except Exception as e:
                logger.exception(f"Error processing quiz: {quiz}. Exception: {str(e)}")
                quiz_ids.append(None)

        # logger.info(f"All quizzes processed, IDs: {quiz_ids}")
        logger.info(f"All quizzes processed.")

        return quiz_ids

    def mongo_write_mcq_tests(self, tests):
        test_ids = []

        for test in tests:
            try:
                # logger.info(f"Processing test: {test}")

                test_id = self.db.add_update_mcq_test(test)

                if test_id:
                    test_ids.append(test_id)
                else:
                    logger.info(f"Failed to create test: {test}")
                    test_ids.append(None)

            except Exception as e:
                logger.exception(f"Error processing test: {test}. Exception: {str(e)}")
                test_ids.append(None)

        # logger.info(f"All tests processed, IDs: {test_ids}")
        logger.info(f"All tests processed.")

        return test_ids

    def mongo_write_saq_tests(self, tests):
        test_ids = []

        for test in tests:
            try:
                # logger.info(f"Processing open-ended test: {test}")

                test_id = self.db.add_update_saq_test(test)

                if test_id:
                    test_ids.append(test_id)
                else:
                    logger.info(f"Failed to create open-ended test: {test}")
                    test_ids.append(None)

            except Exception as e:
                logger.exception(f"Error processing open-ended test: {test}. Exception: {str(e)}")
                test_ids.append(None)

        # logger.info(f"All open-ended tests processed, IDs: {test_ids}")
        logger.info(f"All open-ended tests processed.")

        return test_ids

    def mongo_get_course_by_course_id(self, course_id):
        logger.info(f"Entering mongo_get_course_by_course_id with course_id: {course_id}")
        if isinstance(course_id, str):
            course_id = ObjectId(course_id)

        try:
            course_data = self.db.get_course(course_id)
            if course_data:
                logger.info(f"Course data retrieved successfully: {course_data}")
            else:
                logger.info(f"No course found with course_id: {course_id}")

            return course_data

        except Exception as e:
            logger.exception(f"Error retrieving course data for course_id: {course_id}. Exception: {str(e)}")
            return None

    def get_flashcard_id_by_question(self, question: str, flashcards):
        try:
            # logger.info(f"Searching for flashcard with question: '{question}'")
            for flashcard in flashcards:
                # logger.info(f"Checking flashcard question: {flashcard}")
                if "question" in flashcard and "_id" in flashcard:
                    if flashcard["question"] == question:
                        # logger.info(f"Match found for question '{question}'. Flashcard ID: {flashcard['_id']}")
                        return flashcard["_id"]
                else:
                    missing_keys = []
                    if "question" not in flashcard:
                        missing_keys.append("question")
                    if "_id" not in flashcard:
                        missing_keys.append("_id")
                    # logger.info(f"Flashcard missing keys {missing_keys}: {flashcard}")
            logger.info(f"No match found for question: '{question}'")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error occurred: {str(e)}")
            return None

    def mongo_write_course(self, user_id, course_id, pipeline_id, chapters:list=[], chapter_flashcards:list=[], quizzes:list=[], tests_mcq:list=[], tests_saq:list=[], zeroshot:bool=False, course_description:str="", material_url:str="", supplementary_material_urls:list=[], course_name:str="", generate_hash:str="", is_shareable:bool=False, course_generation_type="ZEROSHOT"):
        if isinstance(user_id, ObjectId):
            user_id = str(user_id)
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        if isinstance(pipeline_id, ObjectId):
            pipeline_id = str(pipeline_id)

        sections_data = []
        flashcards_data = []
        quizzes_data = []
        tests_mcq_data = []
        tests_saq_data = []
        max_chapter_count = min(len(chapter_flashcards), len(chapters))
        logger.info(f"len(chapter_flashcards): {len(chapter_flashcards)}, len(chapters): {len(chapters)}, len(quizzes): {len(quizzes)}")
        for index, chapter in enumerate(chapters):
            sections_data.append({"index": index, "order": index, "title": str(chapter), "status": "NORMAL"})
            if index + 1 > max_chapter_count:
                break
            for question, answer in chapter_flashcards[index].items():
                flashcards_data.append({"question": question, "mdQuestion": question, "answer": answer["definition"], "expandedAnswer": {"mdContent": answer["expansion"]}, "sectionId": index, "status": "SHOW", "courseId": course_id, "userId": user_id})
        # logger.info(f"Constructed flashcard_data: {flashcards_data}")
        self.mongo_write_sections(course_id, sections_data)
        updated_flashcards = self.mongo_write_flashcards(flashcards_data)
        # logger.info(f"Updated flashcards: {updated_flashcards}")
        if course_name:
            logger.info(f"Updating course name to: {course_name}")
            self.mongo_update_course_name(course_id, course_name)
        else:
            logger.info("No course name provided; skipping course name update.")
        if is_shareable:
            logger.info("Setting course as shareable.")
            self.mongo_update_course_shareable(course_id, is_shareable)
        else:
            logger.info("Course is not shareable; skipping shareable update.")

        logger.info(f"Processing quizzes, len(quizzes): {len(quizzes)}")
        max_chapter_count = min(len(chapter_flashcards), len(chapters), len(quizzes))
        for index, chapter in enumerate(chapters):
            if index + 1 > max_chapter_count:
                break
            logger.info(f"Processing chapter {index} for quizzes.")
            for quiz_name, quiz in quizzes[index].items():
                # logger.info(f"Processing quiz: {quiz_name}")
                if "question" in quiz and "choices" in quiz and "correct_answer" in quiz:
                    flashcard_id = self.get_flashcard_id_by_question(quiz_name, updated_flashcards)
                    if flashcard_id is None:
                        logger.info(f"Flashcard ID not found for quiz: {quiz_name}")
                    else:
                        # logger.info(f"Flashcard ID found: {flashcard_id} for quiz: {quiz_name}")
                        pass
                    question_choice = []
                    for choice, choice_desciption in quiz["choices"].items():
                        isCorrect = False
                        if choice == quiz["correct_answer"]:
                            isCorrect = True
                        choice = choice.upper()[0]
                        choice_index = ord(choice) - ord('A')
                        question_choice.append({"index": choice_index, "description": choice_desciption, "isCorrect": isCorrect})

                quizzes_data.append({"courseId": course_id, "sectionId": index, "flashcardId": flashcard_id, "question": quiz["question"], "choices": question_choice, "explanation": "", "fromExpandedContent": True})

        self.mongo_write_quizzes(quizzes_data)

        max_chapter_count = min(len(chapters), len(tests_mcq))
        if isinstance(tests_mcq, list):
            for index, chapter in enumerate(chapters):
                if index + 1 > max_chapter_count:
                    break
                test_list_item = tests_mcq[index]
                if not isinstance(test_list_item, list):
                    continue

                for test_item_index, test_item in enumerate(test_list_item):
                    question_choice = []
                    for choice, choice_desciption in test_item["choices"].items():
                        isCorrect = False
                        if choice == quiz["correct_answer"]:
                            isCorrect = True
                        choice = choice.upper()[0]
                        choice_index = ord(choice) - ord('A')
                        question_choice.append({"index": choice_index, "description": choice_desciption, "isCorrect": isCorrect})

                    tests_mcq_data.append({"courseId": course_id, "question": test_item["question"], "choices": question_choice, "explanation": "", "fromExpandedContent": True})

            self.mongo_write_mcq_tests(tests_mcq_data)

        if isinstance(tests_saq, list):
            max_chapter_count = min(len(chapters), len(tests_saq))
            for index, chapter in enumerate(chapters):
                if index + 1 > max_chapter_count:
                    break
                test_list_item = tests_saq[index]
                if not isinstance(test_list_item, list):
                    continue

                for test_item_index, test_item in enumerate(test_list_item):
                    if "answer" in test_item:
                        tests_saq_data.append({"courseId": course_id, "question": test_item["question"], "answer": test_item["answer"], "fromExpandedContent": True})
                    else:
                        tests_saq_data.append({"courseId": course_id, "question": test_item["question"], "answer": "", "fromExpandedContent": True})

            self.mongo_write_saq_tests(tests_saq_data)

        if material_url:
            uploaded_docs = []
            prefix, filename = os.path.split(material_url)
            uploaded_docs.append({"fileName": filename, "url": material_url, "isPrimary": True})
            if supplementary_material_urls:
                for index, supplementary_material_url in enumerate(supplementary_material_urls):
                    if isinstance(supplementary_material_url, str):
                        prefix, filename = os.path.split(material_url)
                        uploaded_docs.append({"fileName": filename, "url": supplementary_material_url, "isPrimary": False})
            self.mongo_update_course_uploaded_docs(course_id, uploaded_docs)

        self.mongo_update_course_status(course_id, "READY", course_generation_type)
        self.mongo_update_pipeline_status(pipeline_id, generate_hash, "DONE", "PASS")
        logger.info(f"mongo_write_course session done, user_id: {user_id}, course_id: {course_id}, pipeline_id: {pipeline_id}, generate_hash: {generate_hash}, course_generation_type: {course_generation_type}")


if __name__ == "__main__":
    ops = MongoOps()

    try:
        # Test: discord_bot_send_message
        ops.discord_bot_send_message(123456789, "Hello, this is a test message.", "UserMention")
        logger.info("Pass: discord_bot_send_message")
    except Exception as e:
        logger.exception(f"Fail: discord_bot_send_message - {e}")

    try:
        # Test: mongo_write_discord_zeroshot_course
        zeroshot_course_id = ops.mongo_write_discord_zeroshot_course("507f1f77bcf86cd799439011")
        logger.info(f"Zeroshot Course ID: {zeroshot_course_id}")
        logger.info("Pass: mongo_write_discord_zeroshot_course")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_discord_zeroshot_course - {e}")

    try:
        # Test: mongo_write_pipeline
        pipeline_id = ops.mongo_write_pipeline(
            user_id="507f1f77bcf86cd799439011",
            course_id="507f191e810c19729de860ea",
            zeroshot=True,
            course_description="Sample Course Description",
            full_material_url="http://example.com/material",
            full_supplementary_material_urls=["http://example.com/supplementary1", "http://example.com/supplementary2"]
        )
        logger.info(f"Pipeline ID: {pipeline_id}")
        logger.info("Pass: mongo_write_pipeline")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_pipeline - {e}")

    try:
        # Test: mongo_update_pipeline_status
        ops.mongo_update_pipeline_status(
            pipeline_id=pipeline_id,
            generate_hash="some-hash-value",
            status="DONE",
            reason="PASS"
        )
        logger.info("Pass: mongo_update_pipeline_status")
    except Exception as e:
        logger.exception(f"Fail: mongo_update_pipeline_status - {e}")

    try:
        # Test: mongo_update_course_shareable
        ops.mongo_update_course_shareable(
            course_id="507f191e810c19729de860ea",
            is_shareable=True
        )
        logger.info("Pass: mongo_update_course_shareable")
    except Exception as e:
        logger.exception(f"Fail: mongo_update_course_shareable - {e}")

    try:
        # Test: mongo_update_course_uploaded_docs
        ops.mongo_update_course_uploaded_docs(
            course_id="507f191e810c19729de860ea",
            uploaded_docs=[{"fileName": "material.pdf", "url": "http://example.com/material.pdf", "isPrimary": True}]
        )
        logger.info("Pass: mongo_update_course_uploaded_docs")
    except Exception as e:
        logger.exception(f"Fail: mongo_update_course_uploaded_docs - {e}")

    try:
        # Test: mongo_update_course_name
        ops.mongo_update_course_name(
            course_id="507f191e810c19729de860ea",
            name="Updated Course Name"
        )
        logger.info("Pass: mongo_update_course_name")
    except Exception as e:
        logger.exception(f"Fail: mongo_update_course_name - {e}")

    try:
        # Test: mongo_update_course_status
        ops.mongo_update_course_status(
            course_id="507f191e810c19729de860ea",
            status="READY",
            course_generation_type="ZEROSHOT"
        )
        logger.info("Pass: mongo_update_course_status")
    except Exception as e:
        logger.exception(f"Fail: mongo_update_course_status - {e}")

    try:
        # Test: mongo_write_sections
        ops.mongo_write_sections(
            course_id="507f191e810c19729de860ea",
            sections=[{"index": 0, "order": 0, "title": "Introduction", "status": "NORMAL"}, {"index": 1, "order": 1, "title": "Chapter 1", "status": "NORMAL"}]
        )
        logger.info("Pass: mongo_write_sections")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_sections - {e}")

    try:
        # Test: mongo_write_flashcard
        flashcard_id = ops.mongo_write_flashcard(
            user_id="507f1f77bcf86cd799439011",
            course_id="507f191e810c19729de860ea",
            section_id=0,
            question="What is MongoDB?",
            answer={"definition": "MongoDB is a NoSQL database.", "expansion": "It stores data in flexible, JSON-like documents."}
        )
        logger.info(f"Flashcard ID: {flashcard_id}")
        logger.info("Pass: mongo_write_flashcard")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_flashcard - {e}")

    try:
        # Test: mongo_write_flashcards
        flashcard_ids = ops.mongo_write_flashcards([
            {"userId": "507f1f77bcf86cd799439011", "courseId": "507f191e810c19729de860ea", "question": "What is MongoDB?", "answer": {"definition": "MongoDB is a NoSQL database.", "expansion": "It stores data in flexible, JSON-like documents."}},
            {"userId": "507f1f77bcf86cd799439011", "courseId": "507f191e810c19729de860ea", "question": "What is a Collection?", "answer": {"definition": "A collection is a group of MongoDB documents.", "expansion": "It is equivalent to a table in RDBMS."}}
        ])
        logger.info(f"Flashcard IDs: {flashcard_ids}")
        logger.info("Pass: mongo_write_flashcards")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_flashcards - {e}")

    try:
        # Test: mongo_write_quizzes
        quiz_ids = ops.mongo_write_quizzes([
            {"courseId": "507f191e810c19729de860ea", "sectionId": 0, "flashcardId": flashcard_id, "question": "Quiz question?", "choices": [{"index": 0, "description": "Choice A", "isCorrect": True}, {"index": 1, "description": "Choice B", "isCorrect": False}], "explanation": "Explanation text.", "fromExpandedContent": True}
        ])
        logger.info(f"Quiz IDs: {quiz_ids}")
        logger.info("Pass: mongo_write_quizzes")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_quizzes - {e}")

    try:
        # Test: mongo_write_mcq_tests
        mcq_test_ids = ops.mongo_write_mcq_tests([
            {"courseId": "507f191e810c19729de860ea", "question": "MCQ question?", "choices": [{"index": 0, "description": "Choice A", "isCorrect": True}, {"index": 1, "description": "Choice B", "isCorrect": False}], "explanation": "Explanation text.", "fromExpandedContent": True}
        ])
        logger.info(f"MCQ Test IDs: {mcq_test_ids}")
        logger.info("Pass: mongo_write_mcq_tests")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_mcq_tests - {e}")

    try:
        # Test: mongo_write_saq_tests
        saq_test_ids = ops.mongo_write_saq_tests([
            {"courseId": "507f191e810c19729de860ea", "question": "SAQ question?", "answer": "Short answer."}
        ])
        logger.info(f"SAQ Test IDs: {saq_test_ids}")
        logger.info("Pass: mongo_write_saq_tests")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_saq_tests - {e}")

    try:
        # Test: mongo_get_course_by_course_id
        course_data = ops.mongo_get_course_by_course_id("507f191e810c19729de860ea")
        logger.info(f"Course Data: {course_data}")
        logger.info("Pass: mongo_get_course_by_course_id")
    except Exception as e:
        logger.exception(f"Fail: mongo_get_course_by_course_id - {e}")

    try:
        # Test: mongo_write_course
        ops.mongo_write_course(
            user_id="507f1f77bcf86cd799439011",
            course_id="507f191e810c19729de860ea",
            pipeline_id=pipeline_id,
            chapters=["Introduction", "Chapter 1"],
            chapter_flashcards=[
                {
                    "What is MongoDB?": {
                        "definition": "MongoDB is a NoSQL database.",
                        "expansion": "It stores data in flexible, JSON-like documents."
                    }
                },
                {
                    "What is a Collection?": {
                        "definition": "A collection is a group of MongoDB documents.",
                        "expansion": "It is equivalent to a table in RDBMS."
                    }
                }
            ],
            quizzes=[
                {
                    "What is MongoDB?": {
                        "question": "What type of database is MongoDB?",
                        "choices": {
                            "A": "SQL database",
                            "B": "NoSQL database",
                            "C": "Graph database",
                            "D": "Relational database"
                        },
                        "correct_answer": "B"
                    }
                },
                {
                    "What is a Collection?": {
                        "question": "What is a MongoDB collection equivalent to in RDBMS?",
                        "choices": {
                            "A": "Table",
                            "B": "Row",
                            "C": "Column",
                            "D": "Database"
                        },
                        "correct_answer": "A"
                    }
                }
            ],
            tests_mcq=[
                [
                    {
                        "question": "Which of the following is true about MongoDB?",
                        "choices": {
                            "A": "It uses SQL.",
                            "B": "It stores data in tables.",
                            "C": "It is a NoSQL database.",
                            "D": "It requires a fixed schema."
                        },
                        "correct_answer": "C"
                    }
                ],
                [
                    {
                        "question": "Which statement is correct about MongoBB collections?",
                        "choices": {
                            "A": "They are like rows in RDBMS.",
                            "B": "They enforce schemas strictly.",
                            "C": "They can store JSON-like documents.",
                            "D": "They require primary keys."
                        },
                        "correct_answer": "C"
                    }
                ]
            ],
            tests_saq=[
                [
                    {
                        "question": "Explain the benefits of using MongoDB over traditional RDBMS.",
                        "answer": "MongoDB provides flexibility with schema-less data storage, scalability, and high performance for large-scale data."
                    }
                ],
                [
                    {
                        "question": "Describe the concept of a MongoDB collection.",
                        "answer": "A MongoDB collection is a group of documents, similar to a table in RDBMS, but without enforced schemas."
                    }
                ]
            ],
            course_name="Test Course with Quizzes and Tests",
            generate_hash="test-hash-value",
            course_generation_type="ZEROSHOT"
        )
        logger.info("Pass: mongo_write_course")
    except Exception as e:
        logger.exception(f"Fail: mongo_write_course - {e}")

    logger.info("All functions executed. Please check the database to verify the results.")
