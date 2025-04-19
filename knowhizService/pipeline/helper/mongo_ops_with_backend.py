#!/usr/bin/env python3
# coding: utf-8

# from pipeline.dao_mongo import DaoMongodb
from pipeline.config.config import Config
from bson.objectid import ObjectId
import logging
import requests
import os

logger = logging.getLogger("kzpipeline.helper.mongo_ops_with_backend")
class MongoOps(object):
    def __init__(self):
        # self.db = DaoMongodb()
        self.config = Config()
        self.backend_url = self.config.get_backend_url()
        self.discord_bot_url = self.config.get_discord_bot_url()

    def discord_bot_send_message(self, channel_id:int, message:str, mention=""):
        json = {"channel_id": channel_id, "message": message, "mention": mention}
        response = requests.post(f"{self.discord_bot_url}/v1/discord/send", json=json)
        data = {}
        if response.status_code == 200:
            logger.info(f"Data retrieved successfully (discord bot send message): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (discord bot send message): {response.status_code}")

    def mongo_write_discord_zeroshot_course(self, user_id, course_description):
        if isinstance(user_id, ObjectId):
            user_id = str(user_id)
        payload = {"prompt": course_description}
        response = requests.post(f"{self.backend_url}/api/discord/create/zero-shot/{user_id}", json=payload)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (write_pipeline): {data}")
            return {"status": "SUCCESS", "id": data["id"]}

        elif response.status_code == 409:
            logger.info("Duplicate course detected, stopping process.")
            data = response.json()
            return {"status": "CONFLICT", "id": data["id"]}

        else:
            logger.info(f"Failed to retrieve data (write_pipeline): {response.status_code}")
            return {"status": "ERROR", "message": f"Failed with status {response.status_code}"}

    def mongo_write_pipeline(self, user_id, course_id, zeroshot=False, course_description="", full_material_url="", full_supplementary_material_urls=[]):
        if isinstance(user_id, ObjectId):
            user_id = str(user_id)
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        json = {"userId": user_id, "courseId": course_id, "isZeroshot": zeroshot, "materialUrl": full_material_url, "supplementaryUrls": full_supplementary_material_urls, "courseDescription": course_description, "status": "STARTED"}
        response = requests.post(self.backend_url + "/api/pipelineBackfiller/create", json=json)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (write_pipeline): {data}")
        else:
            logger.info(f"Failed to retrieve data (write_pipeline): {response.status_code}")
        return data["id"]

    def mongo_update_pipeline_status(self, pipeline_id, generate_hash, status, reason):
        if isinstance(pipeline_id, ObjectId):
            pipeline_id = str(pipeline_id)
        pipeline_status_update_url = f"{self.backend_url}/api/pipelineBackfiller/{pipeline_id}/status"
        new_status = {"generateHash": generate_hash, "newStatusString": status, "newFailReasonString": reason}
        response = requests.put(pipeline_status_update_url, params=new_status)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (update_pipeline_status): {data}")
        else:
            logger.info(f"Failed to retrieve data (update_pipeline_status): {response.status_code}")

    def mongo_update_course_shareable(self, course_id, is_shareable:bool):
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        json_data = { "isShareable": is_shareable }
        course_shareable_update_url = f"{self.backend_url}/api/pipelineBackfiller/course/{course_id}/shareable"
        response = requests.put(course_shareable_update_url, json=json_data)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (update_course_shareable): {data}")
        else:
            logger.info(f"Failed to retrieve data (update_course_shareable): {response.status_code}")

    def mongo_update_course_uploaded_docs(self, course_id, uploaded_docs:list):
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        course_uploaded_docs_update_url = f"{self.backend_url}/api/pipelineBackfiller/course/setUploadedDocs/{course_id}"
        response = requests.put(course_uploaded_docs_update_url, json=uploaded_docs)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (update_course_uploaded_docs): {data}")
        else:
            logger.info(f"Failed to retrieve data (update_course_uploaded_docs): {response.status_code}")

    def mongo_update_course_name(self, course_id, name, title, description):
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        json = {"courseName": name, "title": title, "description": description}
        course_name_update_url = f"{self.backend_url}/api/pipelineBackfiller/course/{course_id}/name"
        response = requests.put(course_name_update_url, json=json)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (update_course_name): {data}")
        else:
            logger.info(f"Failed to retrieve data (update_course_name): {response.status_code}")

    def mongo_update_course_status(self, course_id, status, course_generation_type, viewed_count, total_count):
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        payload = {"status": status, "courseGenerationType": course_generation_type, "viewedCount": viewed_count, "totalCount": total_count}
        course_status_update_url = f"{self.backend_url}/api/pipelineBackfiller/course/{course_id}/status"
        response = requests.put(course_status_update_url, json=payload)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (update_course_status): {data}")
        else:
            logger.info(f"Failed to retrieve data (update_course_status): {response.status_code}")

    def mongo_write_sections(self, course_id, sections):
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        set_sections_url = f"{self.backend_url}/api/pipelineBackfiller/course/{course_id}/setSections"
        response = requests.put(set_sections_url, json=sections)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (write_sections): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (write_sections): {response.status_code}")
        return data

    def mongo_write_flashcard(self, user_id, course_id, section_id, question, answer):
        if isinstance(user_id, ObjectId):
            user_id = str(user_id)
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        # if isinstance(chapter_id, ObjectId):
        #     chapter_id = str(chapter_id)
        flashcard_data = {"question": question, "mdQuestion": question, "answer": answer["definition"], "expandedAnswer": {"mdContent": answer["expansion"]}, "status": "SHOW", "courseId": course_id, "sectionId": section_id, "userId": user_id}
        response = requests.post(self.backend_url + "/api/pipelineBackfiller/flashcards/create", json=flashcard_data)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (write_flashcard): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (write_flashcard): {response.status_code}")
        # return data["id"]
        return data

    def mongo_write_flashcards(self, flashcards):
        response = requests.post(self.backend_url + "/api/pipelineBackfiller/flashcards/multipleCreate", json=flashcards)
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (write_flashcards): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (write_flashcards): {response.status_code}")
        return data

    # Save quiz of multi choice questions
    def mongo_write_quizzes(self, quizzes):
        response = requests.post(self.backend_url + "/api/pipelineBackfiller/quiz/saveMultiChoiceQuestions", json=quizzes)
        # data = {}
        data = ""
        if response.status_code == 200:
            data = response.text
            logger.info(f"Data retrieved successfully (write_quizzes): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (write_quizzes): {response.status_code}")
        return data

    # Save test of multi choice questions
    def mongo_write_mcq_tests(self, tests):
        response = requests.post(self.backend_url + "/api/pipelineBackfiller/test/saveMultiChoiceQuestions", json=tests)
        # data = {}
        data = ""
        if response.status_code == 200:
            data = response.text
            logger.info(f"Data retrieved successfully (write_mcq_tests): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (write_mcq_tests): {response.status_code}")
        return data

    # Save test of open ended questions
    def mongo_write_saq_tests(self, tests):
        response = requests.post(self.backend_url + "/api/pipelineBackfiller/test/saveOpenEndedQuestions", json=tests)
        # data = {}
        data = ""
        if response.status_code == 200:
            data = response.text
            logger.info(f"Data retrieved successfully (write_saq_tests): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (write_saq_tests): {response.status_code}")
        return data

    def mongo_get_course_by_course_id(self, course_id):
        response = requests.get(f"{self.backend_url}/api/pipelineBackfiller/byCourseId/{course_id}")
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (get_course): {data}")
        else:
            logger.info(f"Failed to retrieve data (get_course): {response.status_code}")
        return data

    def mongo_get_flashcards_by_course_id(self, course_id):
        response = requests.get(f"{self.backend_url}/api/pipelineBackfiller/flashcard/byCourseId/{course_id}")
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (get_flashcards): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (get_flashcards): {response.status_code}")
        return data

    def mongo_get_flashcard_by_flashcard_id(self, flashcard_id):
        response = requests.get(f"{self.backend_url}/api/pipelineBackfiller/flashcard/byFlashcardId/{flashcard_id}")
        data = {}
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data retrieved successfully (get_flashcard): {response.status_code}")
        else:
            logger.info(f"Failed to retrieve data (get_flashcard): {response.status_code}")
        return data

    def get_flashcard_id_by_question(self, question:str, flashcards):
        for flashcard in flashcards:
            if flashcard["question"] == question:
                return flashcard["id"]

    def mongo_write_sections_and_flashcards(self, user_id, course_id, chapter_index, flashcards, chapters:list=[]):
        # logger.info(f"user_id: {user_id}")
        # logger.info(f"course_id: {course_id}")
        # logger.info(f"chapter_index: {chapter_index}")
        # logger.info(f"flashcards: {flashcards}")
        # logger.info(f"chapters: {chapters}")
        sections_data = []
        flashcards_data = []
        for index, chapter in enumerate(chapters):
            sections_data.append({"index": index, "order": index, "title": str(chapter), "status": "NORMAL"})
            # if index + 1 > max_chapter_count:
            #     break
        for question, answer in flashcards.items():
            flashcards_data.append({"question": question, "mdQuestion": question, "answer": answer["definition"], "expandedAnswer": {"mdContent": answer["expansion"]}, "sectionId": chapter_index, "status": "SHOW", "courseId": course_id, "userId": user_id})

        # Update sections
        self.mongo_write_sections(course_id, sections_data)
        # Create flashcards
        updated_flashcards = self.mongo_write_flashcards(flashcards_data)

    def mongo_write_course(self, user_id, course_id, pipeline_id, chapters:list=[], chapter_flashcards:list=[], quizzes:list=[], tests_mcq:list=[], tests_saq:list=[], zeroshot:bool=False, course_description:str="", material_url:str="", supplementary_material_urls:list=[], course_name:str="", generate_hash:str="", is_shareable:bool=False, course_generation_type="ZEROSHOT"):
        if isinstance(user_id, ObjectId):
            user_id = str(user_id)
        if isinstance(course_id, ObjectId):
            course_id = str(course_id)
        if isinstance(pipeline_id, ObjectId):
            pipeline_id = str(pipeline_id)

        # Create flashcards and sections
        # sections_data = []
        # flashcards_data = []
        quizzes_data = []
        tests_mcq_data = []
        tests_saq_data = []
        total_flashcard_count = 0
        max_chapter_count = min(len(chapter_flashcards), len(chapters))
        logger.info(f"len(chapter_flashcards): {len(chapter_flashcards)}, len(chapters): {len(chapters)}, len(quizzes): {len(quizzes)}")
        for index, chapter in enumerate(chapters):
            if index + 1 > max_chapter_count:
                break
            total_flashcard_count += len(chapter_flashcards[index])
        #     sections_data.append({"index": index, "order": index, "title": str(chapter), "status": "NORMAL"})
        #     for question, answer in chapter_flashcards[index].items():
        #         flashcards_data.append({"question": question, "mdQuestion": question, "answer": answer["definition"], "expandedAnswer": {"mdContent": answer["expansion"]}, "sectionId": index, "status": "SHOW", "courseId": course_id, "userId": user_id})

        # # Update sections
        # self.mongo_write_sections(course_id, sections_data)
        # # Create flashcards
        # updated_flashcards = self.mongo_write_flashcards(flashcards_data)
        if course_name:
            self.mongo_update_course_name(course_id, course_name, course_name, course_description)
        if is_shareable:
            self.mongo_update_course_shareable(course_id, is_shareable)

        # Update quizzes
        # # FIXME
        # with open(".test_updated_flashcards.json", 'w') as file:
        #     json.dump(updated_flashcards, file, indent=2)
        # with open(".test_quizzes0.json", 'w') as file:
        #     json.dump(quizzes, file, indent=2)

        if not (course_generation_type == "ANKI"): # ANKI will not generate quizzes and tests
            updated_flashcards = self.mongo_get_flashcards_by_course_id(course_id)
            max_chapter_count = min(len(chapter_flashcards), len(chapters), len(quizzes))
            for index, chapter in enumerate(chapters):
                if index + 1 > max_chapter_count:
                    break
                for quiz_name, quiz in quizzes[index].items():
                    if "question" in quiz and "choices" in quiz and "correct_answer" in quiz:
                        flashcard_id = self.get_flashcard_id_by_question(quiz_name, updated_flashcards)
                        question_choice = []
                        for choice, choice_desciption in quiz["choices"].items():
                            isCorrect = False
                            if choice == quiz["correct_answer"]:
                                isCorrect = True
                            choice = choice.upper()[0]
                            choice_index = ord(choice) - ord('A')
                            question_choice.append({"index": choice_index, "description": choice_desciption, "isCorrect": isCorrect})

                    quizzes_data.append({"courseId": course_id, "sectionId": index, "flashcardId": flashcard_id, "question": quiz["question"], "choices": question_choice, "explanation": "", "fromExpandedContent": True})

            # # FIXME
            # with open(".test_quizzes.json", 'w') as file:
            #     json.dump(quizzes_data, file, indent=2)

            self.mongo_write_quizzes(quizzes_data)

        # Update tests
        # # FIXME
        # with open(".test_updated_flashcards.json", 'w') as file:
        #     json.dump(updated_flashcards, file, indent=2)
        # with open(".test_test_mcq0.json", 'w') as file:
        #     json.dump(tests_mcq, file, indent=2)
        # with open(".test_test_saq0.json", 'w') as file:
        #     json.dump(tests_saq, file, indent=2)

        if not (course_generation_type == "ANKI"): # ANKI will not generate quizzes and tests
            max_chapter_count = min(len(chapters), len(tests_mcq))
            # process test mcq data only if tests_mcq is a list
            if isinstance(tests_mcq, list):
                for index, chapter in enumerate(chapters):
                    if index + 1 > max_chapter_count:
                        break
                    test_list_item = tests_mcq[index]
                    # Continue if test_list_item is not a list
                    if not isinstance(test_list_item, list):
                        continue

                    for test_item_index, test_item in enumerate(test_list_item):
                        question_choice = []
                        for choice, choice_desciption in test_item["choices"].items():
                            isCorrect = False
                            if choice == test_item["correct_answer"]:
                                isCorrect = True
                            choice = choice.upper()[0]
                            choice_index = ord(choice) - ord('A')
                            question_choice.append({"index": choice_index, "description": choice_desciption, "isCorrect": isCorrect})

                        tests_mcq_data.append({"courseId": course_id, "question": test_item["question"], "choices": question_choice, "explanation": "", "fromExpandedContent": True})

                self.mongo_write_mcq_tests(tests_mcq_data)

            # process test saq data only if tests_saq is a list
            if isinstance(tests_saq, list):
                max_chapter_count = min(len(chapters), len(tests_saq))
                for index, chapter in enumerate(chapters):
                    if index + 1 > max_chapter_count:
                        break
                    test_list_item = tests_saq[index]
                    # Continue if test_list_item is not a list
                    if not isinstance(test_list_item, list):
                        continue

                    for test_item_index, test_item in enumerate(test_list_item):
                        if "answer" in test_item:
                            tests_saq_data.append({"courseId": course_id, "question": test_item["question"], "answer": test_item["answer"], "fromExpandedContent": True})
                        else:
                            tests_saq_data.append({"courseId": course_id, "question": test_item["question"], "answer": "", "fromExpandedContent": True})

                self.mongo_write_saq_tests(tests_saq_data)

        # # FIXME
        # with open(".test_mcq_tests.json", 'w') as file:
        #     json.dump(tests_mcq_data, file, indent=2)
        # with open(".test_saq_tests.json", 'w') as file:
        #     json.dump(tests_saq_data, file, indent=2)

        # Update course uploaded_docs
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

        self.mongo_update_course_status(course_id, "READY", course_generation_type, 0, total_flashcard_count)
        self.mongo_update_pipeline_status(pipeline_id, generate_hash, "DONE", "PASS")
        logger.info(f"mongo_write_course session done, user_id: {user_id}, course_id: {course_id}, pipeline_id: {pipeline_id}, generate_hash: {generate_hash}, course_generation_type: {course_generation_type}")

if __name__ == "__main__":
    ops = MongoOps()

    print("write course status")
    ops.mongo_update_course_status("65034a0dc47a307eaec0525a", "READY", "ZEROSHOT", 0, 100)
    print("write course status done")
    # ops.mongo_update_pipeline_status("6617fd8d14bc4f4cbf083000", "6617fd8d14bc4f4cbf083000", "DONE", "PASS")
