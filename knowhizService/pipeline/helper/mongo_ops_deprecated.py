#!/usr/bin/env python3
# coding: utf-8

from pipeline.dao_mongo import DaoMongodb
from bson.objectid import ObjectId
import os
from datetime import datetime, timezone

class MongoOps(object):
    def __init__(self):
        self.db = DaoMongodb()

    def mongo_write_pipeline(self, user_id, course_id, full_material_url):
        iso_datetime = datetime.now(tz=timezone.utc)
        pipeline_id = str(self.db.add_update_pipeline({"userId": ObjectId(user_id), "materialUrl": full_material_url, "courseId": ObjectId(course_id), "status": "STARTED", "creationTime": iso_datetime, "lastUpdatedTime": iso_datetime}))
        return pipeline_id

    def mongo_write_flashcard(self, user_id, course_id, section_id, question, answer):
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        if isinstance(course_id, str):
            course_id = ObjectId(course_id)
        if isinstance(chapter_id, str):
            chapter_id = ObjectId(chapter_id)
        flashcard_data = {"question": question, "answer": answer["definition"], "expandedAnswer": {"expansion": answer["expansion"]}, "status": "SHOW", "courseId": course_id, "sectionId": section_id, "userId": user_id}
        return self.db.add_update_flashcard(flashcard_data)

    def mongo_write_course(self, user_id, course_id, material_url, pipeline_id, chapter_flashcards, course_generation_type="ZEROSHOT"):
        if isinstance(user_id, str):
            user_id = ObjectId(user_id)
        if isinstance(course_id, str):
            course_id = ObjectId(course_id)
        if isinstance(pipeline_id, str):
            pipeline_id = ObjectId(pipeline_id)
        # chapter_list = []

        # Update mongodb with transaction
        with self.db.get_client().start_session() as session:
            with session.start_transaction():
                # Create flashcards
                for key, value in chapter_flashcards.items():
                    # chapter = {"_id": ObjectId(), "chapterName": "chapter " + str(key), "description": "", "courseId": course_id}
                    # chapter_id = self.db.add_update_chapter(chapter)
                    # chapter_list.append(chapter)
                    for question, answer in value.items():
                        self.mongo_write_flashcard(user_id, course_id, question, answer)

                course_data = self.db.get_course(course_id)
                if (not course_data) or (not isinstance(course_data, dict)):
                    # If course data not exist
                    course_data = {}

                # Update course
                course_data["_id"] = course_id
                if "courseName" not in course_data:
                    course_data["courseName"] = ""
                if "userId" not in course_data:
                    course_data["userId"] = user_id
                course_data["status"] = "READY"
                # course_data["chapters"] = chapter_list
                # course_data["chapterCount"] = len(course_data["chapters"])
                course_data["sections"] = []

                # Create uploadDocs struct
                if "uploadDocs" not in course_data:
                    course_data["uploadDocs"] = []
                find_upload_doc = 0
                for uploadDoc in course_data["uploadDocs"]:
                    if uploadDoc["url"] == material_url:
                        find_upload_doc = 1
                if find_upload_doc == 0:
                    # Update uploadDocs if not find material_url
                    uploadDoc = {"_id": ObjectId(), "fileName": os.path.basename(material_url), "url": material_url}
                    course_data["uploadDocs"].append(uploadDoc)
                course_data["fileCount"] = len(course_data["uploadDocs"])

                self.db.add_update_course(course_data)
                pipeline_data = self.db.get_pipeline(pipeline_id)
                pipeline_data["status"] = "DONE"
                pipeline_data["lastUpdatedTime"] = datetime.now(tz=timezone.utc)
                self.db.add_update_pipeline(pipeline_data)

        print(f"mongo_write_course session done, user_id: {user_id}, course_id: {course_id}, pipeline_id: {pipeline_id}, course_generation_type: {course_generation_type}")

if __name__ == "__main__":
    pass
