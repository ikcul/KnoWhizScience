#!env python3
# coding: utf-8

import os
from dotenv import load_dotenv
load_dotenv()

class Config(object):
    def __init__(self):
        self.dev = True

        # Local test: True, Server deploy: False
        self.self_test = False
        if self.self_test:
            self.mongo_string = 'mongodb+srv://knowhiz:Xo4G82lh3Ds7F8Sp@staging.nfz1drt.mongodb.net/?retryWrites=true&w=majority&appName=staging&connectTimeoutMS=30000&socketTimeoutMS=30000'

        else:
            self.mongo_string = os.getenv("MONGO_CONNECTION_STRING")
            if self.mongo_string == None:
                print("Please set env: SERVER_MONGODB_CONNECT!!! Use default staging string.")
                self.mongo_string = 'mongodb+srv://knowhiz:Xo4G82lh3Ds7F8Sp@staging.nfz1drt.mongodb.net/?retryWrites=true&w=majority&appName=staging&connectTimeoutMS=30000&socketTimeoutMS=30000'

        self.db_name = os.getenv("SERVER_MONGODB_db_name")
        if self.db_name == None:
            self.db_name = 'knowhiz'
        self.pipeline_collection_name = 'pipeline'
        self.course_collection_name = 'courses'
        self.flashcard_collection_name = 'flashcards'
        self.quiz_collection_name = 'quizMultiChoiceQuestions'
        self.test_mcq_collection_name = 'testMultiChoiceQuestions'
        self.test_saq_collection_name = 'testOpenEndedQuestions'
        self.s3_bucket_name = "knowhiz-dev"
        self.azure_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "knowhiz")
        self.azure_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        self.backend_url = os.getenv("BACKEND_URL")
        if self.backend_url == None:
            self.backend_url = "http://localhost:8080"

        self.discord_bot_url = os.getenv("DISCORD_BOT_URL")
        if self.discord_bot_url == None:
            self.discord_bot_url = "http://localhost:8083"

        self.file_path_prefix = os.getenv("FILE_PATH_PREFIX")
        if self.file_path_prefix == None:
            self.file_path_prefix = "/tmp/"


    def get_mongo_string(self):
        return self.mongo_string

    def get_mongodb_name(self):
        return self.db_name

    def get_mongo_pipeline_collection_name(self):
        return self.pipeline_collection_name

    def get_mongo_course_collection_name(self):
        return self.course_collection_name

    def get_mongo_flashcard_collection_name(self):
        return self.flashcard_collection_name

    def get_mongo_quiz_collection_name(self):
        return self.quiz_collection_name

    def get_mongo_test_mcq_collection_name(self):
        return self.test_mcq_collection_name

    def get_mongo_test_saq_collection_name(self):
        return self.test_saq_collection_name

    def get_backend_url(self):
        return self.backend_url

    def get_discord_bot_url(self):
        return self.discord_bot_url

    def get_s3_bucket_name(self):
        return self.s3_bucket_name

    def get_azure_container_name(self):
        return self.azure_container_name

    def get_azure_connection_string(self):
        return self.azure_connection_string

    def get_file_path_prefix(self):
        return self.file_path_prefix

    def isDev(self):
        return self.dev

    def isProduction(self):
        return not self.dev

if __name__ == "__main__":
    print("get_backend_url:", Config().get_backend_url())
    print("get_azure_container_name:", Config().get_azure_container_name())
    print("get_azure_connection_string:", Config().get_azure_connection_string())