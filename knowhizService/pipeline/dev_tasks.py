from pipeline.science.flashcards import Flashcards
from pipeline.science.zeroshot_flashcards import Zeroshot_Flashcards
from pipeline.science.quiz import Quiz
from pipeline.science.exam import Test

import time

def generate_flashcards(para):
    st = time.time()
    if para["zero_shot"]:
        myflashcards = Zeroshot_Flashcards(para)
        myflashcards.create_chapters()
        elapsed_time = time.time() - st     
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create chapters: {minutes} mins and {seconds} seconds for the course for the request {para['course_info']}.")

        myflashcards.create_keywords()
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create keywords: {minutes} mins and {seconds} seconds for the course for the request {para['course_info']}.")

        myflashcards.create_flashcards()
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create flashcards: {minutes} mins and {seconds} seconds for the course for the request {para['course_info']}.")

        myquiz = Quiz(para, myflashcards.flashcard_dir, myflashcards.quiz_dir)
        myquiz.create_quiz()
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create quizzes: {minutes} mins and {seconds} seconds for the course for the request {para['course_info']}.")

        mytest = Test(para, myflashcards.flashcard_dir, myflashcards.test_dir)
        mytest.create_test()
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create tests: {minutes} mins and {seconds} seconds for the course for the request {para['course_info']}.")

        # para['course_id'] = myflashcards.course_id
        # print(f"\nCourse ID: {para['course_id']}")

    else:
        myflashcards = Flashcards(para)
        
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create embedding: {minutes} mins and {seconds} seconds for the file {[para['main_filenames'], para['supplementary_filenames']]}.")

        myflashcards.create_keywords()
        
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create keywords and chapters: {minutes} mins and {seconds} seconds for the file {[para['main_filenames'], para['supplementary_filenames']]}.")

        myflashcards.create_flashcards("user_id", "hash_1")

        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create flashcards: {minutes} mins and {seconds} seconds for the file {[para['main_filenames'], para['supplementary_filenames']]}.")

        myquiz = Quiz(para, myflashcards.docs.flashcard_dir, myflashcards.docs.quiz_dir)
        myquiz.create_quiz()
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create quizzes: {minutes} mins and {seconds} seconds for the file {[para['main_filenames'], para['supplementary_filenames']]}.")

        mytest = Test(para, myflashcards.docs.flashcard_dir, myflashcards.docs.test_dir)
        mytest.create_test()
        elapsed_time = time.time() - st
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time to create tests: {minutes} mins and {seconds} seconds for the file {[para['main_filenames'], para['supplementary_filenames']]}.")

        para['course_id'] = myflashcards.docs.course_id
        print(f"\nCourse ID: {para['course_id']}")
