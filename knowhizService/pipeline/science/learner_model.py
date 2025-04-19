import hashlib
import json
import os
import pandas as pd
import numpy as np
import logging

import tkinter as tk
from tkinter import ttk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional

from pipeline.science.api_handler import ApiHandler
from pipeline.science.doc_handler import DocHandler
from pipeline.science.prompt_handler import PromptHandler

logger = logging.getLogger("kzpipeline.science.learner_model")

def is_valid_dict(input_dict):
    """
    Check if the input is a dictionary with keys as strings and values as numbers.

    Args:
        input_dict (dict): The dictionary to validate.

    Returns:
        bool: True if the dictionary is valid, False otherwise.
    """
    if not isinstance(input_dict, dict):
        return False

    for key, value in input_dict.items():
        if not isinstance(key, str):
            return False
        if not isinstance(value, (int, float)):
            return False

    return True

# Defining structures for Langchain output parsing
class Transcripts_type(BaseModel):
    """
    A Pydantic model to represent course evaluation scores where keys are course names
    and values are the scores.
    """
    scores: Dict[str, int] = Field(default={}, description="A dictionary of course names and their respective scores")

# Defining structures for Langchain output parsing
class Prerequisites(BaseModel):
    """
    A Pydantic model for the prerequisites of a learner.
    """
    prerequisites: List[str] = Field(default=[], description="List of prerequisite courses")

# Defining structures for Langchain output parsing
class Description(BaseModel):
    """
    A Pydantic model for the description of a learner.
    """
    description: str = Field(default='', description="List of prerequisite courses")

class Learner_model:
    def __init__(self, para):
        """
        Initialize the learner model with the given parameters.
        """
        self.number_of_prerequisites = para['number_of_prerequisites']
        self.learner_description = {}

        # Load llm
        self.api = ApiHandler(para)
        self.llm_advance = self.api.models['advance']['instance']
        self.llm_basic = self.api.models['basic']['instance']

        # Load the learner's information
        self.learner_info = para['learner_info']    # Load the learner's basic information, with only learner's ID is mandatory
        self.learner_info_path = para['learner_info_path']  # Path to the all learners' profiles
        self._hash_learner_info()   # Hash the learner's ID to create a unique identifier
        self.results_dir = para['results_dir']  # Path to the results folder
        self.learners_dir = os.path.join(self.learner_info_path + self.learner_hash_id + "/")   # This learner's directory
        os.makedirs(self.learners_dir, exist_ok=True)
        self._initialize_transcripts()  # Initialize the learner's transcripts by loading the transcripts.json file. If the file does not exist, initialize the transcripts dictionary with an empty dictionary.

    def _hash_learner_info(self):
        """
        Hash the learner's ID to create a unique identifier.
        """
        # Initialize a hashlib object for SHA-224
        sha224_hash = hashlib.sha224()
        sha224_hash.update(self.learner_info['learner_id'].encode("utf-8"))

        # Calculate the final hash
        self.learner_hash_id = sha224_hash.hexdigest()

    def _initialize_transcripts(self):
        """
        Initialize the learner's transcripts from the transcripts.json file.
        If the file does not exist, initialize the transcripts dictionary with an empty dictionary.
        """
        with open(self.learners_dir + "learner_info.json", 'w') as file:
            json.dump(self.learner_info, file, indent=2)
        path = self.learners_dir + "transcripts.json"
        if(os.path.exists(path)):
            # Initialize the transcripts dictionary with an empty dictionary
            with open(path, 'r') as file:
                self.transcripts = json.load(file)
        else:
            self.transcripts = {}
            with open(path, 'w') as file:
                json.dump(self.transcripts, file, indent=2)

    def _merge_transcripts(self, original, temp, threshold=0.5):
        """
        Merge two dictionaries of transcripts based on cosine similarity of keys.
        If the cosine similarity of a key in the temp dictionary with a key in the original dictionary is above the threshold,
        the values of the two keys are averaged and the key-value pair is added to the original dictionary.
        If the cosine similarity is below the threshold, the key-value pair is added to the original dictionary.
        """
        # Convert dictionary keys to lists
        original_keys = list(original.keys())
        temp_keys = list(temp.keys())

        # Create a vectorizer and fit on all keys
        vectorizer = TfidfVectorizer()
        vectorizer.fit(original_keys + temp_keys)
        original_vectors = vectorizer.transform(original_keys)
        temp_vectors = vectorizer.transform(temp_keys)

        # Iterate through each temp key and vector
        for temp_key, temp_vec in zip(temp_keys, temp_vectors):
            # Calculate cosine similarities
            similarities = cosine_similarity(temp_vec, original_vectors)[0]

            # Find the best matching key in the original dictionary
            max_similarity_index = np.argmax(similarities)
            max_similarity = similarities[max_similarity_index]

            if max_similarity >= threshold:
                matched_key = original_keys[max_similarity_index]
                # Averaging the values (assuming values are lists of numerical data)
                original_value = np.array(original[matched_key])
                temp_value = np.array(temp[temp_key])
                average_value = (original_value + temp_value) / 2
                original[matched_key] = average_value.tolist()
            else:
                # If no match found with sufficient similarity, add new key-value pair
                original[temp_key] = temp[temp_key]
        return original

    def _update_transcript(self, transcripts_temp):
        """
        Update the learner's transcript with the new transcript.
        """
        # transcripts_temp = {key.lower(): value for key, value in transcripts_temp.items()}
        if(self.transcripts == {}):
            self.transcripts = transcripts_temp
        else:
            self.transcripts = self._merge_transcripts(original = self.transcripts, temp = transcripts_temp)
        path = self.learners_dir + "transcripts.json"
        with open(path, 'w') as file:
            json.dump(self.transcripts, file, indent=2)

    def _learner_transcripts_evaluation(self, evaluation_results_temp):
        """
        Given a description of a learner (like results from user quiz about prerequisites for the learning topics),
        evaluate the learner's score in different subjects and return the evaluation.
        Storing the results in a json file.

        # Example transcripts
        transcript1 = {'Math': 90, 'English': 88, 'Science': 85}
        transcript2 = {'Math': 92, 'English': 84, 'History': 90}
        """
        if(not is_valid_dict(evaluation_results_temp)):
            llm = self.llm_basic
            parser = JsonOutputParser(pydantic_object=Transcripts_type)
            prompt = PromptTemplate(
                template=
                    """
                    Requirements: \n\n\n
                    Evaluation results: ```{evaluation_results_temp}```
                    Based on the evaluation of the learner, follow the following steps to evaluate the learner's score in different subjects:
                    1. Based on the evaluation results, evaluate the learner's capability in different subjects. Refine the name of the subjects if needed.
                    2. Format the evaluation as a transcript of the learner in a dictionary.
                    3. For the subjects that cannot give a numeric score or the value does not make sense or the subject name is not really a subject, exclude them from the transcript.
                    The response must be formated as json, subject name should be string format and score should be float format:
                    ```json
                    {{
                    \"<subject_1>\": <score for this subject here, range from 0 to 100>,
                    \"<subject_2>\": <score for this subject here, range from 0 to 100>,
                    ...
                    \"<subject_n>\": <score for this subject here, range from 0 to 100>
                    }}
                    ```
                    Do not include "```" in response.
                    """,
                input_variables=["evaluation_results_temp"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            chain = prompt | llm | parser
            try:
                transcripts_temp = chain.invoke({'evaluation_results_temp': str(evaluation_results_temp)})
            except Exception as e:
                logger.exception(f"Exception: {e}")
                transcripts_temp = chain.invoke({'evaluation_results_temp': str(evaluation_results_temp)})
        else:
            transcripts_temp = evaluation_results_temp
        self._update_transcript(transcripts_temp)
        logger.info(f"\nLearner's current transcripts: {self.transcripts}")

    def prerequisites_analysis(self, learning_topic):
        """
        Given basic information about a learner, as well as the leaning topic for the course that this learner is interested in,
        Generate a list of prerequisites that are not contained in the learner's transcript profile and are important for the learner to learn the course.
        """
        llm = self.llm_advance
        parser = JsonOutputParser(pydantic_object=Prerequisites)
        prompt = PromptTemplate(
            template=
                """
                You are a professor teaching learning topic: ```{learning_topic}```.
                The student that you are teaching has current transcript profile: ```{transcripts}```.
                Generate a list of prerequisites that this student must take before learning this topic.
                1. Based on the learning topic, identify the prerequisites courses for this learning topic as a list.
                The list should be no longer than {number_of_prerequisites} items.
                2. Refine the list of prerequisites to only include the ones that this student has not taken yet.
                3. If the prerequisites is already in the student's transcripts, return an empty list under "prerequisites".
                The response should be formated as json:
                ```json
                {{
                "prerequisites": [
                    "<prerequisite_1>",
                    "<prerequisite_2>",
                    ...
                    "<prerequisite_n>"
                ]
                }}
                ```
                Do not include "```" in response.
                """,
            input_variables=["learning_topic", "transcripts", "number_of_prerequisites"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        prerequisites_temp = chain.invoke({'learning_topic': learning_topic, 'transcripts': self.transcripts, 'number_of_prerequisites': self.number_of_prerequisites})
        logger.info(f"\nThe prerequisites for the learner to learn the course are: {prerequisites_temp["prerequisites"]}")
        path = self.learners_dir + "prerequisites_temp.json"
        with open(path, 'w') as file:
            json.dump(prerequisites_temp, file, indent=2)

        # For local testing: get the prerequisites in the pop-up window

        if(prerequisites_temp != []):
            # Define the scores
            scores = {
                "poor": 0,
                "fair": 75,
                "good": 85,
                "excellent": 95
            }
            # Define the prerequisites
            prerequisites = prerequisites_temp["prerequisites"]
            # Create the main application window
            root = tk.Tk()
            root.title("Rate Prerequisites")
            # Create a dictionary to store user selections
            user_selections = {}
            # Function to submit the ratings
            def submit():
                results = {}
                for prereq in prerequisites:
                    rating = user_selections[prereq].get()
                    if rating in scores:
                        results[prereq] = scores[rating]
                    else:
                        results[prereq] = 0
                logger.info(results)
                # Save the results to a JSON file
                results_path = self.learners_dir + "prerequisite_temp_results.json"
                with open(results_path, 'w') as file:
                    json.dump(results, file, indent=2)
                root.destroy()
                return results
            # Create UI elements for each prerequisite
            for prereq in prerequisites:
                frame = ttk.Frame(root, padding="10")
                frame.pack(fill='x')
                label = ttk.Label(frame, text=f"Rate your proficiency in {prereq}:")
                label.pack(side='left')
                user_selections[prereq] = tk.StringVar()
                for rating in scores.keys():
                    rb = ttk.Radiobutton(frame, text=rating.capitalize(), variable=user_selections[prereq], value=rating)
                    rb.pack(side='left')
            # Add a submit button
            submit_button = ttk.Button(root, text="Submit", command=submit)
            submit_button.pack(pady=20)
            # Run the application
            root.mainloop()

            # Load the results from the JSON file
            with open(self.learners_dir + "prerequisite_temp_results.json", 'r') as file:
                results = json.load(file)
            logger.info(f"\nThe results of the prerequisites evaluation are: {results}")

            self._learner_transcripts_evaluation(results)

        return prerequisites_temp["prerequisites"]

    def learner_description_generation(self, learning_topic):
        """
        Generate a description of the learner based on the learner's transcript profile.
        """
        llm = self.llm_advance
        path = os.path.join(self.learners_dir, "learner_description.json")
        if os.path.exists(path):
            with open(path, 'r') as file:
                self.learner_description = json.load(file)
            logger.info(f"\nThe original description of the learner is: {self.learner_description}")
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            template=
                """
                As a professor, you are teaching course: ```{learning_topic}```,
                to students with the following information:
                Students' current transcript profile: ```{transcripts}```
                Based on the students' transcript profile, provide a description of the students.
                The description should be a summary of the students' capabilities and areas of expertise, only focus on the students' capabilities that are related to this specific course.
                Do not include any personal information of the student or specific course information or grades.
                It should include the students' strengths and weaknesses, organized as a list as below:
                1. A intoduction of the student about students' background.
                2. Going through a few main areas and describe the students' capabilities in these areas.
                3. Point out the students' strengths and weaknesses.
                Therefore these information can be used as a reference for the instructor to provide better teaching.
                The entire description should be no longer than 100 words.
                The response should be formated as json:
                ```json
                {{
                "description": "<description of the student here>"
                }}
                ```
                Do not include "```" in response.
                """,
            input_variables=["transcripts", "learning_topic"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        description_temp = chain.invoke({'transcripts': self.transcripts, 'learning_topic': learning_topic})
        self.learner_description = description_temp["description"]
        with open(path, 'w') as file:
            # json.dump(description_temp, file, indent=2)
            json.dump(self.transcripts, file, indent=2)
        logger.info(f"\nThe updated description of the learner is: {self.learner_description}")

        return self.transcripts # self.learner_description