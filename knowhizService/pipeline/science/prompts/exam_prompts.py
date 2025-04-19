from pipeline.science.prompts.system_prompts import Prompts

class ExamPrompts(Prompts):
    @staticmethod
    def multiple_choice_definition_exam_generation_prompt():
        """
        Prompt for generating multiple choice exam questions based on the definition.
        """
        prompt = \
        """
        "In order to comprehensively evaluate a student's grasp of the course material, your task is to formulate {qnum} multiple choice questions with real world context that can be sovled by applying one or several key concepts from the provided keywords and their definitions:"
        "\n\n keywords: {keyword}:"
        "\n\n definitions: {text}."
        "\n\n The question should have four choices labeled by A, B, C and D."
        "\n\n Provide the label of the correct answer for the question."
        "\n\n Format the output as a JSON file as follows."
        ------------------------------------------
        ```json
        {{
            "question": <question here>,
            "choices": {{
                "A": <choice A here>,
                "B": <choice B here>,
                "C": <choice C here>,
                "D": <choice D here>
            }},
            "correct_answer": <correct answer here, should be "A", "B", "C", or "D">
        }}
        ```
        ------------------------------------------
        Do not include special characters in the question or choices.
        """
        return prompt
    
    @staticmethod
    def review_multiple_choice_exam_prompt():
        """
        Prompt for reviewing generating multiple choice quiz questions
        """
        prompt = \
        """
        "Multiple choice question and answer: {exam}."
        "\n\n  Instructions: "
        "\n\n 1. Review the provided multiple choice question and its answer choices thoroughly;“
        "\n\n 2. Ensure that the correct answer is not always the same option (e.g., not always 'A', 'B', 'C', or 'D'). If necessary, adjust the answer choices accordingly;"
        "\n\n 3. Verify the correctness of the answer, especially if calculations or logical reasoning are involved. Recalculate or re-derive the answer if necessary;"
        "\n\n 4. If any descriptions or reasoning in the question or answer choices are ambiguous, inaccurate, or lack clarity, revise them for correctness and precision;"
        "\n\n 5. Return the revised multiple choice question and answer set as a JSON object following the specified structure:"
        ------------------------------------------
        ```json
        {{
            "question": <question here>,
            "choices": {{
                "A": <choice A here>,
                "B": <choice B here>,
                "C": <choice C here>,
                "D": <choice D here>
            }},
            "correct_answer": <correct answer here, should be "A", "B", "C", or "D">
        }}
        ```
        ------------------------------------------
        Do not include special characters in the question or choices.
        """
        return prompt
    
    
    @staticmethod
    def multiple_choice_expansion_exam_generation_prompt():
        """
        Prompt for generating multiple choice exam questions based on the expansion.
        """
        prompt = \
        """
        "In order to comprehensively evaluate a student's grasp of the course material, your task is to formulate {qnum} multiple choice questions with real world context that  can be sovled by applying one or several key concepts from the provided keywords and their expansion documents. Do not use existing examples from the expansion documents, instead create concrete example, scenario or use case (application) that illustrates the concept, and test student's understanding about these content:"
        "\n\n keywords: {keyword}:"
        "\n\n expansion documents: {text}."
        "\n\n The question should have four choices labeled by A, B, C and D."
        "\n\n Provide the label of the correct answer for the question."
        "\n\n Format the output as a JSON file as follows."
        ------------------------------------------
        ```json
        {{
            "question": <question here>,
            "choices": {{
                "A": <choice A here>,
                "B": <choice B here>,
                "C": <choice C here>,
                "D": <choice D here>
            }},
            "correct_answer": <correct answer here, should be "A", "B", "C", or "D">
        }}
        ```
        ------------------------------------------
        Do not include special characters in the question or choices.
        """
        return prompt
    
    @staticmethod
    def short_answer_questions_expansion_exam_generation_prompt():
        """
        Prompt for generating short answer exam questions.
        """
        prompt = \
        """
        "In order to comprehensively evaluate a student's grasp of the course material, your task is to formulate {qnum} short anwser questions with real world context that  can be sovled by applying one or several key concepts from the provided keywords and their expansion documents:"
        "\n\n keywords: {keyword}:"
        "\n\n expansion documents: {text}."
        "\n\n Provide the answer of the question."
        "\n\n Format the output as a JSON file with question as key and answer as value."
        ------------------------------------------
        ```json
        {{
            "question": <question here>,
            "answer": <answer here>
        }}
        ```
        ------------------------------------------
        Do not include special characters in the question or choices.
        """
        return prompt
    
    @staticmethod
    def short_answer_questions_definition_exam_generation_prompt():
        """
        Prompt for generating short answer exam questions.
        """
        prompt = \
        """
        "In order to comprehensively evaluate a student's grasp of the course material, your task is to formulate {qnum} short anwser questions with real world context that  can be sovled by applying one or several key concepts from the provided keyword list and their definitions:"
        "\n\n keyword list: {keyword}:"
        "\n\n definitions: {text}."
        "\n\n Provide the answer of the question."
        "\n\n Format the output as a JSON file with question as key and answer as value."
        ------------------------------------------
        ```json
        {{
            "question": <question here>,
            "answer": <answer here>
        }}
        ```
        ------------------------------------------
        Do not include special characters in the question or choices.
        """
        return prompt
    
    @staticmethod
    def review_short_answer_exam_prompt():
        """
        Prompt for reviewing generating short answer questions
        """
        prompt = \
        """
        "Short answer question and answer: {exam}."
        "\n\n  Instructions: "
        "\n\n 1. Review the provided short answer question and its answer choices thoroughly;“
        "\n\n 2. Verify the correctness of the answer, especially if calculations or logical reasoning are involved. Recalculate or re-derive the answer if necessary;"
        "\n\n 3. If any descriptions or reasoning in the question or answer choices are ambiguous, inaccurate, or lack clarity, revise them for correctness and precision;"
        "\n\n 4. Return the revised multiple choice question and answer set as a JSON object following the specified structure:"
        ------------------------------------------
        ```json
        {{
            "question": <question here>,
            "answer": <answer here>
        }}
        ```
        ------------------------------------------
        Do not include special characters in the question or choices.
        """
        return prompt