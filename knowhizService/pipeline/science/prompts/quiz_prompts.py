from pipeline.science.prompts.system_prompts import Prompts

class QuizPrompts(Prompts):
    @staticmethod
    def multiple_choice_definition_quiz_generation_prompt():
        """
        Prompt for generating multiple choice quiz questions based on the definition.
        """
        prompt = \
        """
        "To test whether a student understands the concept '{keyword}', please use the definition {text} to create a multiple-choice question, by coming up with a concrete example, scenario or use case (application) that illustrates the concept."

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
    def review_multiple_choice_quiz_prompt():
        """
        Prompt for reviewing generating multiple choice quiz questions
        """
        prompt = \
        """
        "Multiple choice question and answer: {quiz}."
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
    def multiple_choice_expansion_quiz_generation_prompt():
        """
        Prompt for generating multiple choice quiz questions based on the expansion.
        """
        prompt = \
        """
        "To assess a student's comprehension of the concept '{keyword}' beyond mere memorization, please use the expansion document to create a multiple-choice question:"
        "\n\n expansion document: {text}."
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