from pipeline.science.prompts.system_prompts import Prompts

class ZeroshotPrompts(Prompts):
    @staticmethod
    # Step 1: Topic extraction
    def topic_extraction_prompt():
        """
        Prompt for extracting the zero-shot topic from the course information.
        """
        prompt = \
        """
        Requirements: \n\n\n
        Based on the information of the course information about the course that a student wants to learn: ```{course_info}```.
        "Context" is a restrictive description of the course,
        and "subject" is the general topic of the course,
        and "text" is the detailed description about the content that this user wants to learn.
        Please answer: what is the course_name_domain of this course should be by combining "context", "subject", and "text".
        For example, input can be like this:
        ```
        context: "Bayesian"
        level: "Beginner"
        subject: "Computer Science"
        text: "Bayesian machine learning techniques"
        ```
        The response should be formated as json:
        ```json
        {{
        "text": <what is the input text part for the course generation request, but add one emoji at the beginning>,
        "level": <what is the difficulty level of this course, like beginner level, medium level, or advanced level etc.>,
        "subject": <what is the subject of this course, options include {subject_options}>,
        "course_name_domain": <what is the detailed full description for the topic of this course (will be auto generated if not provided)>
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 2: Chapters generation
    def chapters_generation_prompt():
        """
        Prompt for generating chapters based on the zero-shot topic.
        """
        prompt = \
        """
        You are a course creator. I will provide a topic, but **no explicit coverage type**. You must determine whether the topic should be treated as "quick learn" (or marked as "pre-college level") or a more in-depth course:

        - If, from context, you believe the topic is for quick and easy learning, create 3–5 chapters.
        - Otherwise, create 7–10 chapters like a more in-depth course in college.

        Output your response in **valid JSON format only**, using the structure:

        {{
        "Course name": "Your Course Title Here",
        "Chapters": [
            "🗂️ Chapter 1",
            "🧠 Chapter 2",
            ...
        ]
        }}

        Ensure:
        1. The "Course name" key exactly matches the topic or a suitably rephrased version.
        2. The "Chapters" key contains an appropriate list of chapter titles based on your inferred coverage type (3–5 if "quick learn", 7–10 if in-depth).
        3. Each chapter starts with an suitable emoji as examples below.
        4. Provide no additional text or explanation—only the JSON.

        Example of the desired output for a "quick learn" topic:

        ```json
        {{
        "Course name": "How to bake a cake",
        "Chapters": [
            "🧙 Mixing Magic",
            "🥘 Dough Essentials",
            "✨ Rise and Shine",
            "🍪 Oven Mastery",
            "🖌️ Finishing Touches",
        ]
        }}
        ```

        Example of the desired output for a "in-depth learn" topic:

        ```json
        {{
        "Course name": "College Level Linear Algebra",
        "Chapters": [
            "🧮 Systems of Linear Equations",
            "👨‍🔬 Matrix Algebra",
            "🧠 Determinants",
            "🛰️ Vector Spaces",
            "🚂 Linear Transformations",
            "📡 Orthogonality and Inner Product Spaces",
            "🚀 Eigenvalues and Eigenvectors",
            "📑 Diagonalization and Spectral Theorems",
            "💯 Advanced/Additional Topics",
        ]
        }}
        ```

        Now, here is my topic: {extracted_course_name_domain}
        """
        return prompt
    
    @staticmethod
    # Step 3: Keywords generation for each chapter
    def keywords_generation_prompt():
        """
        Prompt for generating keywords for each chapter.
        """
        prompt = \
        """
        You are an expert in {course_name_domain}.
        And for this learning topic we have total list of chapters: {chapters_list}.
        Now come up with the keywords specifically in one of the chapter: {chapter_name}.
        If the course is a "quick learn" (or marked as "pre-college level") course (the number of chapters is 6-8), provide more concise (but no less than 3) keywords for this chapter.
        If the course is a "in-depth learn" course (the number of chapters is 9-12), provide more detailed (but no more than 10) keywords for this chapter.
        Only list the most important keywords.
        The response should be a proper JSON format:
        ```json
        {{
        "keywords": [
            <Keyword_1>,
            <Keyword_2>,
            ...
            <Keyword_n>,
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 3: Keywords generation: remove duplicate keywords and organize them into a list of lists
    def keywords_cleaning_prompt():
        """
        Prompt for cleaning the keywords and organizing them into a list of lists.
        """
        prompt = \
        """
        Based on {raw_keywords_in_chapters}, the keywords in a list of lists. The length of the list should be the same as the number of chapters.
        Make sure every keyword is unique: If one keyword has a similar meaning with another keyword in another chapter,
        only keep the first one (with lower chapter index) and remove the other keywords.
        Chapter list: ```{chapters_list}```.
        Use the following json format:
        ----------------
        {{
        "keywords": [
            [<keyword_1>, <keyword_2>, ..., <keyword_n>],
            [<keyword_1>, <keyword_2>, ..., <keyword_m>],
            ...
            [<keyword_1>, <keyword_2>, ..., <keyword_p>],
        ]
        }}
        ----------------
        """
        return prompt

    @staticmethod
    # Step 4: Flashcards definition generation
    def flashcards_definition_generation_prompt():
        """
        Prompt for generating flashcards definitions for each keyword.
        """
        prompt = \
        """
        For course {course_name}, chapter {chapter_name}, provide the definition of the keyword: {keyword} in a sentence that is accurate and easy to understand.
        In the response include no prefix or suffix.
        Make sure the definition is accurate.
        Max words for definition: {definition_length}
        The response should use markdown syntax to highlight important words / parts in bold or underlined,
        but do not include "```markdown" in the response.
        """
        return prompt

    @staticmethod
    # Step 5: Flashcards expansion generation
    def flashcards_expansion_generation_prompt():
        """
        Prompt for generating flashcards expansions for each keyword.
        """
        prompt = \
        """
        You are given the following information:
        •	Course: {course_name}
        •	Chapter: {chapter_name}
        •	Whole keywords list in this chapter: {keyword_list}

        Task is to provide a short and concise explanation in addition to the keyword: {keyword}'s definition "{definition}":
        1.	Use short sentences to explain things in a friendly, educational tone.
        2.	Do not assume the reader has any prior knowledge.
        3.	Use relevant examples, or analogies to clarify the concept.
        4.	Avoid including an "Overview" "Conclusion" or "Summary"
        5.  Avoid repeating the definition in the explanation.

        Markdown Formatting Requirements
        •	Use bold formatting to highlight certain important words.
        •	Use emojis at the beginning of bullet points to make the content visually engaging.
        •	Use more mathematical notation is needed, and for all mathematical expressions only use $$ ... $$ / $ ... $ as syntax for LaTeX formulas.
        """
        return prompt

    @staticmethod
    # Step 5: Flashcards expansion generation
    def flashcards_expansion_generation_prompt_serious():
        """
        Prompt for generating flashcards expansions for each keyword.
        """
        prompt = \
        """
        Complete the task step by step:

        For the course: {course_name}, chapter: {chapter_name}, provide an Example section for the keyword: {keyword}.
        {keyword}'s definition is: {definition}.

        Make sure the expansions are accurate.
        Max words for expansion: {expansion_length}
        It should formated as markdown:
        {markdown_format_string}

        1. The section name is 'Example', which only inludes a real world example to help memerize and understand the keyword in {course_name}.
        2. Please do not provide the definition of the keyword in the example.
        3. Within the example, if you need to display formulas, include them in LaTeX syntax formatted in markdown, as shown below:
            ----------------
            $$
            \frac{{a}}{{b}} = \frac{{c}}{{d}}
            $$
            ----------------
        4. Within the example, if you need to display tables, format them using markdown as follows:
            ----------------
            ## Table

            | Header 1   | Header 2   | Header 3   |
            |------------|------------|------------|
            | Row 1 Col 1| Row 1 Col 2| Row 1 Col 3|
            | Row 2 Col 1| Row 2 Col 2| Row 2 Col 3|
            | Row 3 Col 1| Row 3 Col 2| Row 3 Col 3|
            ----------------

        5. Do not include "```markdown" in the response. Final whole response must be in correct markdown format.
        6. Specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.
        7. For in-line formulas, use the syntax: $E = mc^2$. Remember must use double ```$``` for display formulas.
        """
        return prompt

    # def flashcards_expansion_generation_prompt():
    #     """
    #     Prompt for generating flashcards expansions for each keyword.
    #     """
    #     prompt = \
    #     """
    #     Complete the task step by step:

    #     For the course: {course_name}, chapter: {chapter_name}, provide the expansions with a few pre-defined regions for the keyword: {keyword}.
    #     {keyword}'s definition is: {definition}.

    #     Make sure the expansions are accurate.
    #     Max words for expansion: {expansion_length}
    #     It should formated as markdown:
    #     {markdown_format_string}

    #     1. The first region is "Outline" which should be some really brief bullet points about the following content around that keywords.
    #     2. If the concept can be better explained by formulas, use LaTeX syntax in markdown, like:
    #         ----------------
    #         $$
    #         \frac{{a}}{{b}} = \frac{{c}}{{d}}
    #         $$
    #         ----------------
    #     3. If you find you need to add tables, use markdown format, like:
    #         ----------------
    #         ## Table

    #         | Header 1   | Header 2   | Header 3   |
    #         |------------|------------|------------|
    #         | Row 1 Col 1| Row 1 Col 2| Row 1 Col 3|
    #         | Row 2 Col 1| Row 2 Col 2| Row 2 Col 3|
    #         | Row 3 Col 1| Row 3 Col 2| Row 3 Col 3|
    #         ----------------

    #     4. Do not include "```markdown" in the response. Final whole response must be in correct markdown format.
    #     5. Specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.
    #     6. For in-line formulas, use the syntax: $E = mc^2$. Remember must use double ```$``` for display formulas.
    #     """
    #     return prompt