from pipeline.science.prompts.system_prompts import Prompts

class FlashcardsPrompts(Prompts):
    @staticmethod
    # Step 0: Chapters creation
    def chapters_creation_with_content_prompt():
        """
        Prompt for creating chapters with content pages.
        """
        prompt = \
        """
        Requirements: \n\n\n
        As as a professor teaching course: {course_name_domain}.
        Using textbook with content ```{textbook_content_pages}```.
        Please work through the following steps:
        1. Find the textbook name and author, note down it as ```textbook and author```.
        2. Based on the content attached, find the chapters of this book.
        3. Then note down the chapters with the following format. For each chapter name, do not include the chapter number.
        4. The length of each chapter should be no more than 50 charactors.
        The output format should be:
        ```json
        {{
        "Course name": <course name here>,

        "Textbooks": [
            <textbook here>,
        ]

        "Chapters": [
            <chapter_1>,
            <chapter_2>,
            ...
            <chapter_n>,
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 0: Chapters creation
    def chapters_creation_no_content_prompt():
        """
        Prompt for creating chapters without content pages.
        """
        prompt = \
        """
        Requirements: \n\n\n
        As as a professor teaching course: {course_name_domain}.
        Please work through the following steps:
        1. Find a textbook name and author for this book, note down it as ```textbook and author```.
        2. Based on the content attached, find the chapters of this book. The number of chapters should be between 5 and 15.
        3. Then note down the chapters with the following format. For each chapter name, do not include the chapter number.
        The output format should be:
        ```json
        {{
        "Course name": <course name here>,

        "Textbooks": [
            <textbook here>,
        ]

        "Chapters": [
            <chapter_1>,
            <chapter_2>,
            ...
            <chapter_n>,
        ]
        }}
        ```
        """
        return prompt

    @staticmethod
    # Step 1: Keywords extraction
    def keywords_extraction_links_prompt():
        """
        Prompt for extracting keywords from links.
        """
        prompt = \
        """
        For the given text ```{text}```, please identify a few central keywords that should be emphasized and remembered.

        Output json format:
        ```json
        {{
        "concepts": [
            <concept 1>,
            <concept 2>,
            ...
            <concept n>
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 1: Keywords extraction
    def keywords_extraction_fix_prompt():
        """
        Prompt for fixing keywords extraction
        if(len(self.keywords) > self.link_flashcards_size)
        """
        prompt = \
        """
        Note: please identify no more than {nkeys} central keywords that should be emphasized and remembered.
        """
        return prompt

    @staticmethod
    # Step 1: Keywords extraction
    def keywords_extraction_prompt():
        """
        Prompt for extracting keywords from given index text.
        """

        prompt = \
        """
        As as a professor teaching course: {course_name_domain}\n\n"
        "As if you were the professor teaching this course, please identify {nkey} critical keywords
        from the provided index section that are essential for students to understand and memorize.
        Do not include the explanations of keywords.
        Do not include keywords that are not central to
        the course content, such as examples, datasets, exercises, problems, or introductory keywords:
        "\n\n Index section: {index_docs}"

        Output json format:
        ```json
        {{
        "Keywords": [
            <keyword 1>,
            <keyword 2>,
            ...
            <keyword n>
        ]
        }}
        ```
        """

        prompt = \
        """
        As as a professor teaching course: {course_name_domain}.
        From the following text, please identify and return only a list of {nkey} critical keywords
        from the provided word section (many words included) that are essential 
        for students to understand and memorize.
        Do not include keywords that are not central it the course content, 
        such as examples, datasets, exercises, problems, or introductory keywords.
        text section: {index_docs}

        Output json format:
        ```json
        {{
        "Keywords": [
            <keyword 1>,
            <keyword 2>,
            ...
            <keyword n>
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 1: Keywords extraction
    def keywords_extraction_refine_prompt():
        """
        Prompt for refining keywords extraction.
        """
        prompt = \
        """
        "Given a course and its domain as: {course_name_domain} \n\n"
        "Assume you are a professor teaching this course. Using your knowledge of the general subject
        matter within this domain, please filter out any irrelevant keywords from the given keywords.  Please return the remaining keywords as a list separated by commas. "
        "\n\n keywords: {keywords}"

        Output json format:
        ```json
        {{
        "Keywords": [
            <keyword 1>,
            <keyword 2>,
            ...
            <keyword n>
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 2: Keywords assignment
    def keywords_assignment_c2k_prompt():
        """
        Prompt for assigning keywords to chapters.
        """
        prompt = \
        """
        Based on {keywords_text_in_chapters}, please extract the keywords as a list of lists. The length of the list should be the same as the number of chapters.
        Chapter list: ```{chapters_list}```.
        Do not miss out any chapters. The output length should be a list (length equal to length of chapters) of lists.
        Output format in json, with number of lists is equal to the number of chapters:
        ```json
        {{
        "keywords_list": [
            [<keyword_1>, <keyword_2>, ..., <keyword_n>],
            [<keyword_1>, <keyword_2>, ..., <keyword_m>],
            ...
            [<keyword_1>, <keyword_2>, ..., <keyword_p>],
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 2: Keywords assignment refinement
    def keywords_assignment_refinement_prompt():
        """
        Prompt for refining keywords assignment.
        """
        prompt = \
        """
        Your task is classifying key concepts of a given course into its given chapter list.
        To solve the problem do the following:
        Things you should know: based on the content of this textbook, it has the following learning chapters
        ```{course_name_textbook_chapters}```
        Based on your own knowledge, for each learning topic (chapters),
        find most relavant keywords that belongs to this chapter from a keywords list: ```{sorted_keywords}```.
        1. Make sure each keyword is assigned to at least one chapter.
        2. Make sure each chapter has at least {min_num} keywords and no more than {max_num} keywords.
        3. Do not have similar keywords in different chapters.
        Use the following format:
        Course name:
        <course name here>
        Learning topics:
        <chapter here>
        <Keywords: keywords for the above topic here>
        """
        return prompt
    
    @staticmethod
    # Step 2: Keywords assignment refinement
    def keywords_assignment_refinement_format_prompt():
        """
        Prompt for refining keywords assignment format (as list of lists).
        """
        prompt = \
        """
        Based on {keywords_text_in_chapters}, please extract the keywords as a list of lists. The length of the list should be the same as the number of chapters.
        Chapter list: ```{chapters_list}```.
        Do not miss out any chapters. The output length should be a list (length equal to length of chapters) of lists.
        Output format in json, with number of lists is equal to the number of chapters:
        ```json
        {{
        "keywords_list": [
            [<keyword_1>, <keyword_2>, ..., <keyword_n>],
            [<keyword_1>, <keyword_2>, ..., <keyword_m>],
            ...
            [<keyword_1>, <keyword_2>, ..., <keyword_p>],
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 2: Keywords assignment refinement
    def keywords_assignment_refinement_fix_prompt():
        """
        Prompt for fixing keywords assignment refinement.
        if(any(len(keyword_group) < 5 for keyword_group in self.keywords_list))
        """
        prompt = \
        """
        Your task is classifying key concepts of a given course into its given chapter list.
        To solve the problem do the following:
        Things you should know: based on the content of this textbook, it has the following learning chapters
        ```{course_name_textbook_chapters}```
        And the keywords list for the list of chapters is ```{keywords_list_original}```.

        Refine tha chapters list, and refine the number of keywords in each chapter by picking and using keywords from a large keywords list: ```{sorted_keywords}```. Add more keywords to each chapter if the number of keywords is less than {min_num}.
        Make sure each chapter has at least {min_num} keywords.
        Use the following format:
        Course name:
        <course name here>
        Learning topics:
        <chapter here>
        <Keywords: keywords for the above topic here>
        """
        return prompt
    
    @staticmethod
    # Step 2: Keywords assignment refinement
    def keywords_assignment_refinement_fix_format_prompt():
        """
        Prompt for fixing keywords assignment refinement format (as list of lists).
        """
        prompt = \
        """"
        Based on {keywords_text_in_chapters}, please extract the keywords as a list of lists. The length of the list should be the same as the number of chapters.
        Chapter list: ```{chapters_list}```.
        Do not miss out any chapters. The output length should be a list (length equal to length of chapters) of lists.
        Output format in json, with number of lists is equal to the number of chapters:
        ```json
        {{
        "keywords_list": [
            [<keyword_1>, <keyword_2>, ..., <keyword_n>],
            [<keyword_1>, <keyword_2>, ..., <keyword_m>],
            ...
            [<keyword_1>, <keyword_2>, ..., <keyword_p>],
        ]
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 2: Keywords assignment refinement
    def keywords_assignment_k2c_prompt():
        """
        Prompt for assigning chapters to keywords.
        """
        prompt = \
        """
        Course name: {course_name}
        Chapter list: {chapters}
        Which chapter does the keyword '{keyword}' belong to? Use the same chapter name as in the chapter list.
        response in the json format:
        ```json
        {{
        "chapter": <chapter name here>
        }}
        ```
        """
        return prompt
    
    @staticmethod
    # Step 3: Flashcards definition generation
    def flashcards_definition_prompt():
        """
        Prompt for generating flashcards definition.
        """
        prompt = \
        """
        Provide the definition of the keyword: {keyword} in a sentence that is accurate and easy to understand, based on the given context as below:
        Context to extract keyword definition: {text}.
        In the response include no prefix or suffix.
        Max words for definition: {max_words_flashcards}
        The response should use markdown syntax to highlight important words / parts in bold or underlined,
        but do not include "```markdown" in the response.
        """
        return prompt
    
    @staticmethod
    # Step 4: Flashcards expansion generation
    def flashcards_expansion_prompt():
        """
        Prompt for generating flashcards expansion.
        """
        prompt = \
        """
        For the course: {course_name_domain}, provide an Example section for the keyword: {keyword}.
        {keyword}'s definition is: {definition}.
        
        Generate expansions based on the given context as below:
        Context to extract keyword definition: {text}.
        Max words for expansion: {max_words_expansion}
        It should formated as markdown:
        {markdown_format_string}

        1. The section name is 'Example', which only inludes a real world example to help memerize and understand the keyword in {course_name_domain}.
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
    
    @staticmethod
    # Step 5: Flashcards rich content generation
    def flashcards_rich_content_prompt():
        """
        Prompt for generating flashcards rich content.
        """
        prompt = \
        """
        For the course: {course_name_domain}, provide rich content for the keyword: {keyword}.
        {keyword}'s definition is: {definition}.
        
        Generate rich content based on the given context as below:
        Context to extract keyword definition: {text}.
        Max words for rich content: {max_words_rich_content}
        It should formated as markdown:
        {markdown_format_string}

        1. The section name is 'Example', which only inludes a real world example to help memerize and understand the keyword in {course_name_domain}.
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
    
    # @staticmethod
    # # Step 4: Flashcards expansion generation
    # def flashcards_expansion_prompt():
    #     """
    #     Prompt for generating flashcards expansion.
    #     """
    #     prompt = \
    #     """
    #     For the course: {course_name_domain}, provide the expansions with a few pre-defined regions for the keyword: {keyword}.
    #     {keyword}'s definition is: {definition}.
        
    #     Generate expansions based on the given context as below:
    #     Context to extract keyword definition: {text}.
    #     Max words for expansion: {max_words_expansion}
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
    
    # @staticmethod
    # # Step 5: Flashcards rich content generation
    # def flashcards_rich_content_prompt():
    #     """
    #     Prompt for generating flashcards rich content.
    #     """
    #     prompt = \
    #     """
    #     For the course: {course_name_domain}, provide rich content for the keyword: {keyword}.
    #     {keyword}'s definition is: {definition}.
        
    #     Generate rich content based on the given context as below:
    #     Context to extract keyword definition: {text}.
    #     Max words for rich content: {max_words_rich_content}
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