class Prompts:
    @staticmethod
    def subject_options():
        return ['Language', 'Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science', 'Data Science', 'Engineering', \
                'Economics', 'History', 'Geography', \
                'Literature', 'Philosophy', 'Psychology', 'Sociology', 'Political Science', \
                'Law', 'Business', 'Medicine', 'Art', 'Music', 'Sports', 'Others']

    @staticmethod
    # Step 0: System prompt
    def system_prompt():
        """
        System prompt for the LLM.
        """
        prompt = \
        """
        You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.
        """
        prompt = \
        """You are a professional and educational professor in college, writing educational and easy to understand lecture notes for students."""
        return prompt
    
    @staticmethod
    def summary_prompt():
        """
        Prompt for summarizing a given text.
        """
        prompt = \
        """
        Summarization tast:
        From the following text, please summarize every 500 words up to 50 words.
        text section: {text}
        """
        return prompt
    
    @staticmethod
    # Rich content generation options
    def rich_content_generation_prompts_map():
        """
        Prompts map for generating rich content in markdown format.
        """
        prompts_map = {
            "Mindmap": """
                Mermaid mindmap in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ```mermaid
                mindmap
                root((Mind Map))
                    subtopic1(Main Topic 1)
                    subsubtopic1(Sub Topic 1.1)
                    subsubtopic2(Sub Topic 1.2)
                        subsubsubtopic1(Sub Sub Topic 1.2.1)
                    subtopic2(Main Topic 2)
                    subsubtopic3(Sub Topic 2.1)
                    subsubtopic4(Sub Topic 2.2)
                    subtopic3(Main Topic 3)
                    subsubtopic5(Sub Topic 3.1)
                    subsubtopic6(Sub Topic 3.2)
                ```
                ----------------
                """,
            "Table": """
                Tables in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ## Table

                | Header 1   | Header 2   | Header 3   |
                |------------|------------|------------|
                | Row 1 Col 1| Row 1 Col 2| Row 1 Col 3|
                | Row 2 Col 1| Row 2 Col 2| Row 2 Col 3|
                | Row 3 Col 1| Row 3 Col 2| Row 3 Col 3|
                ----------------
                ```
            """,
            "Formula": """
                Formulas in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ## Formulas
                This is an inline formula: $E = mc^2$.

                Here is a display formula:
                $$
                \frac{a}{b} = \frac{c}{d}
                $$

                Inline summation formula: $\sum_{i=1}^n i = \frac{n(n+1)}{2}$.
                ----------------
                """,
            "Code": """
                Code Snippets in Markdwon. Example as below. But remember to replace the content with the actual content about the keyword:
                ----------------
                ## Code block
                Here is a markdown document that includes both inline and block code snippets:

                ```python
                # This function prints a greeting
                def hello_world():
                    print("Hello, World!")

                hello_world()
                ```
                ----------------
                """,
            "Image": "Images in Markdwon"   # No specific format for now
        }
        return prompts_map
    
    @staticmethod
    # Rich content generation options
    def rich_content_generation_options_prompt():
        """
        Prompt for selecting the rich content generation options.
        """
        prompt = \
        """
        In the course: {course_name}, chapter: {chapter_name},
        For keyword: {keyword} what is the most suitable format to illustrate its meaning?
        Answer only one string from the list of options: {option}.
        Do not answer anything other than the options list.
        """
        return prompt
    
    @staticmethod
    # Rich content generation
    def rich_content_generation_prompt():
        """
        Prompt for generating rich content in markdown format.
        """
        prompt = \
        """
        For keyword: {keyword}, refine its illustration content: {content}
        by inserting: {option} in the markdown format at the suitable place in the original content
        to make the content more informative.

        Example template for the response:
        {format}

        Important: 
        Refine the rich format content with the real content of the keyword.
        Final whole response must be in correct markdown format. And please specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.

        Do not include the original version in the response. Only respond the refined version.
        """
        return prompt