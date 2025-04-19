from pipeline.science.prompts.zeroshot_flashcards_prompts import ZeroshotPrompts

class CSDS_ZeroshotPrompts(ZeroshotPrompts):
    """
    This class is used to generate prompts for the CS and DS domain
    •    Computer Science
    •    Data Science
    """
    @staticmethod
    # Step 5: Flashcards expansion generation
    def flashcards_expansion_generation_prompt():
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
        3. Within the example, provide a step-by-step breakdown only if it significantly enhances memorization and understanding of the concept.
        4. Within the example, if you need to display formulas, include them in LaTeX syntax formatted in markdown, as shown below:
            ----------------
            $$
            \frac{{a}}{{b}} = \frac{{c}}{{d}}
            $$
            ----------------
        5. Within the example, if you need to display tables, format them using markdown as follows:
            ----------------
            ## Table

            | Header 1   | Header 2   | Header 3   |
            |------------|------------|------------|
            | Row 1 Col 1| Row 1 Col 2| Row 1 Col 3|
            | Row 2 Col 1| Row 2 Col 2| Row 2 Col 3|
            | Row 3 Col 1| Row 3 Col 2| Row 3 Col 3|
            ----------------

        6. Do not include "```markdown" in the response. Final whole response must be in correct markdown format.
        7. Specify the text with intuitive markdown syntax like bold, italic, etc, bullet points, etc.
        8. For in-line formulas, use the syntax: $E = mc^2$. Remember must use double ```$``` for display formulas.
        """
        return prompt
    # def flashcards_expansion_generation_prompt():
    #     """
    #     Prompt for generating flashcards expansions for each keyword.
    #     """
    #     prompt = \
    #     """
    #     For the course: {course_name}, chapter: {chapter_name},
    #     {keyword}'s definition in this course is: {definition},
    #     Provide the more detailed explanation in addition to its definition.

    #     Key requirements:
    #     1. The course is about Data Science & Computer Science,
    #         organize the expansions in the style of a wiki page,
    #         and try to include more code snippets in Markdown syntax.
    #     2. The wiki page should focus more on practical and hands-on oriented content, do not talk about conceptional exlanations.
    #     3. The wiki page should be at most 150 words long.
    #     4. For code snippets, must use triple backticks "```" for code blocks so we have nice display in markdown.
    #     """
    #     return prompt
