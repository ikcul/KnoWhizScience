import logging
import sys
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import OutputFixingParser

# Dynamically adjust imports to work either as a module or direct script
try:
    from knowhizService.pipeline.science.api_handler import ApiHandler
except ModuleNotFoundError:
    # For direct script execution
    # Add the root directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
    if root_dir not in sys.path:
        sys.path.append(root_dir)
    from knowhizService.pipeline.science.api_handler import ApiHandler

from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger("kzpipeline.science.regen_flashcard")


def generate_improved_flashcard_content(flashcard_content, feedback, title):
    """
    Generate improved flashcard content based on feedback using LLM.
    
    Args:
        flashcard_content (dict): Original flashcard content with 'answer' and 'expandedAnswer'
        feedback (str): User feedback for improvement
        title (str): Flashcard question/title
        
    Returns:
        dict: New flashcard content with 'answer' and 'expandedAnswer'
    """
    # Implement the logic to call the existing Python pipeline
    # and get the response from the LLM
    para = {
        'llm_source': 'openai',  # or 'anthropic'
        'temperature': 0.2,
        "creative_temperature": 0.5,
        "openai_key_dir": ".env",
        "anthropic_key_dir": ".env",
    }
    load_dotenv(para['openai_key_dir'])
    api = ApiHandler(para)
    llm = api.models['advance']['instance']
    parser = JsonOutputParser()
    error_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    
    # Step 1: Planning - Analyze feedback and propose improvement strategies
    logger.info(f"Analyzing feedback: {feedback}\nfor flashcard content: {flashcard_content}")
    planning_prompt = """
        You are an educational content improvement specialist.
        Analyze the following student feedback for a flashcard and create a detailed improvement plan.
        
        Original flashcard:
        - Title: "{title}"
        - Current answer: "{flashcard_content}"
        
        Student feedback: "{feedback}"
        
        Create a detailed analysis of:
        1. What specific issues the feedback identifies
        2. What aspects of the content need improvement
        3. Specific strategies to address each issue
        4. Concrete suggestions for new information, examples, or explanations to add
        
        Respond with a JSON containing:
        ```json
        {{
            "analysis": "Your analysis of the feedback and content issues",
            "improvement_strategies": ["List of specific improvement strategies"],
            "content_suggestions": ["List of specific content additions or changes to make"]
        }}
        ```
    """
    planning_prompt_template = ChatPromptTemplate.from_template(planning_prompt)
    planning_chain = planning_prompt_template | llm | error_parser
    improvement_plan = planning_chain.invoke({
        'flashcard_content': flashcard_content, 
        'feedback': feedback, 
        'title': title
    })
    
    logger.info(f"Improvement plan generated: {improvement_plan}")
    
    # Step 2: Implementation - Generate new flashcard content based on the improvement plan
    prompt = """
        You are an educational and patient professor teaching a course in college.
        Your task is to improve and modify a flashcard based on student feedback and a detailed improvement plan.

        Original flashcard:
        - Title: "{title}"
        - Current answer: "{flashcard_content}"
        
        Student feedback: "{feedback}"
        
        Improvement plan:
        - Analysis: {improvement_plan_analysis}
        - Improvement strategies: {improvement_plan_strategies}
        - Content suggestions: {improvement_plan_suggestions}

        Instructions:
        1. Follow the improvement plan to address the student's feedback
        2. Create a NEW version of the answer that:
           - Directly addresses the feedback points
           - Adds new information not present in the original
           - Uses different examples or explanations
           - Is more detailed or clearer than the original
        3. Make sure the new content is substantially different from the original
        4. Keep the core concept accurate but present it in a fresh way

        Respond with a JSON containing two fields:
        - "answer": A concise definition in markdown format that is different from the original
        - "expandedAnswer": A detailed explanation in markdown format that incorporates new examples and perspectives

        ```json
        {{
            "answer": <A concise definition in markdown format that is different from the original>,
            "expandedAnswer": <A detailed explanation in markdown format that incorporates new examples and perspectives>
        }}
        ```

        The new content must be noticeably different from: "{flashcard_content}"
        """
    prompt = ChatPromptTemplate.from_template(prompt)
    chain = prompt | llm | error_parser
    new_flashcard_content = chain.invoke({
        'flashcard_content': flashcard_content, 
        'feedback': feedback, 
        'title': title,
        'improvement_plan_analysis': improvement_plan.get('analysis', ''),
        'improvement_plan_strategies': ', '.join(improvement_plan.get('improvement_strategies', [])),
        'improvement_plan_suggestions': ', '.join(improvement_plan.get('content_suggestions', []))
    })
    
    # Log comparison between old and new content
    logger.info("Content Comparison:")
    logger.info("Original Flashcard:")
    logger.info(f"- Answer: {flashcard_content.get('answer', '')}")
    logger.info(f"- Expanded Answer: {flashcard_content.get('expandedAnswer', '')}")
    logger.info("\nNew Flashcard:")
    logger.info(f"- Answer: {new_flashcard_content.get('answer', '')}")
    logger.info(f"- Expanded Answer: {new_flashcard_content.get('expandedAnswer', '')}")
    
    return new_flashcard_content

if __name__ == "__main__":
    import json
    import os
    from pathlib import Path
    
    # Get the repository root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
    
    # Path to the source flashcards JSON file
    source_file = os.path.join(
        root_dir, 
        "knowhizService/pipeline/test_outputs/flashcards/cf84063afd8a01e05be322cb02854d5bf5cabdda69bd13b28ce26b5d/flashcards_set0.json"
    )
    
    # Check if file exists
    if not os.path.exists(source_file):
        print(f"File not found: {source_file}")
        # Try alternative path if direct execution
        alt_source_file = os.path.join(
            os.path.dirname(current_dir),
            "test_outputs/flashcards/cf84063afd8a01e05be322cb02854d5bf5cabdda69bd13b28ce26b5d/flashcards_set0.json"
        )
        if os.path.exists(alt_source_file):
            source_file = alt_source_file
            print(f"Using alternative file path: {source_file}")
        else:
            print(f"Alternative file not found either: {alt_source_file}")
            sys.exit(1)
    
    # Create output file path
    output_dir = os.path.dirname(source_file)
    output_file = os.path.join(output_dir, "regenerated_flashcards.json")
    
    print(f"Source file: {source_file}")
    print(f"Output file: {output_file}")
    
    # Generic feedback to use for testing
    test_feedback = "Please make the explanation more detailed and include more examples. Also, simplify the language to make it easier to understand."
    
    # Load the flashcards
    with open(source_file, 'r') as f:
        flashcards_data = json.load(f)
    
    # Process each flashcard
    regenerated_flashcards = {}
    total_flashcards = len(flashcards_data)
    
    print(f"Starting regeneration of {total_flashcards} flashcards")
    
    for i, (title, content) in enumerate(flashcards_data.items()):
        print(f"Processing flashcard {i+1}/{total_flashcards}: {title}")
        
        # Extract the necessary information
        flashcard_content = {
            "answer": content.get("definition", ""),
            "expandedAnswer": content.get("expansion", "")
        }
        
        # Generate improved content
        try:
            new_content = generate_improved_flashcard_content(flashcard_content, test_feedback, title)
            
            # Create a new flashcard object with updated content
            regenerated_flashcards[title] = {
                "definition": new_content["answer"],
                "expansion": new_content["expandedAnswer"]
            }
            print(f"Successfully regenerated flashcard {i+1}")
        except Exception as e:
            print(f"Error regenerating flashcard {i+1}: {str(e)}")
            # Keep the original content if regeneration fails
            regenerated_flashcards[title] = content
    
    # Save the regenerated flashcards
    with open(output_file, 'w') as f:
        json.dump(regenerated_flashcards, f, indent=2)
    
    print(f"Regeneration complete. Saved to {output_file}")