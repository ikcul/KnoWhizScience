import hashlib
import os
from urllib.parse import urlparse
from pipeline.config.config import Config
import logging

config = Config()
file_path_prefix = config.get_file_path_prefix()

def gen_config(global_file_dir=None):
    if not global_file_dir:
        global_file_dir = file_path_prefix

    return {
        "global_file_dir": global_file_dir,
        "llm_source": "openai",
        "temperature": 0,
        "openai_key_dir": "pipeline/.env",
        "book_dir": global_file_dir + "pipeline/test_inputs/",
        "course_id_mapping_file": global_file_dir + "pipeline/test_outputs/course_id_mapping.json",
        "results_dir": global_file_dir + "pipeline/test_outputs/",
        "keywords_per_chapter": 10,
        "flashcards_set_size": 10,
        "quality_check_size": 50,
        "max_flashcards_size": 300,
        "max_flashcard_definition_words": 50,
        "max_flashcard_expansion_words": 200,
        "max_quiz_questions_per_section": 10,
        "quiz_random_seed": 5,
        "max_test_multiple_choice_questions_per_section": 1,
        "max_test_short_answer_questions_per_section": 1,
        "creative_temperature": 0.5,
        "regions": ["Example"],    # Regions for flashcards expansion
        "definition_detail_level": 0,   # 0: no detail, 1: medium detail, 2: high detail
        "expansion_detail_level": 0,    # 0: no detail, 1: medium detail, 2: high detail
        "rich_content": False,
        "options_list": ["Mindmap", "Table", "Formula", "Code"],
        "generate_quiz": True,
        "generate_test": True,
        "save_to_mongo": True,
    }

def zero_shot_flashcards_para(course_description, pipeline_id, user_id, course_id, course_generation_type="ZEROSHOT", discord_bot:bool=False, discord_channel_id:int=1, discord_mention:str=""):
    global_file_dir = file_path_prefix
    para = {
        "zero_shot": True,
        "course_info": course_description,
        "pipeline_id": pipeline_id,
        "user_id": user_id,
        "course_id": course_id,
        "course_generation_type": course_generation_type,
        "discord_bot": discord_bot,
        "discord_channel_id": discord_channel_id,
        "discord_mention": discord_mention,
    }
    para.update(gen_config(global_file_dir=global_file_dir))
    return para


def flashcards_para(main_filenames:str, pipeline_id, full_material_url, user_id, course_id, course_generation_type="OTHER", supplementary_filenames=[], full_supplementary_material_urls=[], discord_bot:bool=False, discord_channel_id:int=1, discord_mention:str=""):
    global_file_dir = file_path_prefix
    # main_filenames_without_ext = str(hashlib.md5(main_filenames.encode('utf-8')).hexdigest())
    if full_material_url.startswith("blob:") and not is_valid_url(full_material_url):
        main_base_filename, main_ext = os.path.splitext(main_filenames)
        main_filenames_with_ext = str(hashlib.md5(main_base_filename.encode('utf-8')).hexdigest()) + main_ext
    else:   #   full_material_url.startswith("link:")
        main_filenames_with_ext = main_filenames  # Direcly use the URL

    supplementary_filenames_with_ext = None
    for supplementary_filename in supplementary_filenames:
        if isinstance(supplementary_filename, str) and supplementary_filename != "":
            if supplementary_filenames_with_ext == None:
                supplementary_filenames_with_ext = []
            if not is_valid_url(supplementary_filename):
                supplementary_base_filename, supplementary_ext = os.path.splitext(supplementary_filename)

                supplementary_hashed_filename = str(hashlib.md5(supplementary_base_filename.encode('utf-8')).hexdigest()) + supplementary_ext
                supplementary_filenames_with_ext.append(supplementary_hashed_filename)
            else:
                supplementary_filenames_with_ext.append(supplementary_filename)

    para = {
        "zero_shot": False,
        "main_filenames": [ main_filenames_with_ext ],
        "supplementary_filenames": supplementary_filenames_with_ext,
        "chunk_size": 2000,
        "similarity_score_thresh": 0.8,
        "num_context_pages": 50,
        "keywords_per_page": 1.5,
        "page_set_size": 5,
        "overlapping": 0,
        "link_flashcards_size": 30,
        "pipeline_id": pipeline_id,
        "material_url": full_material_url,
        "supplementary_material_urls": full_supplementary_material_urls,
        "user_id": user_id,
        "course_id": course_id,
        "course_generation_type": course_generation_type,
        "discord_bot": discord_bot,
        "discord_channel_id": discord_channel_id,
        "discord_mention": discord_mention,
    }
    para.update(gen_config(global_file_dir=global_file_dir))
    return para

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
