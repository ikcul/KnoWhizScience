import re
import os
import json
import sys
import zipfile
import sqlite3
import pandas as pd
import markdownify
from anki_export import ApkgReader
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from helper.azure_blob import AzureBlobHelper 

class Page:
    def __init__(self, content, anki_content=None):
        self.page_content = content
        self.metadata = {}
        self.anki_content = anki_content

class AnkiLoader:
    def __init__(self, full_file_path):
        self.full_file_path = full_file_path

        # Extract directory path
        directory_path = os.path.dirname(self.full_file_path)
        # Extract file name without extension
        file_name = os.path.splitext(os.path.basename(self.full_file_path))[0]

        # Create the dictionary
        self.para = {
            "Anki_file_path": directory_path,
            "Anki_file_name": file_name
        }

    def load(self):
        anki_json = Anki2Json(self.para)
        json_data = anki_json.convert_apkg_to_json()
        
        # Transform json_data into a list of Page objects with a 'page_content' attribute
        structured_data = []
        for item in json_data:
            # print("item: ", str(item))
            structured_data.append(Page(str(item), item))

        # print("Structured data loaded successfully: ", (structured_data[0].page_content))
        return structured_data

class Anki2Json:
    def __init__(self, para = None):
        self.file_path = para["Anki_file_path"]
        self.file_name = para["Anki_file_name"]

        # TEST
        print("\nAnki file name: ", self.file_name)

        self.apkg_path = os.path.join(self.file_path, self.file_name + ".apkg")
        self.json_output_path = os.path.join(self.file_path, self.file_name + ".json")
        self.html_output_path = os.path.join(self.file_path, self.file_name + ".html")

    def _clean_text(self, text):
        if text is None:
            return ""
        clean_text = re.sub(r'\{\{c\d+::(.*?)\}\}', r'\1', text)
        return clean_text

    def _is_cloze(self, text):
        return bool(re.search(r'\{\{c\d+::.*?\}\}', text))
    
    def _adjust_asterisks(self, text):
        """
        Adjusts the position of double asterisks in the given text so that they
        enclose only the content before the last '::' in each match.

        Parameters:
        text (str): The input string containing double asterisks and '::' patterns.

        Returns:
        str: The adjusted string with corrected asterisk placement.
        """
        pattern = r'\*\*(.+?)\*\*'  # Matches content enclosed by '**' and '**'

        def repl(match):
            inner_text = match.group(1)
            if '::' in inner_text:
                idx = inner_text.rfind('::')  # Find the last '::'
                before_colon = inner_text[:idx].rstrip()
                after_colon = inner_text[idx+2:].lstrip()
                # Reconstruct the string with adjusted asterisks
                new_text = '**' + before_colon + '**::' + after_colon
                return new_text
            else:
                return match.group(0)  # No adjustment needed

        result = re.sub(pattern, repl, text)
        return result
    
    def _display_anki_front(self, input_text, number):
        input_text = self._adjust_asterisks(input_text)
        pattern = re.compile(r'{{c\d+::(.*?)(::(.*?))?}}', re.DOTALL)
        output = ''
        last_end = 0
        cloze_count = 0
        for match in pattern.finditer(input_text):
            start, end = match.span()
            content = match.group(1)
            label = match.group(3) if match.group(3) else '...'
            cloze_count += 1
            # Append text before the cloze deletion
            output += input_text[last_end:start]
            if cloze_count == number:
                # Mask the Nth cloze deletion
                output += f'**[{label}]** '
            else:
                # Reveal other cloze deletions
                output += content
            last_end = end
        # Append any remaining text after the last cloze deletion
        output += input_text[last_end:]
        return output.strip()

    def _display_anki_back(self, input_text, number):
        input_text = self._adjust_asterisks(input_text)
        pattern = re.compile(r'{{c\d+::(.*?)(::(.*?))?}}', re.DOTALL)
        # Reveal all cloze deletions
        back_content = pattern.sub(lambda m: m.group(1), input_text)
        return back_content.strip()
    
    def _remove_tags_or_deck_at_end(self, strings):
        # List of substrings to remove, but only if they are at the end of the list
        remove_list = ["tags", "deck", "audio", ""]
        
        # Check and remove elements from the end if they are "tags" or "deck"
        while strings and strings[-1] in remove_list:
            strings.pop()

        return strings

    def convert_apkg_to_json(self):
        extract_path = self.apkg_path + "_extracted"
        with ApkgReader(self.apkg_path) as apkg:
            data = apkg.export()
            self.anki_chapters = list(data.keys())
            self.anki_titles_list = []
            for chapter in self.anki_chapters:
                self.anki_titles_list.append(data[chapter][0])
            self.anki_titles = data[list(data.keys())[0]][0]
            self.anki_titles = self._remove_tags_or_deck_at_end(self.anki_titles)
            print("Anki titles list: ", self.anki_titles_list)
            print("Anki titles: ", self.anki_titles)
        with zipfile.ZipFile(self.apkg_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        # Rename numbered media files
        for root, _, files in os.walk(extract_path):
            for file in files:
                # If the file is media file, rename it to .json
                if file.startswith("media"):
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.splitext(old_file_path)[0] + '.json'
                    os.rename(old_file_path, new_file_path)
        media_dict_path = os.path.join(extract_path, 'media.json')
        with open(media_dict_path, 'r') as file:
            media_dict = json.load(file)
        # Create a reverse dictionary for media files
        reverse_media_dict = {
            v: (k + '.' + v.split('.')[-1] if len(v.split('.')) > 1 else k + '.jpg')
            for k, v in media_dict.items()
        }
        def convert_dict_strings(input_dict):
            converted_dict = {f'{key}': f'{value}' for key, value in input_dict.items()}
            return converted_dict
        reverse_media_dict = convert_dict_strings(reverse_media_dict)
        path = os.path.join(extract_path, 'reverse_media_dict.json')
        with open(path, 'w') as file:
            json.dump(reverse_media_dict, file, indent=2)

        # Rename numbered media files in the saved directory
        for root, _, files in os.walk(extract_path):
            for file in files:
                # If the file is numbered, rename it to .jpg
                if file.isdigit():
                    old_file_path = os.path.join(root, file)
                    # print("Test: ", type(media_dict[str(file)]))
                    # print("Test: ", media_dict[str(file)])
                    if(len(media_dict[str(file)].split('.')) > 1):
                        new_file_path = os.path.splitext(old_file_path)[0] + "." + media_dict[str(file)].split('.')[-1]
                    else:
                        new_file_path = os.path.splitext(old_file_path)[0] + ".jpg"
                    os.rename(old_file_path, new_file_path)

        # Split the content_list into question and answer
        json_data = []
        for chapter in self.anki_chapters:
            titles = data[chapter][0]
            titles = self._remove_tags_or_deck_at_end(titles)
            content_lists = data[chapter][1:]
            for content_list in content_lists:
                content_list = self._remove_tags_or_deck_at_end(content_list)
                min_length = min(len(content_list), len(titles))
                if(min_length > 2):
                    question_html = f"<br><h3>{titles[0]}:</h3> {content_list[0]}<br>"
                else:
                    question_html = f"<br>{content_list[0]}<br>"
                answer_html = ""
                if(min_length > 2):
                    for i in range(1, min_length):
                        answer_html += f"<br><h3>{titles[i]}:</h3> {content_list[i]}<br>"
                else:
                    for i in range(1, min_length):
                        answer_html += f"<br>{content_list[i]}<br>"

                # Add the formatted note to json_data
                json_data.append({
                    "Question": question_html,
                    "Answer": answer_html.strip()  # Remove the trailing <br> if present
                })

        # # Split the fields into question and answer (assuming the fields are separated by a delimiter '\x1f')
        # db_path = os.path.join(extract_path, 'collection.anki2')
        # conn = sqlite3.connect(db_path)
        # cursor = conn.cursor()

        # cursor.execute("SELECT id, flds FROM notes;")
        # notes = cursor.fetchall()
        # json_data = []
        # for note in notes:
        #     note_id = note[0]
        #     fields = note[1].split('\x1f')  # Anki typically separates fields with this delimiter
        #     fields = self._remove_tags_or_deck_at_end(fields)
            
        #     if len(fields) >= 1:
        #         # Construct the question and answer in HTML format
        #         question_html = f"<br><h3>{self.anki_titles[0]}:</h3> {fields[0]}<br>"
                
        #         # Construct the answer by adding titles before each of the remaining fields
        #         answer_html = ""
        #         min_length = min(len(fields), len(self.anki_titles))
        #         for i in range(1, min_length):
        #             answer_html += f"<br><h3>{self.anki_titles[i]}:</h3> {fields[i]}<br>"
                
        #         # Add the formatted note to json_data
        #         json_data.append({
        #             "id": note_id,
        #             "Question": question_html,
        #             "Answer": answer_html.strip()  # Remove the trailing <br> if present
        #         })

        # # Close the database connection
        # conn.close()

        # Write the Q&A data to a JSON file
        with open(self.json_output_path, "w", encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)

        # Load the flashcards and media dictionary
        with open(self.json_output_path, 'r') as file:
            flashcards = json.load(file)

        # Function to replace media links
        def replace_media_links(text):
            # Define a set of valid image extensions
            valid_extensions = {'.jpg', '.png', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.ico', '.tiff', '.mp4', 'mp3'}
            # Iterate over the media dictionary
            for original, replacement in reverse_media_dict.items():
                # Check if the original media link ends with a valid image extension
                if any(original.endswith(ext) for ext in valid_extensions):
                    text = text.replace(original, replacement)
            return text
        
        # Update the flashcards
        for flashcard in flashcards:
            if(self._is_cloze(flashcard['Question']) is False):
                question = replace_media_links(flashcard['Question'])
                flashcard['Question'] = markdownify.markdownify(question, heading_style="ATX")

                answer = replace_media_links(flashcard['Answer'])
                flashcard['Answer'] = markdownify.markdownify(answer, heading_style="ATX")
            else:   # If the flashcard is a cloze deletion
                question = replace_media_links(flashcard['Question'])
                question = markdownify.markdownify(question, heading_style="ATX")
                flashcard['Question'] = self._display_anki_front(question, 1)

                answer = replace_media_links(flashcard['Answer'])
                answer = markdownify.markdownify(answer, heading_style="ATX")
                flashcard['Answer'] = self._display_anki_back(question, 1) + "\n" + answer

                # question = replace_media_links(flashcard['Question'])
                # flashcard['Question'] = markdownify.markdownify(question, heading_style="ATX")

                # answer = replace_media_links(flashcard['Answer'])
                # flashcard['Answer'] = markdownify.markdownify(answer, heading_style="ATX")
                
        # Save the updated flashcards to a new JSON file
        updated_flashcards_path = self.json_output_path
        with open(updated_flashcards_path, 'w') as file:
            json.dump(flashcards, file, indent=2)

        return flashcards