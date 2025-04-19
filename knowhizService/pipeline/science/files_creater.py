import json
import os

def convert_json_to_html(course_name, json_file_path, output_file_path):
    """
    Converts a JSON file (current flashcards JSON files) with HTML content to a single HTML file.

    Key variables required:
    - course_name, the name of the course

    Key variables changed:
    - N/A

    Key files required:
    - path = json_file_path, JSON file with flashcards data

    Key files changed/created:
    - path = output_file_path, html file for reviewing the flashcards
    """

    # Load the JSON data from the file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)

    # Start the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{course_name}</title>""" + \
    """
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                vertical-align: top;
            }
            th {
                background-color: #f2f2f2;
                color: #2e6da4;
            }
            h2 {
                color: #2e6da4;
                margin-top: 0;
            }
        </style>
    </head>""" + \
    f"""
    <body>
        <h1>{course_name}</h1>
        <table>
            <thead>
                <tr>
                    <th>Concept</th>
                    <th>Content</th>
                </tr>
            </thead>
            <tbody>
    """
    # Append each concept and its content to the HTML
    for concept, content in data_dict.items():
        html_content += f"""
                <tr>
                    <td><h2>{concept}</h2></td>
                    <td>{content}</td>
                </tr>
        """

    # End the HTML content
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Write the HTML content to the output file
    with open(output_file_path, "w", encoding='utf-8') as file:
        file.write(html_content)
    print(f"HTML file successfully created at: {output_file_path}")
