import pandas as pd
import json
import ast
import os
from pathlib import Path
import re
from preprocessing import *


def calculate_max_input_length(tokenizer, model_input, model_output):  
    enc_inp = tokenizer(model_input, add_special_tokens=True, padding=False, truncation=False, return_length=True)
    enc_out = tokenizer(model_output, add_special_tokens=True, padding=False, truncation=False, return_length=True)
    max_input_length = max(enc_inp["length"])
    max_output_length = max(enc_out["length"])
    return max_input_length, max_output_length

def reorder_quadruplets(quadruplets):
    """
    Reorder each quadruplet from (category, aspect, sentiment, opinion)
    to (aspect, category, sentiment, opinion).
    """
    reordered = []
    for quad in quadruplets:
        if len(quad) == 4:
            reordered.append([preprocessing_text(quad[1]), preprocessing_text(quad[0]), preprocessing_text(quad[2]), preprocessing_text(quad[3])])
    return reordered



def parse_absa_line(line, reorder=True): 
    """
    Parse a single line from the ABSA dataset.
    Format: text####[[quadruplet1], [quadruplet2], ...]
    reorder to switch from (category, aspect, sentiment, opinion)
    to (aspect, category, sentiment, opinion).
    """
    if '####' not in line:
        return None, None
    
    parts = line.strip().split('####')
    text = parts[0].strip()
    
    try:
        quadruplets = ast.literal_eval(parts[1].strip())
        if reorder:
            quadruplets = reorder_quadruplets(quadruplets)
        return text, quadruplets
    except:
        print(f"Error parsing line: {line}")
        return None, None

def create_output_format(quadruplets):
    """Create the output format: [{category, aspect, sentiment, opinion}]"""
    if not quadruplets:
        return "[]"
    
    result = []
    for quad in quadruplets:
        if len(quad) >= 4:
            result.append(f"{{preprocessing_text({quad[0]}), preprocessing_text({quad[1]}), preprocessing_text({quad[2]}), preprocessing_text({quad[3]})}}")
    
    return';'.join(result)

def create_coding_format(quadruplets):
    """Create the coding format with quadruplet_list.append statements"""
    if not quadruplets:
        return ""
    
    result = []
    for quad in quadruplets:
        if len(quad) >= 4:
            category, aspect, sentiment, opinion = preprocessing_text(quad[0]), preprocessing_text(quad[1]), preprocessing_text(quad[2]), preprocessing_text(quad[3])
            result.append(f'quadruplet_list.append({{"category": "{category}", "aspect": "{aspect}", "sentiment": "{sentiment}", "opinion": "{opinion}"}})')
    
    return '\n'.join(result)

def create_json_format(quadruplets):
    """Create the JSON format"""
    if not quadruplets:
        return "[]"
    
    result = []
    for quad in quadruplets:
        if len(quad) >= 4:
            category, aspect, sentiment, opinion = quad[0], quad[1], quad[2], quad[3]
            result.append({
                "category": preprocessing_text(category),
                "aspect": preprocessing_text(aspect),
                "sentiment": preprocessing_text(sentiment),
                "opinion": preprocessing_text(opinion)
            })
    
    return json.dumps(result)

def create_paraphrase_format(quadruplets):
    """Create the paraphrase format: <category> <sentiment> because <aspect> is <opinion>"""
    if not quadruplets:
        return ""
    
    result = []
    for quad in quadruplets:
        if len(quad) >= 4:
            category, aspect, sentiment, opinion = quad[0], preprocessing_text(quad[1]), quad[2], preprocessing_text(quad[3])
            
            # Handle NULL/null values for natural language
            if category.lower() == 'null':
                category_text = "implicit category"
            else:
                category_text = category
                
            if opinion.lower() == 'null':
                opinion_text = "implicit opinion"
            else:
                opinion_text = preprocessing_text(opinion)
            
            result.append(f"{category_text} {sentiment} because {aspect} is {opinion_text}")
    
    return ' ssep '.join(result)

def create_free_order_format(quadruplets):
    """Create the free order format: [AT] aspect [AC] category [SP] sentiment [OP] opinion"""
    if not quadruplets:
        return ""
    
    result = []
    for quad in quadruplets:
        if len(quad) >= 4:
            category, aspect, sentiment, opinion = quad[0], preprocessing_text(quad[1]), quad[2], preprocessing_text(quad[3])
            result.append(f"[AT] {aspect} [AC] {category} [SP] {sentiment} [OP] {opinion}")
    
    return ' ; '.join(result)

def process_txt_file(file_path, reorder=True):
    """Process a single txt file and return a list of processed records"""
    records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            text, quadruplets = parse_absa_line(line, reorder=reorder)
            
            if text is None:
                print(f"Skipping line {line_num} in {file_path}")
                continue
            
            record = {
                'input': preprocessing_text(text),
                'output': create_output_format(quadruplets),
                'paraphrase_format': create_paraphrase_format(quadruplets),
                'free_order': create_free_order_format(quadruplets),
                'json_format': create_json_format(quadruplets),
                'coding_format': create_coding_format(quadruplets)
            }
            
            records.append(record)
    
    return records

def main(base_path, reorder=True):
    """
    Main function to process all txt files and create CSV files. 
    Reorder=False only with vietnamese ABSA datasets.
    """
    
    txt_path = base_path / "txt"
    csv_path = base_path / "csv"
    
    csv_path.mkdir(parents=True, exist_ok=True)
    
    files_to_process = ['train.txt', 'dev.txt', 'test.txt']
    
    for file_name in files_to_process:
        txt_file_path = txt_path / file_name
        
        if not txt_file_path.exists():
            print(f"File {txt_file_path} does not exist")
            continue
        
        print(f"Processing {file_name}...")
        
        records = process_txt_file(txt_file_path, reorder=reorder)
        
        df = pd.DataFrame(records)
        
        csv_file_name = file_name.replace('.txt', '.csv')
        csv_file_path = csv_path / csv_file_name
        
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
        
        print(f"Created {csv_file_path} with {len(records)} records")
        
        # Verify
        print(f"First 3 rows of {csv_file_name}:")
        print(df.head(3))
        print("-" * 80)

# Test function to check a single line
def test_single_line():
    test_line = "their sake list was extensive , but we were looking for purple haze , which was n ' t listed but made for us upon request !####[['sake list', 'drinks style_options', 'positive', 'extensive'], ['NULL', 'service general', 'positive', 'NULL']]"
    test_line = test_line.lower()
    
    text, quadruplets = parse_absa_line(test_line)
    
    print("Testing single line:")
    print(f"Input text: {text}")
    print(f"Quadruplets: {quadruplets}")
    print()
    
    if quadruplets:
        print("Generated formats:")
        print(f"output: {create_output_format(quadruplets)}")
        print(f"coding_format: {create_coding_format(quadruplets)}")
        print(f"json_format: {create_json_format(quadruplets)}")
        print(f"paraphrase_format: {create_paraphrase_format(quadruplets)}")
        print(f"free_order: {create_free_order_format(quadruplets)}")


def parse_paraphrase_format_to_json_format(paraphrase_format: str):
    quadruplet_list = []

    # Split into entries using ' ssep '
    entries = paraphrase_format.replace("*","").replace("�","").strip().split(' ssep ')
    
    for entry in entries:
        if ' because ' not in entry:
            continue

        try:
            category_sentiment, aspect_opinion = entry.split(' because ', 1)
            category, sentiment = category_sentiment.rsplit(' ', 1)

            # Split "aspect is opinion"
            aspect_opinion_parts = aspect_opinion.strip().split(' is ', 1)
            if len(aspect_opinion_parts) != 2:
                continue

            aspect = aspect_opinion_parts[0].strip()
            opinion = aspect_opinion_parts[1].strip()

            quadruplet_list.append({
                'category': category.strip(),
                'aspect': aspect,
                'sentiment': sentiment.strip(),
                'opinion': opinion
            })
        except Exception as e:
            print(f"Skipping invalid entry: {entry} - Error: {e}")

    return quadruplet_list

def parse_coding_format_to_json_format(coding_format_str: str):
    quadruplet_list = []

    # Extract each dictionary from the append(...) calls
    pattern = r"quadruplet_list\.append\((\{.*?\})\)"
    matches = re.findall(pattern, coding_format_str)

    for match in matches:
        try:
            data = ast.literal_eval(match)  # safely parse the dict
            quadruplet_list.append(data)
        except Exception as e:
            print(f"Skipping invalid entry: {match} - Error: {e}")
    return quadruplet_list


def parse_quadruplets_free_order(text: str):
    pattern = re.compile(
        r"\[AT\]\s*(.*?)\s*\[AC\]\s*(.*?)\s*\[SP\]\s*(.*?)\s*\[OP\]\s*(.*?)(?:\s*;|$)",
        re.DOTALL,
    )
    quadruplet_list = []
    for aspect, category, sentiment, opinion in pattern.findall(text):
        quadruplet_list.append({
            "category": category.strip(),
            "aspect": aspect.strip(),
            "sentiment": sentiment.strip(),
            "opinion": opinion.strip(),
        })
    return quadruplet_list


def convert_original_to_json(text: str):
    # Split by semicolon
    items = text.strip().split(";")
    results = []

    for item in items:
        item = item.strip("{} ")
        if not item:
            continue

        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 4:
            continue
            
        category, aspect, sentiment, opinion = parts
        results.append({
            "category": category.lower().strip(),
            "aspect": aspect.lower().strip(),
            "sentiment": sentiment.lower().strip(),
            "opinion": opinion.lower().strip()
        })

    return results



def convert_predict_to_json_format(y_dict,prompt_type):
    list_result = []
    for item in y_dict:
        if prompt_type == "coding_format":
            json_output = parse_coding_format_to_json_format(item)
    
        elif prompt_type == "paraphrase_format":
            json_output = parse_paraphrase_format_to_json_format(item)
        
        elif prompt_type == "free_order":
            json_output = parse_quadruplets_free_order(item)
            
        elif prompt_type == "json_format":
            json_output = item

        elif prompt_type == "original_format":
            json_output = convert_original_to_json(item)
        
        else:
            print("ERROR")
        list_result.append(json_output)
    return list_result