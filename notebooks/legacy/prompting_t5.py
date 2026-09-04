
CODING_FORMAT_PROMPTING = """
def extract_quadruplet_list_from_review(input_review):
    \"\"\"Extract list of quadruplet sentiments from the input review for the {domain_name} domain. \"\"\"

    input_review = "{input_review}"

    quadruplet_list = []
    for phrase in extract_important_phrase(input_review):
        category = classify_category(phrase)
        aspect = extract_aspect_for_category(phrase, category)  # mentioned in review or implicit aspect
        sentiment = classify_sentiment_for_aspect(phrase, aspect, category)  # positive, negative, or neutral
        opinion = extract_opinion_for_sentiment(phrase, sentiment)  # mentioned in review or implicit opinion
        quadruplet_list.append({{
            "category": category,
            "aspect": aspect,
            "sentiment": sentiment,
            "opinion": opinion
        }})
    return quadruplet_list

# Run an example
input_review= "{input_review}"
results = extract_quadruplet_list_from_review(input_review)
for quad in results:
    print(f"quadruplet_list.append({{quad}})")<end_of_turn>
"""

PARAPHRASE_FORMAT_PROMPTING = """<start_of_turn>user
Please generate a paragraph that includes quadruplet sentiment information extracted from a given Vietnamese restaurant review.
+ Input review: "{input_review}".
+ Output: <end_of_turn>
<start_of_turn>model
"""

FREE_ORDER_FORMAT_PROMPTING = """<start_of_turn>user
Please extract all the quadruples in the json format (aspect category, aspect term, sentiment label, opinion term) from a given Vietnamese review for the restaurant domain.

+ Input review: "{input_review}".
+ Output: <end_of_turn>
<start_of_turn>model
"""

JSON_FORMAT_PROMPTING = """<start_of_turn>user
Extract all aspect-opinion quadruples from the following Vietnamese restaurant review and present them in this exact format:
AT: aspect term AC: aspect category SP: sentiment polarity OT: opinion term

+ Input review: "{input_review}".
+ Output: <end_of_turn>
<start_of_turn>model
"""