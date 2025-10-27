"""
This file contains the model prompts for training and inference time.
"""

RLRF_MODEL_INSTRUCTION_WITH_PREV_DOCUMENT = """[INST] You are participating in a search engine optimization (SEO) process.
Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest reward
(1 being the highest).
Focus on editing the most impactful sentences to enhance ranking potential.
Target an edited document length of around 147 words, not exceeding 150 words.
Ensure the edited document is very similar to the candidate document.
Generate only the edited document, without additional comments or titles.
<user> Input:
 - Candidate Query: {}

 - Candidate Document: {}
 
  - Edited Document: [/INST]
"""

RLRF_MODEL_INSTRUCTION_without_prev_document = """[INST] You are participating in a search engine optimization (SEO) process.
You need to create a document to achieve high ranking for the candidate query.
Target a document length of around 147 words, not exceeding 150 words.
Generate only the document, without additional comments or titles.
<user> Input:
 - Candidate Query: {}
 
 - Document: [/INST]
"""

RLRF_MODEL_INSTRUCTION_WITH_PREV_DOCUMENT_full_game_knowledge = """[INST] You are participating in a search engine optimization (SEO) process.
Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest reward
(1 being the highest).
Focus on editing the most impactful sentences to enhance ranking potential.
Target an edited document length of around 147 words, not exceeding 150 words.
Ensure the edited document is very similar to the candidate document.
Generate only the edited document, without additional comments or titles.
<user> Input:
 - Candidate Query: {}

 - Rival Document: {}
 latest ranking: {}
 
 - Your Candidate Document: {}
 latest ranking: {}
 
  - Edited Document: [/INST]
"""

RLRF_MODEL_INSTRUCTION_WITH_PREV_DOCUMENT_for_Llama = """<|start_header_id|>system<|end_header_id|>
Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest).
Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits.
Focus on editing the most impactful sentences to enhance ranking potential.
Target an edited document length of around 147 words, not exceeding 150 words.
Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.

 Input:

- Candidate Query: {}

- Candidate Document:  {}

<|start_header_id|>user<|end_header_id|>

Edited Document: <|start_header_id|>assistant<|end_header_id|>"""

RLRF_MODEL_INSTRUCTION_without_prev_document_for_Llama = """<|start_header_id|>system<|end_header_id|>
Create a document to achieve high ranking for the candidate query, aiming for the highest rank (1 being the highest).
Focus on editing the most impactful sentences to enhance ranking potential.
Target an edited document length of around 147 words, not exceeding 150 words.
Generate only a document, without additional comments or titles.
<|start_header_id|>user<|end_header_id|>
Input:

 - Candidate Query: {}

Document: <|start_header_id|>assistant<|end_header_id|>"""

RLRF_MODEL_INSTRUCTION_WITH_PREV_DOCUMENT_full_game_knowledge_for_Llama = """<|start_header_id|>system<|end_header_id|>
Edit the candidate document to improve its search engine ranking for the candidate query, aiming for the highest rank (1 being the highest). Use the black box search engine's past rankings over various queries, provided as context by the user, to guide your edits. Focus on editing the most impactful sentences to enhance ranking potential. Target an edited document length of around 147 words, not exceeding 150 words. Ensure the edited document is very similar to the candidate document. Generate only the edited document, without additional comments or titles.

 Input:

- Candidate Query: {}

- Candidate Document:  {}
<|start_header_id|>user<|end_header_id|>
query: {}

* documents ordered by latest ranking from highest to lowest in relation to the query:  

* Document: {}

Edited Document: <|start_header_id|>assistant<|end_header_id|>
"""

RM_INSTRUCTION_1_DOC = ("""query: {}
                            document: {}
                              """)

Difference_RM_INSTRUCTION = ("""query: {};
                              Response 1: {};
                                Response 2: {}
                              """)

RM_MODEL_INSTRUCTION_1_DOC_FOR_BERT = ("""query: {}
                                        [SEP] document: {}
                                        """)
