import json

with open('data/dataset_docs_public.json', 'r') as f:
    data = json.load(f)

for item in data.get('search_results', []):
    # Si existe 'question', lo movemos a 'question_str', si no, creamos un string vacío
    item['question_str'] = item.pop('question', "") 
    
    # Aseguramos que cada fuente tenga 'character_index'
    for source in item.get('retrieved_sources', []):
        if 'first_character_index' in source:
            source['character_index'] = source.pop('first_character_index')

with open('data/dataset_docs_public.json', 'w') as f:
    json.dump(data, f, indent=4)import json
