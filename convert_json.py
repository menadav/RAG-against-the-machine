import json
from collections import OrderedDict

# 1. Cargamos el archivo
with open('data/dataset_docs_public.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

new_search_results = []

for item in data.get('search_results', []):
    # 2. Creamos un diccionario ordenado (OrderedDict) para forzar el orden de las claves
    ordered_item = OrderedDict()
    
    # 3. Insertamos en el orden exacto que espera Pydantic
    ordered_item['question_id'] = item.get('question_id', "")
    ordered_item['question_str'] = item.get('question_str') or item.get('question', "")
    ordered_item['retrieved_sources'] = item.get('retrieved_sources', [])
    
    # 4. Validamos y recortamos las fuentes
    for source in ordered_item['retrieved_sources']:
        start = source.get('first_character_index', 0)
        end = source.get('last_character_index', 0)
        if end - start > 2000:
            source['last_character_index'] = start + 2000
            
    new_search_results.append(ordered_item)

# 5. Creamos la estructura final
final_data = {
    "search_results": new_search_results,
    "k": data.get('k', 10)
}

# 6. Guardamos con indentación para que sea legible
with open('data/dataset_docs_public.json', 'w') as f:
    json.dump(final_data, f, indent=4)

print("¡Archivo ordenado y estructurado correctamente, mi pana!")
