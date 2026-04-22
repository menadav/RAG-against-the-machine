import json

# Rutas
student_path = "data/output/search_results/dataset_code_public.json"
gt_path = "datasets/public/AnsweredQuestions/dataset_docs_public.json"

# 1. Cargamos el Ground Truth para obtener las preguntas
with open(gt_path, 'r') as f:
    gt_data = json.load(f)
    # Creamos un mapa: id -> texto de la pregunta
    questions_map = {q['question_id']: q['question'] for q in gt_data.get('rag_questions', [])}

# 2. Cargamos lo que generó tu RAG
with open(student_path, 'r') as f:
    student_data = json.load(f)

# 3. Inyectamos el campo que falta
for item in student_data.get("search_results", []):
    if "question_str" not in item:
        q_id = item.get("question_id")
        item["question_str"] = questions_map.get(q_id, "Pregunta no encontrada")

# 4. Guardamos el JSON corregido
with open(student_path, 'w') as f:
    json.dump(student_data, f, indent=4)

print("✅ ¡JSON arreglado! Ahora cada resultado tiene su 'question_str'.")
