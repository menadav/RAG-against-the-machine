import json
import os

# Configuración de rutas
input_path = "data/output/search_results/dataset_code_public.json"
ground_truth_path = "datasets/public/AnsweredQuestions/dataset_docs_public.json"
EXPECTED_PREFIX = "data/raw/vllm-0.10.1/"

def fix_json():
    """
    Normaliza el JSON corrigiendo question_str y asegurando que las rutas
    mantengan la estructura de subcarpetas (vllm, docs, etc) requerida.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: No se encontró {input_path}")
        return

    # 1. Cargar resultados actuales
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Cargar respaldo de preguntas para question_str
    gt_questions = {}
    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
            gt_questions = {q['question_id']: q['question'] for q in gt_data.get('rag_questions', [])}

    # 3. Procesar resultados
    results_list = data.get("search_results", [])
    corrected_results = []

    for item in results_list:
        q_id = item.get("question_id")
        q_str = item.get("question_str") or gt_questions.get(q_id, "Pregunta no encontrada")

        raw_sources = item.get("retrieved_sources") or []
        clean_sources = []
        
        for src in raw_sources:
            path = src.get("file_path") or ""
            path = path.replace("\\", "/") # Normalizar a Unix
            
            # --- LÓGICA DE RUTA CORREGIDA ---
            # Si la ruta ya contiene el nombre de la carpeta raíz del repo
            if "vllm-0.10.1/" in path:
                # Cortamos todo lo que haya antes de vllm-0.10.1/ para que empiece en data/raw/...
                relative_part = path.split("vllm-0.10.1/")[-1]
                path = EXPECTED_PREFIX + relative_part
            elif not path.startswith("data/raw/"):
                # Si la ruta es relativa simple (ej: "vllm/layers.py"), le pegamos el prefijo
                # Eliminamos './' si existe
                if path.startswith("./"): path = path[2:]
                path = EXPECTED_PREFIX + path
            
            clean_sources.append({
                "file_path": path,
                "first_character_index": int(src.get("first_character_index") or 0),
                "last_character_index": int(src.get("last_character_index") or 0)
            })

        corrected_results.append({
            "question_id": q_id,
            "question_str": q_str,
            "retrieved_sources": clean_sources
        })

    # 4. Guardar
    final_output = {"search_results": corrected_results, "k": 10}
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print(f"✅ JSON corregido con éxito en {input_path}")
    print(f"Ruta de ejemplo generada: {corrected_results[0]['retrieved_sources'][0]['file_path'] if corrected_results else 'N/A'}")

if __name__ == "__main__":
    fix_json()