import json


def fix_json_final(student_path, gt_path):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
        questions_map = {q['question_id']: q['question'] for q in gt_data.get(
            'rag_questions', []
            )}
    with open(student_path, 'r') as f:
        raw_data = json.load(f)
    if isinstance(raw_data, str):
        student_data = json.loads(raw_data)
    else:
        student_data = raw_data
    fixed_results = []
    current_results = student_data.get("search_results", [])
    if not isinstance(current_results, list):
        current_results = []
    for q_id, q_text in questions_map.items():
        existing = next(
            (
                item for item in current_results
                if item.get("question_id") == q_id
                ), None
            )
        if existing:
            existing["question_str"] = q_text
            existing.setdefault("retrieved_sources", [])
            fixed_results.append(existing)
        else:
            fixed_results.append({
                "question_id": q_id,
                "question_str": q_text,
                "retrieved_sources": []
            })
    student_data["search_results"] = fixed_results
    with open(student_path, 'w') as f:
        json.dump(student_data, f, indent=4)
    print("✅ JSON normalizado y corregido.")


fix_json_final(
    "data/output/search_results/dataset_code_public.json",
    "datasets/public/AnsweredQuestions/dataset_code_public.json"
    )
