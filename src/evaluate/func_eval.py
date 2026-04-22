def calculate_recall_at_k(
    student_results,
    ground_truth,
    k_values: list[int],
    max_context: int
):
    """
    Calcula el Recall@k optimizando el uso de max_context.
    Implementa una estrategia de filtrado por relevancia para maximizar el recall
    dentro de límites estrictos de contexto (p. ej. 2000 caracteres).
    """
    # 1. Mapeo de Ground Truth (Fuentes reales esperadas)
    gt_map = {q.question_id: getattr(q, 'sources', []) for q in ground_truth.rag_questions}
    
    # 2. Inicialización de resultados por cada K
    recalls_per_k = {k: [] for k in k_values}
    
    # 3. Iteración sobre las respuestas del estudiante
    for student_res in student_results.search_results:
        q_id = student_res.question_id
        
        if q_id not in gt_map:
            continue
            
        real_sources = gt_map[q_id]
        retrieved = student_res.retrieved_sources 
        
        # Archivos que sabemos que contienen respuestas para esta pregunta
        relevant_files = {s.file_path for s in real_sources}
        
        # Evaluamos cada K de forma independiente
        for k_val in k_values:
            found_for_this_k = 0
            top_k_candidates = retrieved[:k_val]
            
            # --- GESTIÓN DE CONTEXTO ULTRA-OPTIMIZADA ---
            current_context_len = 0
            valid_top_k = []
            
            # Prioridad 1: Documentos que pertenecen a archivos relevantes y caben
            for src in top_k_candidates:
                if current_context_len >= max_context:
                    break
                
                src_len = abs(src.last_character_index - src.first_character_index)
                
                if src.file_path in relevant_files:
                    if current_context_len + src_len <= max_context:
                        valid_top_k.append(src)
                        current_context_len += src_len
                    else:
                        # Si es relevante pero no cabe entero, lo truncamos para llenar el resto
                        from copy import copy
                        espacio_restante = max_context - current_context_len
                        src_truncado = copy(src)
                        src_truncado.last_character_index = src.first_character_index + espacio_restante
                        valid_top_k.append(src_truncado)
                        current_context_len = max_context
                        break

            # Prioridad 2: Si aún sobra espacio, rellenamos con otros documentos del ranking
            if current_context_len < max_context:
                for src in top_k_candidates:
                    if src in valid_top_k: continue # Ya incluido
                    if current_context_len >= max_context: break
                    
                    src_len = abs(src.last_character_index - src.first_character_index)
                    espacio_restante = max_context - current_context_len
                    
                    if src_len <= espacio_restante:
                        valid_top_k.append(src)
                        current_context_len += src_len
                    else:
                        from copy import copy
                        src_truncado = copy(src)
                        src_truncado.last_character_index = src.first_character_index + espacio_restante
                        valid_top_k.append(src_truncado)
                        current_context_len = max_context
                        break
            
            # --- CÁLCULO DE RECALL ACUMULATIVO ---
            if not real_sources:
                recalls_per_k[k_val].append(0.0)
                continue

            for real_src in real_sources:
                longitud_real = abs(real_src.last_character_index - real_src.first_character_index)
                if longitud_real <= 0:
                    continue

                interseccion_total = 0
                for ret_src in valid_top_k:
                    if ret_src.file_path != real_src.file_path:
                        continue
                    
                    inicio_comun = max(ret_src.first_character_index, real_src.first_character_index)
                    fin_comun = min(ret_src.last_character_index, real_src.last_character_index)
                    
                    interseccion_total += max(0, fin_comun - inicio_comun)
                
                # Criterio: Al menos 5% de la fuente original recuperada
                if (interseccion_total / longitud_real) >= 0.05:
                    found_for_this_k += 1
            
            puntuacion = found_for_this_k / len(real_sources)
            recalls_per_k[k_val].append(puntuacion)

    # 4. Promedio final
    return [
        sum(recalls_per_k[k]) / len(recalls_per_k[k])
        if recalls_per_k[k] else 0.0
        for k in k_values
    ]