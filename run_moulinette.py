import os
import sys
import subprocess
import glob

def run_evaluation():
    # 1. Rutas de archivos (Formato Windows)
    search_dir = os.path.join("data", "output", "search_results")
    student_path_win = os.path.join(search_dir, "dataset_code_public.json")
    dataset_path_win = os.path.join("datasets", "public", "AnsweredQuestions", "dataset_code_public.json")
    
    # 2. Verificar si el archivo de resultados existe
    if not os.path.exists(student_path_win):
        print(f"ERROR: No se encuentra el archivo de resultados en: {student_path_win}")
        print("Ejecuta primero este comando en POWERSHELL (Windows):")
        print(f"uv run python -m src search_dataset --dataset_path {dataset_path_win} --save_directory {search_dir} --k 10")
        return

    # 3. Convertir rutas a formato WSL (/mnt/c/...)
    def to_wsl_path(win_path):
        abs_path = os.path.abspath(win_path).replace('\\', '/')
        drive = abs_path[0].lower()
        return f"/mnt/{drive}{abs_path[2:]}"

    student_path_wsl = to_wsl_path(student_path_win)
    dataset_path_wsl = to_wsl_path(dataset_path_win)
    moulinette_bin_wsl = "./moulinette/moulinette_pkg/moulinette-ubuntu"

    # 4. Construir comando para ejecutar en WSL desde Windows
    # Usamos 'wsl' para llamar al binario de Linux directamente
    wsl_cmd = [
        "wsl", moulinette_bin_wsl, "evaluate_student_search_results",
        "--student_answer_path", student_path_wsl,
        "--dataset_path", dataset_path_wsl,
        "--k", "10",
        "--max_context_length", "2000"
    ]

    print(f"\n--- Ejecutando Moulinette vía WSL desde Windows ---")
    print(f"Archivo detectado: {student_path_win}")
    print(f"Lanzando comando: {' '.join(wsl_cmd)}\n")
    
    try:
        # Ejecutamos el binario de Linux a través de la capa WSL
        result = subprocess.run(wsl_cmd, capture_output=False, text=True)
        
        if result.returncode != 0:
            print("\n[!] La evaluación en WSL falló.")
            print("Asegúrate de que tienes WSL instalado y el archivo 'moulinette-ubuntu' tiene permisos de ejecución.")
            print("Puedes darlos con: wsl chmod +x " + moulinette_bin_wsl)
            
    except Exception as e:
        print(f"Error al conectar con WSL: {e}")

if __name__ == "__main__":
    run_evaluation()