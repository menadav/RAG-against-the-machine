import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os

def hablar_con_qwen_y_contexto(pregunta: str, nombre_archivo: str) -> None:
    model_name: str = "Qwen/Qwen3-0.6B" # Asegúrate de que el nombre sea el correcto
    try:
        # 1. Leer el archivo de texto en la raíz
        if not os.path.exists(nombre_archivo):
            print(f"¡Oye mi pana, no encuentro el archivo {nombre_archivo}!")
            return
        with open(nombre_archivo, "r", encoding="utf-8") as f:
            texto_contexto = f.read()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        prompt_sistema = (
            "Eres un asistente experto. Utiliza ÚNICAMENTE el siguiente contexto "
            "para responder a la pregunta del usuario. Si la respuesta no está "
            "en el texto, di que no lo sabes."
            "Escribes breve y corto \n\n"
            f"CONTEXTO:\n{texto_contexto}"
        )
        messages = [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": pregunta}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=150, 
            temperature=0.1 # Muy baja para que no invente cosas fuera del texto
        )
        response = tokenizer.batch_decode(
            generated_ids[:, model_inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0]

        print(f"\nRespuesta basada en el archivo '{nombre_archivo}':\n{response}")

    except Exception as e:
        print(f"¡Algo falló, mi pana!: {e}")

if __name__ == "__main__":
    # Crea un archivo llamado 'informacion.txt' con algo escrito dentro
    archivo = "informacion.txt"
    pregunta = "¿Qué dice el documento ?"
    
    hablar_con_qwen_y_contexto(pregunta, archivo)
