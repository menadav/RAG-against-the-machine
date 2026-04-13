import os
import json
import sys
from filetype_scanner.allowed_extensions import ALLOWED_EXTENSIONS, PATH 

def checker(max_chunk_size, config_path, extensions) -> bool:
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                if config.get("max_chunk_size") == max_chunk_size:
                    same_extensions = sorted(config.get("extensions", [])) == sorted(extensions)
                    if same_extensions:
                        return True
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        print("[WARNING] File not Found create a new.\n")
    return False


def check_extensions(extension: list[str], path_cp: str) -> bool:
    if extension == []:
        print("[WARNING] You need any extensio\n")
        return True
    for x in extension:
        if x in ALLOWED_EXTENSIONS:
            pass
        else:
            print("[WARNING] The extension no is in the list.\n")
            return True
    if path_cp != PATH:
        print("[WARNING] Diferrent path.\n")
        return True
    return False
