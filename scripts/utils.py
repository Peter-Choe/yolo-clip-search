import json
import sys, os
import json
import tempfile
import shutil


def safe_write_json(data, path):
    """
    JSON 파일을 직접 덮어쓰지 않고,
    임시 파일에 먼저 완전히 쓰고,
    정상적으로 완료되면 원래 위치로 이동(move)
    이로써 파일이 깨지거나 비정상 상태로 저장되는 것을 방지
    """
    dir_path = os.path.dirname(path)
    with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, suffix=".json") as tmp_file:
        json.dump(data, tmp_file, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        temp_path = tmp_file.name
    shutil.move(temp_path, path)
