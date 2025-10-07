import json
from pathlib import Path

# Пути к файлам
json_file = Path("infra/variables.json")
env_file = Path(".env")

# Проверяем, что файл JSON существует
if not json_file.exists():
    raise FileNotFoundError(f"JSON файл не найден: {json_file.resolve()}")

# Читаем JSON
with open(json_file, "r") as f:
    data = json.load(f)

# Конвертируем в формат .env
lines = []
for key, value in data.items():
    value = str(value)  # на всякий случай приводим к строке
    # Заменяем реальные переносы строк на \n
    value = value.replace("\n", "\\n")
    # Экранируем кавычки
    value = value.replace('"', '\\"')
    # Всегда оборачиваем в двойные кавычки для безопасности
    line = f'{key}="{value}"'
    lines.append(line)

# Если директории для .env нет — создаём
env_file.parent.mkdir(parents=True, exist_ok=True)

# Записываем в .env (создаст файл, если его нет)
with open(env_file, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f".env файл успешно создан по пути {env_file.resolve()}")
