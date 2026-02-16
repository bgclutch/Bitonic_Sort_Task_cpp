#!/bin/bash


BITONIC_EXE="../../build/bitonic_sort/bitonic_sort"

INPUT_DIR="input_files"
OUTPUT_DIR="output_files"

# Создаем папку для вывода, если её нет
mkdir -p "$OUTPUT_DIR"

# Очищаем старые результаты (опционально)
# rm -f "$OUTPUT_DIR"/*.out

echo "Starting tests..."
echo "Binary: $BITONIC_EXE"

# --- ЦИКЛ ПО ФАЙЛАМ ---
for input_file in "$INPUT_DIR"/test_*.in; do
    # Получаем имя файла без пути
    filename=$(basename "$input_file")
    # Меняем расширение на .out для выходного файла
    output_file="$OUTPUT_DIR/${filename%.in}.out"

    echo -n "Running $filename... "

    # 1. ЗАПУСКАЕМ РОДНУЮ КОМАНДУ ЧЕРЕЗ <
    # 2. Перенаправляем stdout в файл через >
    # 3. Перенаправляем stderr в консоль, чтобы видеть ошибки OpenCL

    "$BITONIC_EXE" < "$input_file" > "$output_file" 2> error_log.tmp

    # Проверяем код возврата
    if [ $? -eq 0 ]; then
        echo "OK"
    else
        echo "FAIL (Error log below)"
        cat error_log.tmp
    fi
done

# Удаляем временный лог ошибок
rm -f error_log.tmp
echo "All tests finished."