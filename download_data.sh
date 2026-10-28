#!/usr/bin/env bash
# Скачивает датасет FIFA 15–23 с Kaggle. Нужен ваш ключ в ~/.kaggle (не кладите его в репозиторий).
set -euo pipefail
mkdir -p data
kaggle datasets download -d stefanoleone992/fifa-23-complete-player-dataset -f "male_players (legacy).csv" -p data
unzip -o "data/male_players%20%28legacy%29.csv.zip" -d data 2>/dev/null || unzip -o data/*.zip -d data
mv "data/male_players (legacy).csv" data/male_players_legacy.csv
rm -f data/*.zip
