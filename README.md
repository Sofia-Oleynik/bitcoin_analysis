# Анализ минутных курсов валют: Модель смеси распределений биткоина

## Описание проекта

В данном проекте проведен анализ минутных курсов валюты биткоин, с использованием архива данных, содержащий не менее 10,000 значений. Цель состоит в том, чтобы показать, что распределение минутных изменений курса биткоина не соответствует симметричному безгранично делимому закону. Для этого мы исследуем модель смеси безгранично делимых законов.

## Задачи проекта

- Загрузить и изучить архив минутных курсов валюты.
- Проанализировать распределение минутных изменений курса биткоина.
- Оценить модель смеси распределений, которая лучше всего описывает данные.
- Применить методы статистического анализа и моделирования для достижения высокой точности и предсказательной способности модели.
- Исследовать распределения положительных и отрицательных изменений курса по отдельности.

## Данные

Данные о поминутном изменении курса валюты биткоин представлены в формате CSV.

## Установка

Для работы с проектом понадобится несколько библиотек.

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/ваш_ник/bitcoin-exchange-rate-analysis.git
   cd bitcoin-exchange-rate-analysis
   ```

2. Установите зависимости:
```bash
   pip install -r requirements.txt
```
3. Использование

После установки необходимых библиотек вы можете запустить основной скрипт анализа:
```bash
python main.py
```
