from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import os
from datetime import datetime

app = Flask(__name__, template_folder='templates')

# Загрузка модели
try:
    model_data = joblib.load('model.pkl')
    model = model_data['model']
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка при загрузке модели: {str(e)}")
    raise


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        goals = float(data['goals'])
        assists = float(data['assists'])
        games = float(data['games'])
        position = data.get('position', 'Forward')

        # Защита от деления на 0
        games = max(games, 1)

        # Расчет показателей на игру
        goals_pg = goals / games
        assists_pg = assists / games

        # Кодирование позиции
        pos_features = {
            'Forward': [1, 0, 0],
            'Midfielder': [0, 1, 0],
            'Defender': [0, 0, 1]
        }.get(position, [1, 0, 0])

        # Подготовка данных для предсказания
        features = np.array([[goals_pg, assists_pg] + pos_features])

        # Получение предсказания
        raw_prediction = model.predict(features)[0]

        # Обрезаем предсказание в диапазоне 10-100
        final_prediction = max(10, min(100, round(raw_prediction, 2)))

        return jsonify({
            'prediction': final_prediction,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Ошибка при обработке запроса'
        }), 400


@app.route('/compare', methods=['POST'])
def compare():
    try:
        players = request.json['players']
        if len(players) < 3:
            return jsonify({'error': 'Необходимо минимум 3 игрока для сравнения'}), 400

        results = []

        for player in players:
            try:
                goals = float(player.get('goals', 0))
                assists = float(player.get('assists', 0))
                games = max(float(player.get('games', 1)), 1)
                position = player.get('position', 'Forward')

                # Расчет показателей
                goals_pg = goals / games
                assists_pg = assists / games

                # Кодирование позиции
                pos_features = {
                    'Forward': [1, 0, 0],
                    'Midfielder': [0, 1, 0],
                    'Defender': [0, 0, 1]
                }.get(position, [1, 0, 0])

                # Предсказание
                features = np.array([[goals_pg, assists_pg] + pos_features])
                raw_prediction = model.predict(features)[0]

                # Обрезаем предсказание
                normalized_pred = max(10, min(100, round(raw_prediction, 2)))

                results.append({
                    'name': player.get('name', f"Player {len(results) + 1}"),
                    'performance': normalized_pred,
                    'position': position,
                    'goals': goals,
                    'assists': assists,
                    'games': games
                })

            except Exception as e:
                return jsonify({
                    'error': f"Ошибка обработки игрока {len(results) + 1}: {str(e)}"
                }), 400

        # Генерация графика
        plot_url = generate_comparison_chart(results)

        return jsonify({
            'players': results,
            'plot_url': plot_url,
            'message': 'Сравнение успешно выполнено'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Ошибка при сравнении игроков'
        }), 500


def generate_comparison_chart(players):
    plt.figure(figsize=(14, 7))

    # Цвета для разных позиций
    position_colors = {
        'Forward': '#FF6B6B',
        'Midfielder': '#4ECDC4',
        'Defender': '#45B7D1'
    }

    # Создание графика
    bars = plt.bar(
        [p['name'] for p in players],
        [p['performance'] for p in players],
        color=[position_colors[p['position']] for p in players],
        alpha=0.8
    )

    # Настройка внешнего вида
    plt.title('Сравнение эффективности игроков', fontsize=16, pad=20)
    plt.ylabel('Оценка эффективности (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Добавление значений на столбцы
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{height}%',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=position_colors['Forward'], label='Нападающие'),
        Patch(facecolor=position_colors['Midfielder'], label='Полузащитники'),
        Patch(facecolor=position_colors['Defender'], label='Защитники')
    ]
    plt.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.15, 1))

    # Оптимизация расположения
    plt.tight_layout()

    # Сохранение в base64
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120, bbox_inches='tight')
    img.seek(0)
    plt.close()

    return f"data:image/png;base64,{base64.b64encode(img.getvalue()).decode()}"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)