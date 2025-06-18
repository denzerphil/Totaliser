# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, Tuple
import joblib

# Настройка логирования с цветами
logging.basicConfig(
    level=logging.INFO,
    format='\033[97m%(message)s\033[0m',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 20)

# Настройки стилей
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
sns.set_theme(style="whitegrid", palette="husl")


class FootballAnalyzer:
    def __init__(self):
        load_dotenv()
        self.DB_CONFIG = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'db': os.getenv('DB_NAME', 'football_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        self.engine = None

    def connect_db(self) -> bool:
        """Подключение к базе данных"""
        try:
            conn_str = f"postgresql+psycopg2://{self.DB_CONFIG['user']}:{self.DB_CONFIG['password']}@{self.DB_CONFIG['host']}:{self.DB_CONFIG['port']}/{self.DB_CONFIG['db']}"
            self.engine = create_engine(conn_str, pool_pre_ping=True)
            logger.info("=== Успешное подключение к базе данных ===")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {str(e)}")
            return False

    def load_data(self) -> Optional[pd.DataFrame]:
        """Загрузка и подготовка данных"""
        if not self.engine:
            logger.error("Нет подключения к БД")
            return None

        try:
            logger.info("\nЗагрузка данных из базы...")
            query = """
            SELECT 
                p.playerID as player_id,
                p.name,
                SUM(a.goals) as total_goals,
                SUM(a.assists) as total_assists,
                COUNT(a.gameID) as games_played,
                a.position
            FROM appearances a
            JOIN players p ON a.playerID = p.playerID
            GROUP BY p.playerID, p.name, a.position
            HAVING SUM(a.goals) > 0 OR SUM(a.assists) > 0
            ORDER BY total_goals DESC, total_assists DESC
            """
            df = pd.read_sql(query, self.engine)

            if df.empty:
                logger.warning("Нет данных для обработки")
                return None

            logger.info("Обработка данных...")
            df = df[df['games_played'] >= 5].copy()

            # Рассчет показателей за игру
            df['goals_per_game'] = df['total_goals'] / df['games_played']
            df['assists_per_game'] = df['total_assists'] / df['games_played']

            # ПЕРЕСЧИТАНА ФОРМУЛА ЭФФЕКТИВНОСТИ
            # Новая формула: полезные действия = (голы + передачи) / игры
            # Шкала:
            #   10% = 0.075 полезных действий за игру (1.5 за 20 матчей)
            #   100% = 2.5 полезных действий за игру (50 за 20 матчей)
            min_actions = 0.075
            max_actions = 2.5

            # Рассчет полезных действий за игру
            df['actions_per_game'] = (df['total_goals'] + df['total_assists']) / df['games_played']

            # Линейное преобразование в шкалу 10-100%
            df['performance'] = 10 + ((df['actions_per_game'] - min_actions) *
                                      (90 / (max_actions - min_actions)))

            # Ограничение значений
            df['performance'] = df['performance'].clip(10, 100)

            position_map = {
                'GK': 'Goalkeeper', 'DR': 'Defender', 'DL': 'Defender',
                'DC': 'Defender', 'DF': 'Defender', 'CB': 'Defender',
                'MF': 'Midfielder', 'DM': 'Midfielder', 'AM': 'Midfielder',
                'MC': 'Midfielder', 'DMC': 'Midfielder', 'AMC': 'Midfielder',
                'FW': 'Forward', 'ST': 'Forward', 'CF': 'Forward',
                'WG': 'Forward', 'FWL': 'Forward', 'FWR': 'Forward',
                'AML': 'Forward', 'AMR': 'Forward'
            }
            df['position_group'] = df['position'].map(position_map)
            df = pd.get_dummies(df, columns=['position_group'], prefix='', prefix_sep='')

            for col in ['Goalkeeper']:
                if col in df.columns:
                    df.drop(col, axis=1, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}")
            return None

    # ДОБАВЛЕН ОТСУТСТВУЮЩИЙ МЕТОД
    def print_formatted_table(self, df: pd.DataFrame, columns: list, title: str = ""):
        """Красивое форматирование таблиц"""
        if title:
            logger.info(f"\n=== {title} ===")

        formatted = df[columns].copy()
        for col in formatted.columns:
            if formatted[col].dtype == 'float64':
                formatted[col] = formatted[col].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "")

        col_widths = {col: max(len(col), formatted[col].astype(str).str.len().max()) for col in columns}
        header = "  ".join(f"{col:<{col_widths[col]}}" for col in columns)
        logger.info(header)

        for _, row in formatted.iterrows():
            line = "  ".join(f"{str(row[col]):<{col_widths[col]}}" for col in columns)
            logger.info(line)

    def train_model(self, df: pd.DataFrame) -> Optional[Tuple[CatBoostRegressor, dict]]:
        """Обучение модели и анализ"""
        if df is None or df.empty:
            logger.error("Нет данных для обучения")
            return None

        try:
            logger.info("\nПодготовка к обучению модели...")
            features = [
                'goals_per_game',
                'assists_per_game',
                'Defender',
                'Midfielder',
                'Forward'
            ]

            features = [f for f in features if f in df.columns]
            if not features:
                logger.error("Нет подходящих признаков для обучения")
                return None

            X = df[features]
            y = df['performance']

            # Разделение данных
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42)

            # Инициализация и обучение модели
            logger.info("\nОбучение модели CatBoost...")
            model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.03,
                depth=6,
                loss_function='MAE',
                verbose=100,
                random_seed=42,
                early_stopping_rounds=50
            )

            model.fit(
                X_train, y_train,
                eval_set=(X_test, y_test),
                plot=False
            )

            # Сохранение модели
            joblib.dump({'model': model}, 'model.pkl')
            logger.info("Модель сохранена в model.pkl")

            # Оценка модели
            y_pred = model.predict(X_test)

            # Обрезаем предсказания в диапазоне 10-100
            y_pred = np.clip(y_pred, 10, 100)

            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'min_actual': y.min(),
                'max_actual': y.max(),
                'min_pred': min(y_pred),
                'max_pred': max(y_pred)
            }

            logger.info("\n=== Оценка модели ===")
            logger.info(f"MAE (Средняя абсолютная ошибка): {metrics['mae']:.2f}%")
            logger.info(f"RMSE (Среднеквадратичная ошибка): {metrics['rmse']:.2f}%")
            logger.info(f"R² (Коэффициент детерминации): {metrics['r2']:.2f}")
            logger.info(f"Диапазон фактических значений: {metrics['min_actual']:.2f}-{metrics['max_actual']:.2f}")
            logger.info(f"Диапазон предсказаний: {metrics['min_pred']:.2f}-{metrics['max_pred']:.2f}")

            return model, metrics

        except Exception as e:
            logger.error(f"Ошибка при обучении модели: {str(e)}")
            return None

    # ДОБАВЛЕН ОТСУТСТВУЮЩИЙ МЕТОД
    def analyze_results(self, df: pd.DataFrame, model: CatBoostRegressor) -> bool:
        """Анализ и визуализация результатов"""
        try:
            logger.info("\nАнализ результатов...")

            features = [f for f in ['goals_per_game', 'assists_per_game', 'Defender', 'Midfielder', 'Forward']
                        if f in df.columns]
            feature_importance = pd.DataFrame({
                'Признак': features,
                'Важность (%)': model.get_feature_importance() * 100
            }).sort_values('Важность (%)', ascending=False)

            self.print_formatted_table(feature_importance, ['Признак', 'Важность (%)'], "Важность признаков")

            plt.figure(figsize=(18, 12))

            plt.subplot(2, 2, 1)
            sns.barplot(x='Важность (%)', y='Признак', data=feature_importance)
            plt.title('Влияние признаков на оценку игрока', fontsize=14)

            plt.subplot(2, 2, 2)
            sns.regplot(x='goals_per_game', y='performance', data=df,
                        scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
            plt.title('Зависимость эффективности от голов за игру', fontsize=14)

            plt.subplot(2, 2, 3)
            sns.regplot(x='assists_per_game', y='performance', data=df,
                        scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
            plt.title('Зависимость эффективности от передач за игру', fontsize=14)

            plt.subplot(2, 2, 4)
            pos_melt = df.melt(id_vars=['position'],
                               value_vars=['goals_per_game', 'assists_per_game'],
                               var_name='Метрика', value_name='Значение')
            pos_melt['Метрика'] = pos_melt['Метрика'].replace({
                'goals_per_game': 'Голы',
                'assists_per_game': 'Передачи'
            })

            sns.boxplot(x='position', y='Значение', hue='Метрика',
                        data=pos_melt, showfliers=False)
            plt.title('Продуктивность по позициям', fontsize=14)
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig('football_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("\nГрафики сохранены в файл 'football_analysis.png'")

            top_players = df.sort_values('performance', ascending=False).head(10)
            self.print_formatted_table(
                top_players,
                ['name', 'position', 'total_goals', 'total_assists', 'games_played', 'performance'],
                "Топ-10 игроков по эффективности"
            )

            return True

        except Exception as e:
            logger.error(f"Ошибка при анализе результатов: {str(e)}")
            return False

    def run(self):
        """Основной метод выполнения анализа"""
        try:
            if not self.connect_db():
                return

            df = self.load_data()
            if df is None:
                return

            logger.info(f"\nЗагружено записей: {len(df)}")
            self.print_formatted_table(
                df.head(10),
                ['name', 'position', 'total_goals', 'total_assists', 'games_played', 'performance'],
                "Первые 10 записей"
            )

            result = self.train_model(df)
            if result is None:
                return

            model, metrics = result
            self.analyze_results(df, model)

        except Exception as e:
            logger.error(f"\nКритическая ошибка: {str(e)}")
        finally:
            if self.engine:
                self.engine.dispose()
                logger.info("\n=== Соединение с БД закрыто ===")


if __name__ == "__main__":
    analyzer = FootballAnalyzer()
    analyzer.run()