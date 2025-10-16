# Решение Хакатона Finam X HSE Forecast Октябрь 2025 
[**Go to English Version**](#english-version)

### Источники данных

#### 1. **Свечные данные** (Рыночные OHLCV)
```
Размерность: (N_записей, 7)
Столбцы: [ticker, begin, open, high, low, close, volume]
Частота: Внутридневные → Агрегированы до дневных
Диапазон дат: Исторические до 2025-09-08 (TRAIN_CUTOFF)
```

**Пример:**
```
ticker    begin                open    high    low     close   volume
SBER      2024-01-15 10:00:00  285.5   287.2   284.1   286.8   15000000
SBER      2024-01-15 10:05:00  286.8   288.0   286.5   287.5   12000000
```

**Предобработка:**
- **Агрегация**: Внутридневные → Дневные (первый open, максимальный high, минимальный low, последний close, сумма volume)
- **Заполнение пропусков**: `.asfreq('D')` с заполнением ffill для пропущенных выходных/праздников

#### 2. **Новостные данные**
```
Размерность: (N_статей, 3)
Столбцы: [publish_date, title, publication]
Диапазон дат: Непрерывный поток
Язык: Смешанный русский/английский
```

**Стратегия фильтрации:**
- **Сопоставление ключевых слов**: 117 тикеров × 5-10 ключевых слов каждый
- **Выбор топ-30**: Самые свежие новости по тикеру
- **Временной сдвиг**: +1 день

#### 3. **Словарь ключевых слов тикеров**
```python
TICKER_KEYWORDS = {
    "SBER": ["Сбербанк", "Sberbank", "СБЕР", "Сбер", "Банк", "Ключевая ставка", "Трамп", "Путин"],
    # ... всего 117 тикеров
}
```

**Назначение:**
- Быстрая предварительная фильтрация перед дорогими вызовами LLM
- Захватывает названия компаний, отраслевые термины, геополитические события
- Снижает вызовы API с ~50K до ~5K

### Размерности данных на всём конвейере

```
СЫРЫЕ ДАННЫЕ:
├─ Свечи: (N_внутридневных_записей, 7) → Агрегация → (N_дней, 5)
└─ Новости: (N_статей, 3) → Фильтр+LLM → (N_релевантных_новостей, 2)

ИНЖЕНЕРИЯ ПРИЗНАКОВ:
├─ Технические индикаторы: (N_дней, 21)  # 21 технический признак
├─ LLM признаки: (N_дней, 2)              # sentiment + impact
└─ Объединённые: (N_дней, 23)             # Всего признаков

НОРМАЛИЗАЦИЯ:
└─ RobustScaler: (N_дней, 23) → (N_дней, 23)  # Масштабированные

ФОРМИРОВАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ:
├─ Вход X: (N_примеров, 60, 23)           # 60-дневные окна
├─ Цель Y_returns: (N_примеров, 20)       # 20-дневная доходность
└─ Цель Y_directions: (N_примеров, 20)    # 20-дневные направления

БАТЧИРОВАНИЕ:
├─ Train Batch: (64, 60, 23)
└─ Val Batch: (128, 60, 23)

ОБРАБОТКА МОДЕЛЬЮ:
├─ Проекция признаков: (B, 60, 23) → (B, 60, 128)
├─ Positional Encoding: (B, 60, 128) → (B, 60, 128)
├─ Выход TCN: (B, 60, 128)
├─ Выход Transformer: (B, 60, 128)
├─ Объединённый: (B, 60, 128)
├─ Выход Attention: (B, 60, 128)
├─ Pooling: (B, 60, 128) → (B, 384)       # last+avg+max
├─ Финальное представление: (B, 384) → (B, 128)
├─ Выход дневных голов: (B, 20)
└─ Выход глобальной головы: (B, 20)

ВЫХОД:
├─ Предсказанная доходность: (B, 20)
└─ Предсказанные вероятности: (B, 20)

Где: B = размер batch, N = число записей
```

## Архитектура

### Полный pipeline с размерностями

```
Входная последовательность: (Batch=64, Seq=60, Features=23)
                        ↓
┌───────────────────────────────────────────────┐
│ Проекция признаков: Linear(23 → 128)          │
│ Выход: (64, 60, 128)                          │
└───────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Positional Encoding                           │
│ Размер PE: (1, 5000, 128)                     │
│ Выход: (64, 60, 128)                          │
└───────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────────┐         ┌─────────────────────┐
│   Ветка TCN       │         │ Ветка Transformer   │
│                   │         │                     │
│ Транспонирование: │         │ 4 слоя энкодера     │
│ (64,60,128)       │         │ Каждый слой:        │
│      ↓            │         │ - MultiHead Attn    │
│ (64,128,60)       │         │   8 голов           │
│      ↓            │         │   d_k = 128/8 = 16  │
│ Conv1d слой 1:    │         │ - FFN (128→512→128) │
│ k=7, d=1          │         │ - LayerNorm         │
│ (64,128,60)       │         │ - Dropout           │
│      ↓            │         │                     │
│ Conv1d слой 2:    │         │ Выход: (64,60,128)  │
│ k=7, d=2          │         │                     │
│ (64,128,60)       │         │                     │
│      ↓            │         │                     │
│ Conv1d слой 3:    │         │                     │
│ k=7, d=4          │         │                     │
│ (64,128,60)       │         │                     │
│      ↓            │         │                     │
│ Обратное транспон.:│        │                     │
│ (64,60,128)       │         │                     │
└───────────────────┘         └─────────────────────┘
        ↓                               ↓
        └───────────────┬───────────────┘
                        ↓
┌─────────────────────────────────────────────────┐
│ Остаточное слияние: TCN + Transformer + Исходное│
│ (64,60,128) + (64,60,128) + (64,60,128)         │
│ Выход: (64, 60, 128)                            │
└─────────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Multi-Head Attention (Self-Attention)         │
│ Головы: 8, d_k=16, d_v=16                     │
│ Q,K,V: (64, 60, 128) каждый                   │
│ Оценки attention: (64, 8, 60, 60)             │
│ Выход: (64, 60, 128)                          │
└───────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Тройная стратегия Pooling                     │
│ ├─ Последний: [:, -1, :] → (64, 128)         │
│ ├─ Среднее: .mean(dim=1) → (64, 128)         │
│ └─ Максимум: .max(dim=1) → (64, 128)         │
│ Конкатенация: (64, 384)                       │
└───────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Проекция финального представления             │
│ Linear(384 → 128) + GELU + Dropout            │
│ Выход: (64, 128)                              │
└───────────────────────────────────────────────┘
                        ↓
       ┌─────────────────────────────────┬──────────────────────────────┐
       ↓                                 ↓                               ↓
┌─────────────────┐           ┌─────────────────┐           ┌──────────────────┐
│ Глобальная      │           │ 20 голов        │           │ 20 голов         │
│ голова          │           │ доходности      │           │ вероятности      │
│ доходности      │           │ Каждая:         │           │ Каждая:          │
│ Linear(128→64)  │           │ Linear(128→64)  │           │ Linear(128→64)   │
│ GELU            │           │ GELU            │           │ GELU             │
│ Dropout         │           │ Dropout         │           │ Dropout          │
│ Linear(64→20)   │           │ Linear(64→1)    │           │ Linear(64→1)     │
│ Выход: (64,20)  │           │                 │           │ Sigmoid          │
└─────────────────┘           │ Объединение:    │           │                  │
       ↓                      │ (64, 20)        │           │ Объединение:     │             
       │                      └─────────────────┘           │ (64, 20)         │
       │                               ↓                    └──────────────────┘
       │                               │                              ↓
┌──────┴───────────────────────────────┴──────────────────────────────┴──────┐
│ Финальная взвешенная комбинация                                            │
│ Доходность: 0.7×Дневные + 0.3×Глобальные                                   │
│ Выход: (64, 20) для доходности                                             │
│        (64, 20) для вероятностей                                           │
└────────────────────────────────────────────────────────────────────────────┘
```

#### 2. **Temporal Convolutional Network (TCN)**

Компонент TCN использует dilated causal convolutions для захвата временных паттернов на разных масштабах:

- **Экспоненциальное расширение**: `2^i` для слоя `i`, обеспечивает рост receptive field
- **3 слоя** со скрытой размерностью 128
- **Размер ядра**: 7 (захватывает недельные паттерны)
- **Residual connections** сохраняют поток градиентов
- **Batch normalization** + ReLU + Dropout для регуляризации

**Ключевые преимущества:**
- Параллельные вычисления (быстрее чем RNN)
- Стабильные градиенты на длинных последовательностях
- Захватывает многомасштабные временные паттерны

#### 3. **Transformer Encoder**

4-слойный Transformer с:
- **8 attention heads** для извлечения признаков с разных перспектив
- **512-мерные feedforward** сети (4x скрытый размер)
- **GELU activation** для гладкой нелинейности
- **Sinusoidal positional encoding** для временной осведомлённости

**Назначение:**
- Моделирует долгосрочные зависимости (60-дневное входное окно)
- Захватывает сложные взаимодействия между признаками
- Предоставляет глобальный контекст для предсказаний

#### 4. **Система Multihead предсказаний**

**Индивидуальные дневные головы (20 голов):**
- Каждая голова специализируется на предсказании конкретного будущего дня
- Отдельная регрессия доходности и классификация направления
- Позволяет модели изучать специфичные для дня паттерны

**Глобальная голова:**
- Предсказывает все 20 значений доходности одновременно
- Работает как регуляризация для поддержания согласованности
- Финальное предсказание: `0.7 × индивидуальные + 0.3 × глобальные`

### Извлечение новостных признаков с помощью LLM

#### Конвейер обработки новостей

```
Сырые новости → Фильтрация по ключевым словам → Анализ LLM → Интеграция признаков
```

**1. Предварительная фильтрация по ключевым словам:**
- Индивидуальный словарь с ключевыми словами для каждого тикера
- Включает названия компаний на русском/английском + отраслевые термины
- Захватывает геополитические фигуры для этого периода и российского рынка ("Trump", "Putin")

**2. Анализ настроений через LLM:**
```python
Модель: GPT-3.5-turbo через OpenRouter API
Вход: Заголовок новости + текст новости
Выход: {
  "sentiment": -3 до +3,  # От медвежьего до бычьего
  "impact": 0 до 5        # Значимость для рынка
}
```

**3. Временное выравнивание:**
- Новостные признаки сдвинуты на +1 день (учёт задержки рыночной реакции)
- Агрегированы на дневном уровне
- Пропущенные дни заполнены нулями (нет новостей)

**4. Инженерия признаков:**
- Оценка настроения (-3 до +3)
- Величина влияния (0 до 5)
- Взвешенная интеграция с рыночными данными (вес: 1.0 при обучении)

## Feature Engineering

### Технические индикаторы (21 признак)

**На основе цены:**
- Логарифмы цен (open, high, low, close)
- Логарифмическая доходность (дневная, 20-дневная волатильность)
- Множественные скользящие средние (MA5, MA20, MA50, MA200)
- Ценовой импульс (10-дневный)
- Положение относительно скользящих средних

**На основе объёма:**
- Логарифм объёма
- Изменение объёма
- Отношение объёма к MA10

**Волатильность:**
- Скользящее стандартное отклонение (20, 30, 100 дней)
- Средний истинный диапазон (ATR)
- Годовая доходность (250 дней)

**Технические индикаторы:**
- RSI (индекс относительной силы, 14-периодный)
- MACD (экспоненциальные MA 12/26)
- Стохастический осциллятор (10-дневный high/low)

**Рыночная структура:**
- Диапазон high-low
- Импульс типичной цены

### LLM признаки (2 признака на тикер)
- Оценка настроения
- Величина влияния

**Общая входная размерность: 23 признака**

## Стратегия обучения

### Функция потерь

**Комбинированные потери** (α=0.5, β=0.3, γ=0.2):
```
L = α·MSE(доходность) + β·BCE(вероятности) + γ·Точность направления
```

**Компоненты:**
1. **MSE Loss**: Регрессия непрерывной доходности
2. **BCE Loss**: Бинарная классификация направления (вверх/вниз)
3. **Точность направления**: Штрафует неправильные предсказания направления

**Обоснование:** Многоцелевая оптимизация улучшает как величину, так и точность направления.

### Оптимизация

- **Оптимизатор**: AdamW (weight_decay=1e-5)
- **Скорость обучения**: 5e-4 с планировщиком ReduceLROnPlateau
- **Обрезка градиентов**: max_norm=1.0 (для стабильности)
- **Ранняя остановка**: patience=5 эпох
- **Размер batch**: 64 (обучение), 128 (валидация)

### Разделение данных

```
Обучение: Все данные до (TRAIN_CUTOFF - 200 дней)
Валидация: Последние 200 дней перед TRAIN_CUTOFF
Тест: Данные после TRAIN_CUTOFF
```

**Стратегия валидации:**
- Разделение временных рядов (без утечки из будущего)
- Датасеты по тикерам объединены
- Общий scaler признаков для всех тикеров

## Предобработка данных

### Робастное масштабирование

Использует `RobustScaler` (устойчив к выбросам):
```
X_scaled = (X - медиана) / IQR
```

**Почему RobustScaler?**
- Финансовые данные имеют экстремальные выбросы (рыночные крахи, всплески)
- Медиана/IQR более стабильны чем среднее/стандартное отклонение
- Сохраняет относительные взаимосвязи

### Временная согласованность

- **Обработка пропущенных данных**: 
  - Технические индикаторы: Заполнение вперёд для непрерывности
  - LLM признаки: Заполнение нулями (нет новостей = нейтрально)
- **Валидация последовательностей**: Обеспечивает отсутствие NaN во входных окнах

## Pipeline предсказаний

### Процесс инференса

1. **Подготовка входной последовательности**: Последние 60 дней признаков
2. **Взвешивание LLM признаков**: Применить вес 0.2 (снижает переобучение к новостям)
3. **Прямой проход модели**: Генерирует 20-дневные прогнозы доходности и вероятности
4. **Постобработка**:
   - Обрезка экстремальных предсказаний (±3%)
   - Экспоненциальное сглаживание (α=0.3)
   - Конвертация логарифмической доходности в цены
   - 5-периодное скользящее среднее для сглаживания
   - Пересчёт доходности из сглаженных цен

5. **Финальный выход**: Кумулятивная доходность для 20 дней

### Стабилизация предсказаний

**Обрезка:**
```python
pred_returns = np.clip(pred_returns, -0.03, 0.03)
```
Предотвращает нереалистичные дневные движения.

**Сглаживание:**
```python
pred_returns = pd.Series(pred_returns).ewm(alpha=0.3).mean()
```
Снижает шум и волатильность в предсказаниях.

**Реконструкция на основе цен:**
```python
prices = last_price * np.exp(np.cumsum(log_returns))
smoothed = rolling_mean(prices, window=5)
```
Обеспечивает физическую согласованность (цены не могут быть отрицательными).

## Использование

### Требования

```bash
pip install numpy pandas torch scikit-learn tqdm openai
```

### Конфигурация

```python
# Пути к данным
CANDLES_PATH_1 = "path/to/candles.csv"
NEWS_PATH_1 = "path/to/news.csv"

# API ключ
OPENROUTE_API_KEY = "your_openrouter_api_key"

# Гиперпараметры
INPUT_WINDOW = 60      # Дней исторических данных
PRED_HORIZON = 20      # Длина прогноза
TRAIN_CUTOFF = '2025-09-08'
```

### Запуск

```python
# 1. Загрузить данные
CANDLES, NEWS = load_and_merge_data()

# 2. Извлечь LLM признаки
extractor = NewsFeatureExtractor(NEWS, CANDLES, TICKER_KEYWORDS, 
                                  TRAIN_CUTOFF, OPENROUTE_API_KEY)
LLM_FEATURES = extractor.extract_features(TRAIN_TICKERS)

# 3. Обучить модель
model = ImprovedForecastModel(pred_horizon=20).to(device)
trained_model = train_model(model, train_loader, val_loader)

# 4. Сгенерировать прогнозы
forecast = generate_forecast(trained_model, df_ticker, llm_ticker, 
                             feature_scaler, TRAIN_CUTOFF)
```

## Формат выхода

CSV файл со столбцами:
```
ticker, p1, p2, ..., p20
```

Где `p1` до `p20` это кумулятивная доходность для каждого из 20 дней прогноза:
```python
R_cumulative = exp(sum(log_returns)) - 1
```

## Ключевые технические решения

### 1. **Почему гибридный TCN-Transformer?**

- **TCN**: Эффективен для локальных паттернов (дневные, недельные циклы)
- **Transformer**: Захватывает долгосрочные зависимости (месячные, квартальные тренды)
- **Комбинация**: Лучшее от обоих с остаточными связями

### 2. **Почему множественные головы предсказаний?**

- Каждый день имеет уникальные характеристики (например, эффект понедельника)
- Индивидуальные головы специализируются без взаимного влияния
- Глобальная голова предотвращает переобучение к шуму

### 3. **Почему LLM для новостей?**

- GPT-3.5 понимает контекст и сарказм
- Обрабатывает многоязычный контент (русские финансовые новости)
- Оценка влияния захватывает значимость для рынка

### 4. **Почему сдвиг на +1 день для новостей?**

- Предотвращает утечку данных

### 5. **Почему Robust Scaler?**

- Финансовые данные имеют тяжёлые хвосты (события "Чёрного лебедя")
- Стандартный scaler искажается выбросами
- Медиана/IQR более стабильны для нормализации

## Статистика модели

- **Параметры**: ~2.5M (эффективно для обучения на GPU)
- **Входная последовательность**: 60 дней × 23 признака
- **Время обучения**: ~15 минут на T4 GPU (10 эпох)
- **Инференс**: <1 секунда на тикер
- **Память**: ~4GB GPU RAM (batch_size=64)

## Производительность

**Сильные стороны:**
- Захватывает многомасштабные временные паттерны
- Использует как технические, так и фундаментальные (новостные) данные
- Корректно обрабатывает пропущенные данные
- Производит стабильные, не экстремальные прогнозы

**Ограничения:**
- Предполагает непрерывность рыночной структуры (проблемы при смене режима)
- Вызовы LLM API дорогие (ограничены до 5000)
- Требует значительный объём исторических данных
- Производительность ухудшается в периоды высокой волатильности

##  Визуализация предсказанной цены на 20 дней вперед

На данном графике представлена **визуализация предсказанной цены** на 20 торговых дней вперед для 5 тикеров .

![Визуализация предсказанной цены на 20 дней вперед](https://github.com/MakarRybkin/Finam_hack_oct_2025/raw/master/visual_preds(5_tickers).png)

---

## Сравнение прогноза с реальными графиками (20 дней)

В этом разделе представлены **реальные графики цен** для 5 анализируемых тикеров за тот же 20-дневный период, для которого строился прогноз.

![Реальные графики 5 тикеров на 20 дней](https://github.com/MakarRybkin/Finam_hack_oct_2025/raw/master/real_prices(5_tickers).png)

### Наблюдения по точности прогнозов:

Проведенный анализ показал, что для акций следующих компаний:
* **Лукойл**
* **ГМК "Норильский никель" (ГМК)**
* **Новатек**

Прогнозы на короткий период, а именно на **5–10 дней** вперед, продемонстрировали высокую точность и оказались **очень похожи на настоящие** фактические изменения цены.

На более длинном горизонте (10-20 дней) отклонения могут быть более существенными, что является типичным для краткосрочного финансового прогнозирования. (что объяняется отсутвием новостей и реальных данных за предыдущие 5-10 дней)

## Возможные улучшения

1. **Ансамблевые методы**
2. **Работа на более коротком таймфрейме(1H , 4H)** и предсказание соответственно на 20 часов и на 100 часов вперед
3. **Визуализация attention**: Интерпретируемость важных признаков
4. **Динамическое взвешивание**: Настройка важности LLM признаков по тикеру/сектору
5. **Множественное разрешение**: Добавить внутридневные данные для краткосрочных предсказаний
6. **Моделирование риска**: Предсказывать волатильность вместе с доходностью
7. **Обучение с подкреплением**: Слой оптимизации портфеля

## Ссылки

- **TCN**: Bai et al. (2018) - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- **Transformer**: Vaswani et al. (2017) - "Attention Is All You Need"
- **Финансовое прогнозирование**: Fischer & Krauss (2018) - "Deep learning with long short-term memory networks for financial market predictions"

## Лицензия

MIT License - Свободно для академического и коммерческого использования.

**Примечание**: Эта система предназначена для исследовательских целей. Всегда консультируйтесь с финансовыми советниками перед принятием инвестиционных решений.

# English Version

### Data Sources

#### 1. **Candles Data** (Market OHLCV)
```
Shape: (N_records, 7)
Columns: [ticker, begin, open, high, low, close, volume]
Frequency: Intraday → Aggregated to Daily
Date Range: Historical to 2025-09-08 (TRAIN_CUTOFF)
```

**Example:**
```
ticker    begin                open    high    low     close   volume
SBER      2024-01-15 10:00:00  285.5   287.2   284.1   286.8   15000000
SBER      2024-01-15 10:05:00  286.8   288.0   286.5   287.5   12000000
```

**Preprocessing:**
- **Aggregation**: Intraday → Daily (first open, max high, min low, last close, sum volume)
- **Resampling**: `.asfreq('D')` with forward-fill for missing weekends/holidays

####  **News Data**
```
Shape: (N_articles, 3)
Columns: [publish_date, title, publication]
Date Range: Continuous stream
Language: Russian/English mixed
```

**Filtering Strategy:**
- **Keyword matching**: 117 tickers × 5-10 keywords each
- **Top-30 selection**: Most recent news per ticker
- **Temporal shift**: +1 day 

#### 3. **Ticker Keywords Dictionary**
```python
TICKER_KEYWORDS = {
    "SBER": ["SBER", "Sberbank", "Bank", "Economics", "Trump", "Putin"],
    # ... 117 tickers total
}
```

**Purpose:**
- Fast pre-filtering before expensive LLM calls
- Captures company names, sector keywords, geopolitical events
- Reduces API calls from ~50K to ~5K

### Data Dimensions Throughout Pipeline

```
RAW DATA:
├─ Candles: (N_intraday_records, 7) → Aggregate → (N_days, 5)
└─ News: (N_articles, 3) → Filter+LLM → (N_relevant_news, 2)

FEATURE ENGINEERING:
├─ Technical Indicators: (N_days, 21)  # 21 technical features
├─ LLM Features: (N_days, 2)           # sentiment + impact
└─ Combined: (N_days, 23)              # Total features

NORMALIZATION:
└─ RobustScaler: (N_days, 23) → (N_days, 23)  # Scaled

SEQUENCE FORMATION:
├─ Input X: (N_samples, 60, 23)        # 60-day windows
├─ Target Y_returns: (N_samples, 20)   # 20-day returns
└─ Target Y_directions: (N_samples, 20) # 20-day directions

BATCHING:
├─ Train Batch: (64, 60, 23)
└─ Val Batch: (128, 60, 23)

MODEL PROCESSING:
├─ Feature Projection: (B, 60, 23) → (B, 60, 128)
├─ Positional Encoding: (B, 60, 128) → (B, 60, 128)
├─ TCN Output: (B, 60, 128)
├─ Transformer Output: (B, 60, 128)
├─ Combined: (B, 60, 128)
├─ Attention Output: (B, 60, 128)
├─ Pooling: (B, 60, 128) → (B, 384)    # last+avg+max
├─ Final Representation: (B, 384) → (B, 128)
├─ Daily Heads Output: (B, 20)
└─ Global Head Output: (B, 20)

OUTPUT:
├─ Predicted Returns: (B, 20)
└─ Predicted Probabilities: (B, 20)

Where: B = Batch size, N = Number of records
```

## Architecture Deep Dive

### Complete Data Flow with Dimensions

```
Input Sequence: (Batch=64, Seq=60, Features=23)
                        ↓
┌───────────────────────────────────────────────┐
│ Feature Projection: Linear(23 → 128)         │
│ Output: (64, 60, 128)                         │
└───────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Positional Encoding                           │
│ PE shape: (1, 5000, 128) - broadcasted        │
│ Output: (64, 60, 128)                         │
└───────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌───────────────────┐         ┌─────────────────────┐
│   TCN Branch      │         │ Transformer Branch  │
│                   │         │                     │
│ Transpose:        │         │ 4 Encoder Layers    │
│ (64,60,128)       │         │ Each layer:         │
│      ↓            │         │ - MultiHead Attn    │
│ (64,128,60)       │         │   8 heads           │
│      ↓            │         │   d_k = 128/8 = 16  │
│ Conv1d Layer 1:   │         │ - FFN (128→512→128) │
│ k=7, d=1          │         │ - LayerNorm         │
│ (64,128,60)       │         │ - Dropout           │
│      ↓            │         │                     │
│ Conv1d Layer 2:   │         │ Output: (64,60,128) │
│ k=7, d=2          │         │                     │
│ (64,128,60)       │         │                     │
│      ↓            │         │                     │
│ Conv1d Layer 3:   │         │                     │
│ k=7, d=4          │         │                     │
│ (64,128,60)       │         │                     │
│      ↓            │         │                     │
│ Transpose back:   │         │                     │
│ (64,60,128)       │         │                     │
└───────────────────┘         └─────────────────────┘
        ↓                               ↓
        └───────────────┬───────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Residual Fusion: TCN + Trans + Original       │
│ (64,60,128) + (64,60,128) + (64,60,128)       │
│ Output: (64, 60, 128)                         │
└───────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Multi-Head Attention (Self-Attention)         │
│ Heads: 8, d_k=16, d_v=16                      │
│ Q,K,V: (64, 60, 128) each                     │
│ Attention scores: (64, 8, 60, 60)             │
│ Output: (64, 60, 128)                         │
└───────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Triple Pooling Strategy                       │
│ ├─ Last: [:, -1, :] → (64, 128)              │
│ ├─ Mean: .mean(dim=1) → (64, 128)            │
│ └─ Max: .max(dim=1) → (64, 128)              │
│ Concatenate: (64, 384)                        │
└───────────────────────────────────────────────┘
                        ↓
┌───────────────────────────────────────────────┐
│ Final Representation Projection               │
│ Linear(384 → 128) + GELU + Dropout            │
│ Output: (64, 128)                             │
└───────────────────────────────────────────────┘
                        ↓
       ┌─────────────────────────────────┌──────────────────────────────┐
       ↓                                 ↓                               ↓
                                  ┌─────────────────┐           ┌──────────────────┐
                                  │ 20 Return Heads │           │ 20 Probability   │
┌─────────────────┐               │ Each:           │           │ Heads            │
│ Global Return   │               │ Linear(128→64)  │           │ Each:            │
│ Head            │               │ GELU            │           │ Linear(128→64)   │
│ Linear(128→64)  │               │ Dropout         │           │ GELU             │
│ GELU            │               │ Linear(64→1)    │           │ Dropout          │
│ Dropout         │               │                 │           │ Linear(64→1)     │
│ Linear(64→20)   │               │ Concat to:      │           │ Sigmoid          │
│ Output: (64,20) │               │ (64, 20)        │           │                  │
└─────────────────┘               │                 │           │ Concat to:       │             
                                  │                 │           │ (64, 20)         │
                                  └─────────────────┘           └──────────────────┘
             ↓                             ↓                              ↓
         ┌───────────────────────────────────────────────────────────────────┐
         │           Final Weighted Combination                              │
         │           Returns: 0.7×Daily + 0.3×Global                         │
         │           Output: (64, 20) for returns                            │
         │                   (64, 20) for probabilities                      │
         └───────────────────────────────────────────────────────────────────┘
```



#### 2. **Temporal Convolutional Network (TCN)**

The TCN component uses dilated causal convolutions to capture temporal patterns at multiple scales:

- **Exponential dilation**: `2^i` for layer `i`, enabling receptive field growth
- **3 layers** with hidden dimension 128
- **Kernel size**: 7 (captures weekly patterns)
- **Residual connections** preserve gradient flow
- **Batch normalization** + ReLU + Dropout for regularization

**Key advantages:**
- Parallel computation (faster than RNNs)
- Stable gradients across long sequences
- Captures multi-scale temporal patterns

#### 3. **Transformer Encoder**

4-layer Transformer with:
- **8 attention heads** for multi-perspective feature extraction
- **512-dimensional feedforward** networks (4x hidden size)
- **GELU activation** for smooth non-linearity
- **Sinusoidal positional encoding** for temporal awareness

**Purpose:**
- Models long-range dependencies (60-day input window)
- Captures complex interactions between features
- Provides global context for predictions

#### 4. **Multi-Head Prediction System**

**Individual Day Heads (20 heads):**
- Each head specializes in predicting a specific future day
- Separate return regression and direction classification
- Allows model to learn day-specific patterns

**Global Head:**
- Predicts all 20 returns simultaneously
- Acts as regularization to maintain coherence
- Final prediction: `0.7 × individual + 0.3 × global`

###  LLM-Enhanced News Feature Extraction

#### News Processing Pipeline

```
Raw News → Keyword Filtering → LLM Analysis → Feature Integration
```

**1. Keyword-Based Pre-filtering:**
- Custom dictionary with ticker-specific keywords
- Includes company names in Russian/English + sector keywords
- Captures geopolitical figures for this period and Russian Market ("Trump", "Putin")

**2. LLM Sentiment Analysis:**
```python
Model: GPT-3.5-turbo via OpenRouter API
Input: News title + news text
Output: {
  "sentiment": -3 to +3,  # Bearish to Bullish
  "impact": 0 to 5        # Market significance
}
```

**3. Temporal Alignment:**
- News features shifted +1 day (account for market reaction delay)
- Aggregated at daily level
- Missing days filled with zeros (no news)

**4. Feature Engineering:**
- Sentiment score (-3 to +3)
- Impact magnitude (0 to 5)
- Weighted integration with market data (weight: 1.0 in training)

##  Feature Engineering

### Technical Indicators (18 features)

**Price-based:**
- Log prices (open, high, low, close)
- Log returns (daily, 20-day volatility)
- Multiple moving averages (MA5, MA20, MA50, MA200)
- Price momentum (10-day)
- Position relative to MAs

**Volume-based:**
- Log volume
- Volume change rate
- Volume MA10 ratio

**Volatility:**
- Rolling standard deviation (20, 30, 100 days)
- Average True Range (ATR)
- Annual return (250 days)

**Technical indicators:**
- RSI (Relative Strength Index, 14-period)
- MACD (12/26 exponential MAs)
- Stochastic oscillator (10-day high/low)

**Market structure:**
- High-low range
- Typical price momentum

### LLM Features (2 features per ticker)
- Sentiment score
- Impact magnitude

**Total input dimension: 23 features**

##  Training Strategy

### Loss Function

**Combined Loss** (α=0.5, β=0.3, γ=0.2):
```
L = α·MSE(returns) + β·BCE(probabilities) + γ·DirectionError
```

**Components:**
1. **MSE Loss**: Regression of continuous returns
2. **BCE Loss**: Binary classification of direction (up/down)
3. **Direction Accuracy**: Penalizes incorrect directional predictions

**Rationale:** Multi-objective optimization improves both magnitude and direction accuracy.

### Optimization

- **Optimizer**: AdamW (weight_decay=1e-5)
- **Learning rate**: 5e-4 with ReduceLROnPlateau scheduler
- **Gradient clipping**: max_norm=1.0 (stability)
- **Early stopping**: patience=5 epochs
- **Batch size**: 64 (train), 128 (validation)

### Data Splitting

```
Training: All data up to (TRAIN_CUTOFF - 200 days)
Validation: Last 200 days before TRAIN_CUTOFF
Test: Post-TRAIN_CUTOFF data
```

**Validation strategy:**
- Time-series split (no future leakage)
- Per-ticker datasets concatenated
- Shared feature scaler across all tickers

## 🔧 Data Preprocessing

### Robust Scaling

Uses `RobustScaler` (robust to outliers):
```
X_scaled = (X - median) / IQR
```

**Why RobustScaler?**
- Financial data has extreme outliers (market crashes, spikes)
- Median/IQR more stable than mean/std
- Preserves relative relationships

### Temporal Consistency

- **Frequency alignment**: Daily resampling with forward-fill
- **Missing data handling**: 
  - Technical indicators: Forward-fill for continuity
  - LLM features: Zero-fill (no news = neutral)
- **Sequence validation**: Ensures no NaN in input windows

## Prediction Pipeline

### Inference Process

1. **Prepare input sequence**: Last 60 days of features
2. **LLM feature weighting**: Apply weight 0.2 (reduce overfitting to news)
3. **Model forward pass**: Generate 20-day return and probability forecasts
4. **Post-processing**:
   - Clip extreme predictions (±3%)
   - Exponential smoothing (α=0.3)
   - Convert log-returns to prices
   - 5-period rolling average for smoothing
   - Recalculate returns from smoothed prices

5. **Final output**: Cumulative returns for 20 days

### Prediction Stabilization

**Clipping:**
```python
pred_returns = np.clip(pred_returns, -0.03, 0.03)
```
Prevents unrealistic daily movements.

**Smoothing:**
```python
pred_returns = pd.Series(pred_returns).ewm(alpha=0.3).mean()
```
Reduces noise and volatility in predictions.

**Price-based reconstruction:**
```python
prices = last_price * np.exp(np.cumsum(log_returns))
smoothed = rolling_mean(prices, window=5)
```
Ensures physical consistency (prices can't be negative).

## Usage

### Requirements

```bash
pip install numpy pandas torch scikit-learn tqdm openai
```

### Configuration

```python
# Data paths
CANDLES_PATH_1 = "path/to/candles.csv"
NEWS_PATH_1 = "path/to/news.csv"

# API key
OPENROUTE_API_KEY = "your_openrouter_api_key"

# Hyperparameters
INPUT_WINDOW = 60      # Days of historical data
PRED_HORIZON = 20      # Forecast length
TRAIN_CUTOFF = '2025-09-08'
```

### Running

```python
# 1. Load data
CANDLES, NEWS = load_and_merge_data()

# 2. Extract LLM features
extractor = NewsFeatureExtractor(NEWS, CANDLES, TICKER_KEYWORDS, 
                                  TRAIN_CUTOFF, OPENROUTE_API_KEY)
LLM_FEATURES = extractor.extract_features(TRAIN_TICKERS)

# 3. Train model
model = ImprovedForecastModel(pred_horizon=20).to(device)
trained_model = train_model(model, train_loader, val_loader)

# 4. Generate forecasts
forecast = generate_forecast(trained_model, df_ticker, llm_ticker, 
                             feature_scaler, TRAIN_CUTOFF)
```

## Output Format

CSV file with columns:
```
ticker, p1, p2, ..., p20
```

Where `p1` to `p20` are cumulative returns for each of the 20 forecast days:
```python
R_cumulative = exp(sum(log_returns)) - 1
```

##  Key Technical Decisions

### 1. **Why Hybrid TCN-Transformer?**

- **TCN**: Efficient for local patterns (daily, weekly cycles)
- **Transformer**: Captures long-term dependencies (monthly, quarterly trends)
- **Combination**: Best of both worlds with residual connections

### 2. **Why Multi-Head Predictions?**

- Each day has unique characteristics (e.g., Monday effect)
- Individual heads specialize without interference
- Global head prevents overfitting to noise

### 3. **Why LLM for News?**

- GPT-3.5 understands context and sarcasm
- Handles multilingual content (Russian financial news)
- Impact scoring captures market significance

### 4. **Why +1 Day Shift for News?**

- Prevents data leakage

### 5. **Why Robust Scaler?**

- Financial data has fat tails (Black Swan events)
- Standard scaler distorted by outliers
- Median/IQR more stable for normalization

## Model Statistics

- **Parameters**: ~2.5M (efficient for GPU training)
- **Input sequence**: 60 days × 23 features
- **Training time**: ~15 minutes on T4 GPU (10 epochs)
- **Inference**: <1 second per ticker
- **Memory**: ~4GB GPU RAM (batch_size=64)

## Performance Considerations

**Strengths:**
- Captures multi-scale temporal patterns
- Leverages both technical and fundamental (news) data
- Handles missing data gracefully
- Produces stable, non-extreme forecasts

**Limitations:**
- Assumes market structure continuity (breaks in regime shifts)
- LLM API calls expensive (rate-limited to 5000)
- Requires substantial historical data 
- Performance degrades during high volatility periods
  
## Visualization of Predicted Price for 20 Days Ahead

This chart presents a **visualization of the predicted price** (forecast value) for 20 trading days ahead for 5 tickers.

![Visualization of Predicted Price for 20 Days Ahead](https://github.com/MakarRybkin/Finam_hack_oct_2025/raw/master/visual_preds(5_tickers).png)

---

## Comparison of Forecast with Real Graphs (20 Days)

This section presents the **real price graphs** for the 5 analyzed tickers over the same 20-day period for which the forecast was built.

![Real Graphs of 5 Tickers for 20 Days](https://github.com/MakarRybkin/Finam_hack_oct_2025/raw/master/real_prices(5_tickers).png)

### Observations on Forecast Accuracy:

The conducted analysis showed that for the shares of the following companies:
* **Lukoil(LKOH)**
* **GMK "Norilsk Nickel" (GMKN)**
* **Novatek(NVTK)**

The forecasts for the short-term period, specifically **5–10 days** ahead, demonstrated high accuracy and were **very similar to the actual** price changes.

Over a longer horizon (10-20 days), deviations can be more significant, which is typical for short-term financial forecasting (this is explained by the absence of news and real data from the previous 5-10 days).

## Potential Improvements

1. **Ensemble Methods**
2. **Working on a Shorter Timeframe (1H, 4H)** and forecasting for 20 hours and 100 hours ahead, respectively.
3. **Dynamic weighting**: Adjust LLM feature importance per ticker/sector
4. **Multi-resolution**: Add intraday data for short-term predictions
5. **Risk modeling**: Predict volatility alongside returns
6. **Reinforcement learning**: Portfolio optimization layer

## References

- **TCN**: Bai et al. (2018) - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- **Transformer**: Vaswani et al. (2017) - "Attention Is All You Need"
- **Financial forecasting**: Fischer & Krauss (2018) - "Deep learning with long short-term memory networks for financial market predictions"

## License

MIT License - Free for academic and commercial use.



**Note**: This system is for research purposes. Always consult financial advisors before making investment decisions.
