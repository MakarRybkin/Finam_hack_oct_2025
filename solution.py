import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import RobustScaler
import warnings
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import json
import re
import time
import json
from openai import (
    OpenAI,
    APIStatusError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError
)

warnings.filterwarnings('ignore')

SEED = 42
CANDLES_PATH_1 = "/kaggle/input/finam-hackathon/candles.csv"
CANDLES_PATH_2 = "/kaggle/input/finam-hackathon/candles_2.csv"
NEWS_PATH_1 = "/kaggle/input/finam-hackathon/news.csv"
NEWS_PATH_2 = "/kaggle/input/finam-hackathon/news_2.csv"
TRAIN_CUTOFF = pd.Timestamp('2025-09-08')
INPUT_WINDOW = 60
PRED_HORIZON = 20
INPUT_DIM = 21
OPENROUTE_API_KEY = "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"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n 1. Загрузка и объединение данных...")


def load_and_merge_data():
    candles_1 = pd.read_csv(CANDLES_PATH_1)
    candles_2 = pd.read_csv(CANDLES_PATH_2)
    candles_df = pd.concat([candles_1, candles_2], ignore_index=True)
    news_1 = pd.read_csv(NEWS_PATH_1)
    news_2 = pd.read_csv(NEWS_PATH_2)
    news_df = pd.concat([news_1, news_2], ignore_index=True)
    if 'begin' in candles_df.columns:
        candles_df["begin"] = pd.to_datetime(candles_df["begin"])
        candles_df = candles_df.sort_values(['ticker', 'begin']).reset_index(drop=True)
    if 'publish_date' in news_df.columns:
        news_df["publish_date"] = pd.to_datetime(news_df["publish_date"])

    return candles_df, news_df


CANDLES, NEWS = load_and_merge_data()
TRAIN_TICKERS = CANDLES[CANDLES['begin'] <= TRAIN_CUTOFF]['ticker'].unique()
TICKER_KEYWORDS = {
    "GAZP": ["Газпром", "Gazprom", "ГАЗП", "Gazprom Neft", "Нефть", "Трамп", "Путин"],
    "SBER": ["Сбербанк", "Sberbank", "СБЕР", "Сбер", "Банк", "Ключевая ставка", "Трамп", "Путин"],
    "SBERP": ["Сбербанк-п", "Sberbank-p", "СБЕР-п", "Банк", "Ключевая ставка", "Трамп", "Путин"],
    "LKOH": ["Лукойл", "Lukoil", "ЛУКОЙЛ", "Трамп", "Путин"],
    "GMKN": ["Норникель", "Nornickel", "ГМК", "Трамп", "Путин"],
    "YNDX": ["Яндекс", "Yandex", "YNDX", "Трамп", "Путин"],
    "VTBR": ["ВТБ", "VTB", "ВТБ Банк", "Банк", "Ключевая ставка", "Трамп", "Путин"],
    "ROSN": ["Роснефть", "Rosneft", "РОСНЕФТЬ", "Трамп", "Путин"],
    "NVTK": ["НОВАТЭК", "Novatek", "Трамп", "Путин"],
    "SIBN": ["Газпром нефть", "Gazprom Neft", "ГАЗПРОМ НЕФТЬ", "Трамп", "Путин"],
    "PHOR": ["ФосАгро", "PhosAgro", "ФОСАГРО", "Трамп", "Путин"],
    "PLZL": ["Полюс Золото", "Polyus", "ПОЛЮС", "Трамп", "Путин"],
    "MTSS": ["МТС", "MTS", "МТС Банк", "Трамп", "Путин"],
    "MGNT": ["Магнит", "Magnit", "МАГНИТ", "Трамп", "Путин"],
    "CHMF": ["Северсталь", "Severstal", "СЕВЕРСТАЛЬ", "Трамп", "Путин"],
    "NLMK": ["НЛМК", "NLMK", "Новолипецкий металлургический комбинат", "Трамп", "Путин"],
    "ALRS": ["АЛРОСА", "Alrosa", "Трамп", "Путин"],
    "RUAL": ["РУСАЛ", "Rusal", "Трамп", "Путин"],
    "AFKS": ["АФК Система", "AFK Sistema", "СИСТЕМА", "Трамп", "Путин"],
    "MOEX": ["Московская биржа", "MOEX", "БИРЖА", "Трамп", "Путин"],
    "HYDR": ["РусГидро", "RusHydro", "РУСГИДРО", "Трамп", "Путин"],
    "IRAO": ["Интер РАО", "Inter RAO", "ИНТЕР РАО", "Трамп", "Путин"],
    "AFLT": ["Аэрофлот", "Aeroflot", "АЭРОФЛОТ", "Трамп", "Путин"],
    "SNGS": ["Сургутнефтегаз", "Surgutneftegas", "СНГ", "Трамп", "Путин"],
    "SNGSP": ["Сургутнефтегаз-п", "Surgutneftegas-p", "СНГ-п", "Трамп", "Путин"],
    "TATN": ["Татнефть", "Tatneft", "ТАТНЕФТЬ", "Трамп", "Путин"],
    "TATNP": ["Татнефть-п", "Tatneft-p", "ТАТНЕФТЬ-п", "Трамп", "Путин"],
    "VTBRP": ["ВТБ-п", "VTB-p", "ВТБ Банк-п", "Трамп", "Путин"],
    "MGNP": ["Магнит-п", "Magnit-p", "МАГНИТ-п", "Трамп", "Путин"],
    "TRNFP": ["Транснефть", "Transneft", "ТРАНСНЕФТЬ", "Трамп", "Путин"],
    "TRNF": ["Транснефть", "Transneft", "ТРАНСНЕФТЬ", "Трамп", "Путин"],
    "URKA": ["Уралкалий", "Uralkali", "УРАЛКАЛИЙ", "Трамп", "Путин"],
    "URKAP": ["Уралкалий-п", "Uralkali-p", "УРАЛКАЛИЙ-п", "Трамп", "Путин"],
    "BANE": ["Башнефть", "Bashneft", "Трамп", "Путин"],
    "FESH": ["ДВМП", "Дальневосточное морское пароходство", "FESCO", "Трамп", "Путин"],
    "UWGN": ["ОВК", "Объединенная вагонная компания", "ОВК", "Трамп", "Путин"],
    "T": ["Тинькофф Банк", "Tinkoff", "ТИНЬКОФФ", "Т-Банк", "Т", "Банк", "Ключевая ставка", "IT", "Трамп", "Путин"],
    "VKCO": ["VK Company", "VK", "ВКОНТАКТЕ", "IT", "Трамп", "Путин"],
    "PIKK": ["ПИК-специализированный застройщик", "PIK", "ПИК", "Трамп", "Путин"],
    "SMLT": ["Самолет", "Samolet", "САМОЛЕТ", "Трамп", "Путин"],
    "POSI": ["Positive Technologies", "ПОЗИТИВ", "Трамп", "Путин"],
    "FIVE": ["X5 Retail Group", "X5", "Х5", "Трамп", "Путин"],
    "SVCB": ["Совкомбанк", "Sovcombank", "СОВКОМБАНК", "Трамп", "Путин"],
    "SVCBP": ["Совкомбанк-п", "Sovcombank-p", "СОВКОМБАНК-п", "Трамп", "Путин"],
    "CIAN": ["ЦИАН", "CIAN Group", "ЦИАН ГРУПП", "Трамп", "Путин"],
    "HHRU": ["HeadHunter", "Хедхантер", "HH", "IT", "Трамп", "Путин"],
    "DELI": ["Делимобиль", "Delimobil", "ДЕЛИМОБИЛЬ", "Трамп", "Путин"],
    "DIAS": ["Диасофт", "Diasoft", "DIASOFT", "Трамп", "Путин"],
    "EVI": ["Европлан", "Europlan", "ЕВРОПЛАН", "Трамп", "Путин"],
    "MTBC": ["МТС Банк", "MTS Bank", "МТС-БАНК", "Трамп", "Путин"],
    "OZON": ["Озон", "Ozon Holdings", "OZON", "IT", "Трамп", "Путин"],
    "DATA": ["Аренадата", "Arenadata", "DATA", "Трамп", "Путин"],
    "SPBE": ["СПБ Биржа", "SPB Exchange", "СПБ", "Трамп", "Путин"],
    "CBOM": ["МКБ", "Moscow Credit Bank", "МОСКОВСКИЙ КРЕДИТНЫЙ БАНК", "Трамп", "Путин"],
    "PRMB": ["Промсвязьбанк", "PSB", "ПСБ", "Трамп", "Путин"],
    "ABIO": ["Астра", "ГК Астра", "Astra Group", "ASTRA", "Трамп", "Путин"],
    "SOFL": ["Софтлайн", "Softline", "СОФТЛАЙН", "Трамп", "Путин"],
    "DELPO": ["Депозитарный расписки", "Депозитарные расписки", "ГДР", "Трамп", "Путин"],
    "RASP": ["Распадская", "Raspadskaya", "РАСПАДСКАЯ", "Трамп", "Путин"],
    "POLY": ["Полиметалл", "Polymetal", "ПОЛИМЕТАЛЛ", "Трамп", "Путин"],
    "RUALP": ["РУСАЛ-п", "Rusal-p"],
    "MAGN": ["ММК", "MMK", "Магнитогорский металлургический комбинат", "Трамп", "Путин"],
    "MSNG": ["Мечел", "Mechel", "МЕЧЕЛ", "Трамп", "Путин"],
    "TRMK": ["ТМК", "Трубная металлургическая компания", "TMK", "Трамп", "Путин"],
    "KMAZ": ["КАМАЗ", "KAMAZ", "КАМАЗ", "Трамп", "Путин"],
    "VSMO": ["ВСМПО-Ависма", "VSMPO-AVISMA", "Трамп", "Путин"],
    "AKRN": ["Акрон", "Acron", "Трамп", "Путин"],
    "KZOS": ["Казаньоргсинтез", "Kazanorgsintez", "КОС", "Трамп", "Путин"],
    "GCHE": ["Группа Черкизово", "Cherkizovo Group", "ЧЕРКИЗОВО", "Трамп", "Путин"],
    "RZSP": ["Русагро", "Rusagro", "РУСАГРО", "Трамп", "Путин"],
    "RZSPP": ["Русагро-п", "Rusagro-p", "РУСАГРО-п", "Трамп", "Путин"],
    "OKEY": ["О'КЕЙ", "O'KEY", "ОКЕЙ", "Трамп", "Путин"],
    "APTK": ["Аптечная сеть 36,6", "APTEKA 36,6", "АПТЕКА", "Трамп", "Путин"],
    "DIXY": ["Дикси Групп", "DIXY Group", "ДИКСИ", "Трамп", "Путин"],
    "ABRD": ["Абрау-Дюрсо", "Abrau-Durso", "АБРАУ", "Трамп", "Путин"],
    "MRKC": ["Россети Центр", "MRSK Center", "МРСК ЦЕНТРА", "Трамп", "Путин"],
    "MRKU": ["Россети Урал", "MRSK Urala", "МРСК УРАЛА", "Трамп", "Путин"],
    "MRKP": ["Россети Центр и Приволжье", "MRSK Center & Privolzhye", "МРСК ЦИП", "Трамп", "Путин"],
    "MRKV": ["Россети Волга", "MRSK Volga", "МРСК ВОЛГА", "Трамп", "Путин"],
    "MSRS": ["Россети Московский регион", "МОЭСК", "MSRS", "Трамп", "Путин"],
    "MRKK": ["Россети Северный Кавказ", "MRK-SK", "РОССЕТИ СК", "Трамп", "Путин"],
    "MRKY": ["Россети Юг", "MRK-Yug", "РОССЕТИ ЮГ", "Трамп", "Путин"],
    "MRKZ": ["Россети Северо-Запад", "MRSK SZ", "МРСК СЗ", "Трамп", "Путин"],
    "FEES": ["ФСК ЕЭС", "FSK EES", "ФСК", "Трамп", "Путин"],
    "OGKB": ["ОГК-2", "OGK-2", "Трамп", "Путин"],
    "TGKA": ["ТГК-1", "TGK-1", "Трамп", "Путин"],
    "TGLD": ["ТГК-1", "ТГК-1", "TGC-1", "Трамп", "Путин"],
    "TGKB": ["ТГК-2", "TGK-2", "Трамп", "Путин"],
    "TGKN": ["ТГК-14", "TGK-14", "Трамп", "Путин"],
    "BELU": ["Белуга Групп", "BELUGA Group", "БЕЛУГА", "Трамп", "Путин"],
    "ETLN": ["Эталон", "Etalon Group", "ЭТАЛОН", "Трамп", "Путин"],
    "SGZH": ["Сегежа Групп", "Segezha Group", "СЕГЕЖА", "Трамп", "Путин"],
    "LSRG": ["ЛСР", "LSR Group", "Группа ЛСР", "Трамп", "Путин"],
    "GEMC": ["Мать и дитя", "MD Medical Group", "МАТЬ И ДИТЯ", "Трамп", "Путин"],
    "AQUA": ["Аквакультура", "РусАквакультура", "AQUA", "Трамп", "Путин"],
    "RTKM": ["Ростелеком", "Rostelecom", "РОСТЕЛЕКОМ", "Трамп", "Путин"],
    "RTKMP": ["Ростелеком-п", "Rostelecom pref", "РОСТЕЛЕКОМ-п", "Трамп", "Путин"],
    "RENI": ["Ренессанс Страхование", "РЕНЕССАНС СТРАХОВАНИЕ", "Трамп", "Путин"],
    "ZAYM": ["Займер", "Zaymer", "ЗАЙМЕР", "Трамп", "Путин"],
    "ZAYMP": ["Займер-п", "Zaymer-p", "ЗАЙМЕР-п", "Трамп", "Путин"],
    "PHST": ["Фармсинтез", "Pharmasynthez", "ФАРМСИНТЕЗ", "Трамп", "Путин"],
    "RSG": ["РусГидро", "RusHydro", "РУСГИДРО", "Трамп", "Путин"],
    "LSNGP": ["Ленэнерго-п", "Lenenergo-p", "ЛЕНЭНЭРГО-п", "Трамп", "Путин"],
    "GAZT": ["ГАЗ-Тек", "GAZ-Tek", "ГАЗТЕК", "Трамп", "Путин"],
    "IRKT": ["Иркут", "Irkut", "Трамп", "Путин"],
    "UNAC": ["ОАК", "Объединённая авиастроительная корпорация", "Трамп", "Путин"],
    "NKNC": ["Нижнекамскнефтехим", "Nizhnekamskneftekhim", "Трамп", "Путин"],
    "CHKZ": ["Челябинский кузнечно-прессовый завод", "ЧКПЗ", "Трамп", "Путин"],
    "CHMK": ["Челябинский металлургический комбинат", "ЧМК", "Трамп", "Путин"],
    "ALNU": ["Алмазная промышленность", "ALNU", "Трамп", "Путин"],
    "OKB": ["ОК 'Русь'", "OK", "OKB", "Трамп", "Путин"],
    "ISKJ": ["ГК Самолет", "Samolet Group", "САМОЛЕТ-ИСК", "Трамп", "Путин"],
    "AKB": ["АК БАРС", "Ak Bars Bank", "АКБАРС", "Трамп", "Путин"]
}


def call_llm_for_features(news_text: str, ticker: str, api_key: str) -> tuple[int, int]:
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    system_prompt = (
        "You are an expert financial analyst. Analyze the following news article for "
        f"stock ticker {ticker}. Output the result strictly in JSON format: "
        '{"sentiment": N, "impact": M}, where N is the sentiment (integer from -3 to 3, '
        'and M is the market impact (integer from 0 to 5). '
        "Do not include any other text or explanation."
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print("\n" + "=" * 50)
            print(f"[{ticker}]  Подготовка запроса к LLM:")
            print(news_text[:500] + ("..." if len(news_text) > 500 else ""))
            print("=" * 50)

            completion = client.chat.completions.create(
                extra_headers={"OpenRouter-App-Title": "Financial Feature Extractor"},
                model="openai/gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": news_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            response_text = completion.choices[0].message.content
            data = json.loads(response_text)

            print(f"[{ticker}]  Ответ LLM:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return data.get("sentiment", 0), data.get("impact", 0)

        except (APIStatusError, APIConnectionError, RateLimitError, json.JSONDecodeError) as e:
            print(f"[{ticker}]  Ошибка при вызове API: {type(e).__name__} - {e}")
            if attempt < max_retries - 1:
                print(f"[{ticker}] Повторная попытка ({attempt + 2}/{max_retries})...")
                time.sleep(2 ** attempt)
                continue
            return 0, 0
        except Exception as e:
            print(f"[{ticker}]  Неожиданная ошибка: {type(e).__name__} - {e}")
            return 0, 0
    return 0, 0


class NewsFeatureExtractor:
    def __init__(self, news_df, candles_df, ticker_keywords, train_cutoff, api_key, max_api_calls=5000):
        self.news_df = news_df
        self.candles_df = candles_df
        self.ticker_keywords = ticker_keywords
        self.train_cutoff = train_cutoff
        self.api_key = api_key
        self.max_api_calls = max_api_calls
        self.llm_results = {}
        self.api_calls = 0
        self.log_records = []

    def extract_features(self, train_tickers):
        print(f"📊 Запуск анализа новостей для тикеров из TRAIN...")

        for ticker, keywords in self.ticker_keywords.items():
            if ticker not in train_tickers:
                continue

            if self.api_calls >= self.max_api_calls:
                print(" Достигнут лимит API вызовов")
                break

            keyword_pattern = '|'.join([re.escape(k) for k in keywords])
            news_filtered = self.news_df[
                self.news_df['title'].str.contains(keyword_pattern, case=False, na=False)
            ].copy()

            if news_filtered.empty:
                continue

            news_filtered.sort_values('publish_date', inplace=True, ascending=False)
            recent_news = news_filtered.head(30)

            results = []
            for _, row in recent_news.iterrows():
                if self.api_calls >= self.max_api_calls:
                    break

                news_text = f"Title: {row['title']}\nPublication: {row['publication']}"
                response = call_llm_for_features(news_text, ticker, self.api_key)
                self.api_calls += 1

                self.log_records.append({
                    'ticker': ticker,
                    'date': row['publish_date'],
                    'title': row['title'],
                    'publication': row['publication'],
                    'llm_response': response
                })

                if response:
                    results.append(response)

            if results:
                self.llm_results[ticker] = results

        if self.log_records:
            import pandas as pd
            pd.DataFrame(self.log_records).to_csv('llm_news_logs.csv', index=False, encoding='utf-8-sig')
            print(f" LLM анализ новостей сохранён ({len(self.log_records)} записей)")

        return self.llm_results


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=(dilation_size * (kernel_size - 1)) // 2,
                dilation=dilation_size
            )
            self.layers.append(nn.Sequential(
                conv,
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.output_channels = num_channels[-1]

    def forward(self, x):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2)


class ImprovedForecastModel(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=128, num_layers=4,
                 num_heads=8, dropout=0.2, pred_horizon=20):
        super().__init__()

        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim

        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.tcn = TemporalConvNet(
            hidden_dim,
            [hidden_dim, hidden_dim, hidden_dim],
            kernel_size=7,
            dropout=dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.final_repr_proj = nn.Linear(3 * hidden_dim, hidden_dim)

        self.return_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(pred_horizon)
        ])

        self.prob_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(pred_horizon)
        ])

        self.global_return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_horizon)
        )

    def forward(self, x):
        x = self.feature_proj(x)
        x = self.pos_encoder(x)

        tcn_out = self.tcn(x)
        trans_out = self.transformer(x)
        combined = tcn_out + trans_out + x
        attn_output, _ = self.attention(combined, combined, combined)
        last_state = attn_output[:, -1, :]
        avg_state = attn_output.mean(dim=1)
        max_state = attn_output.max(dim=1)[0]
        pooled = torch.cat([last_state, avg_state, max_state], dim=-1)
        final_repr = self.final_repr_proj(pooled)

        daily_returns = []
        daily_probs = []

        for i in range(self.pred_horizon):
            daily_returns.append(self.return_heads[i](final_repr))
            daily_probs.append(self.prob_heads[i](final_repr))

        daily_returns = torch.cat(daily_returns, dim=1)
        daily_probs = torch.cat(daily_probs, dim=1)
        global_returns = self.global_return_head(final_repr)
        final_returns = 0.7 * daily_returns + 0.3 * global_returns

        return final_returns, daily_probs


class ImprovedTimeSeriesDataset(Dataset):
    def __init__(self, df, llm_features_df, seq_len=60, pred_horizon=20,
                 feature_scaler=None, is_train=True, llm_weight=1):

        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.is_train = is_train
        self.llm_weight = llm_weight

        df_agg = df.groupby("begin").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).reset_index()
        df_agg = df_agg.set_index("begin").asfreq("D").ffill().reset_index()

        # Сдвигаем новости на +1 день
        llm_features_df_shifted = llm_features_df.copy()
        llm_features_df_shifted['begin'] = llm_features_df_shifted['begin'] + pd.Timedelta(days=1)

        llm_features_df_shifted = llm_features_df_shifted.drop_duplicates(subset=['ticker', 'begin'])
        df_agg = pd.merge(df_agg, llm_features_df_shifted.drop(columns='ticker', errors='ignore'),
                          on='begin', how='left')

        llm_cols = [col for col in llm_features_df_shifted.columns if col not in ['ticker', 'begin']]
        for col in llm_cols:
            df_agg[col] = df_agg[col].fillna(0.0)

        # Создаём признаки
        self.features = self._create_features(df_agg, llm_cols)
        if is_train:
            self.targets = self._create_targets(df_agg)
        else:
            self.targets = None

        self.close_prices = df_agg['close'].values

        # Скейлер
        if feature_scaler is None:
            self.feature_scaler = RobustScaler()
            if self.features.size > 0:
                self.features = self.feature_scaler.fit_transform(self.features)
        else:
            self.feature_scaler = feature_scaler
            if self.features.size > 0:
                self.features = self.feature_scaler.transform(self.features)

        self.valid_indices = self._get_valid_indices()

    def _create_features(self, df, llm_cols):
        features = []
        cols = ['open', 'high', 'low', 'close', 'volume']

        features.append(np.log(df['open']).values)
        features.append(np.log(df['high']).values)
        features.append(np.log(df['low']).values)
        features.append(np.log(df['close']).values)
        features.append(np.log(df['volume'] + 1).values)

        log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        features.append(log_returns.values)
        features.append(log_returns.rolling(20).std().fillna(0).values)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50).values / 100.0)

        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()
        features.append((ma5 / ma20 - 1).fillna(0).values)

        features.append((df['volume'] / df['volume'].shift(1) - 1).fillna(0).values)
        features.append(((df['high'] - df['low']) / df['close']).values)

        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        features.append((macd / df['close']).fillna(0).values * 10)

        high_low = df['high'] - df['low']
        high_prevclose = np.abs(df['high'] - df['close'].shift(1))
        low_prevclose = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.DataFrame({'hl': high_low, 'hpc': high_prevclose, 'lpc': low_prevclose}).max(axis=1)
        atr = tr.rolling(14).mean()
        features.append((atr / df['close']).fillna(0).values * 10)

        momentum = df['close'].diff(10) / df['close'].shift(10)
        features.append(momentum.fillna(0).values)

        features.append(df['close'].rolling(30).std().fillna(0).values / df['close'].mean())

        vol_ma10 = df['volume'].rolling(10).mean()
        features.append((df['volume'] / vol_ma10 - 1).fillna(0).values)

        high_10 = df['high'].rolling(10).max()
        low_10 = df['low'].rolling(10).min()
        features.append(((df['close'] - low_10) / (high_10 - low_10 + 1e-8)).fillna(0.5).values)

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        features.append((typical_price / typical_price.shift(1) - 1).fillna(0).values)

        ma50 = df['close'].rolling(50).mean()
        features.append((df['close'] / ma50 - 1).fillna(0).values)

        for col in llm_cols:
            features.append(df[col].values * self.llm_weight)

        current_dim = len(features)
        if current_dim < INPUT_DIM:
            for _ in range(INPUT_DIM - current_dim):
                features.append(np.zeros(len(df)))

        features = features[:INPUT_DIM]

        return np.column_stack(features)

    def _create_targets(self, df):
        returns = []
        directions = []

        for h in range(1, self.pred_horizon + 1):
            if len(df['close']) <= h:
                ret = np.full(len(df), np.nan)
            else:
                ret = (df['close'].shift(-h) / df['close'] - 1).values

            returns.append(ret)
            directions.append((ret > 0).astype(float))

        return {
            'returns': np.column_stack(returns),
            'directions': np.column_stack(directions)
        }

    def _get_valid_indices(self):
        valid = []

        if self.features.size == 0:
            return []

        end_idx = len(self.features) - self.pred_horizon if self.is_train else len(self.features)

        for i in range(self.seq_len, end_idx):
            if not np.isnan(self.features[i - self.seq_len:i]).any():
                if self.is_train and self.targets:
                    if not np.isnan(self.targets['returns'][i]).any():
                        valid.append(i)
                else:
                    valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        X = torch.FloatTensor(self.features[i - self.seq_len:i])

        if self.is_train and self.targets:
            y_returns = torch.FloatTensor(self.targets['returns'][i])
            y_directions = torch.FloatTensor(self.targets['directions'][i])
            return X, y_returns, y_directions
        else:
            return X, torch.zeros(self.pred_horizon), torch.zeros(self.pred_horizon)


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, pred_returns, pred_probs, true_returns, true_directions):
        loss_mse = self.mse(pred_returns, true_returns)
        loss_bce = self.bce(pred_probs, true_directions)

        pred_directions = (pred_returns > 0).float()
        loss_dir = 1.0 - (pred_directions == true_directions).float().mean()

        return self.alpha * loss_mse + self.beta * loss_bce + self.gamma * loss_dir


def train_model(model, train_loader, val_loader, epochs=15, lr=1e-4, patience=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    criterion = CombinedLoss()
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} (Train)")
        for X, y_returns, y_directions in pbar:
            X = X.to(device)
            y_returns = y_returns.to(device)
            y_directions = y_directions.to(device)

            optimizer.zero_grad()
            pred_returns, pred_probs = model(X)
            loss = criterion(pred_returns, pred_probs, y_returns, y_directions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_batches

        model.eval()
        val_loss = 0.0
        val_mse_sum = 0.0
        val_bce_sum = 0.0
        val_dir_acc_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for X, y_returns, y_directions in val_loader:
                X = X.to(device)
                y_returns = y_returns.to(device)
                y_directions = y_directions.to(device)

                pred_returns, pred_probs = model(X)

                loss = criterion(pred_returns, pred_probs, y_returns, y_directions)
                val_loss += loss.item()

                val_mse_sum += mse_criterion(pred_returns, y_returns).item()
                val_bce_sum += bce_criterion(pred_probs, y_directions).item()
                pred_directions = (pred_returns > 0).float()
                val_dir_acc_sum += (pred_directions == y_directions).float().mean().item()

                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_mse = val_mse_sum / val_batches
        avg_val_bce = val_bce_sum / val_batches
        avg_val_acc = val_dir_acc_sum / val_batches

        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MSE: {avg_val_mse:.5f} | Val BCE: {avg_val_bce:.4f} | Val Acc: {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if os.path.exists('best_model.pt'):
        model.load_state_dict(torch.load('best_model.pt'))
    return model


extractor = NewsFeatureExtractor(
    NEWS, CANDLES, TICKER_KEYWORDS,
    TRAIN_CUTOFF, OPENROUTE_API_KEY
)
LLM_FEATURES = extractor.extract_features(train_tickers=TRAIN_TICKERS)

llm_rows = []
for ticker, results_list in LLM_FEATURES.items():
    for i, (sentiment, impact) in enumerate(results_list):
        date = extractor.log_records[i]['date'] if i < len(extractor.log_records) else None
        llm_rows.append({
            'ticker': ticker,
            'begin': date,
            'sentiment': sentiment,
            'impact': impact
        })

LLM_FEATURES_DF = pd.DataFrame(llm_rows)
LLM_FEATURES_DF['begin'] = pd.to_datetime(LLM_FEATURES_DF['begin'])

ticker_data = {}
for ticker in TRAIN_TICKERS:
    df_ticker = CANDLES[CANDLES['ticker'] == ticker].copy()
    llm_ticker = LLM_FEATURES_DF[LLM_FEATURES_DF['ticker'] == ticker].copy()

    df_train = df_ticker[df_ticker['begin'] <= TRAIN_CUTOFF].copy()
    val_start = TRAIN_CUTOFF - pd.Timedelta(days=INPUT_WINDOW)
    df_val = df_ticker[df_ticker['begin'] >= val_start].copy()

    ticker_data[ticker] = {
        'train': df_train,
        'val': df_val,
        'llm': llm_ticker
    }

val_window = 200

train_datasets = []
val_datasets = []
feature_scaler = None

for i, ticker in enumerate(TRAIN_TICKERS):
    data = ticker_data[ticker]

    if len(data['train']) < INPUT_WINDOW + PRED_HORIZON:
        continue

    df_train = data['train'].iloc[:-val_window]
    train_ds = ImprovedTimeSeriesDataset(
        df_train,
        data['llm'],
        seq_len=INPUT_WINDOW,
        pred_horizon=PRED_HORIZON,
        is_train=True
    )
    if i == 0:
        feature_scaler = train_ds.feature_scaler
    if len(train_ds) > 0:
        train_datasets.append(train_ds)

    df_val = data['train'].iloc[-val_window:]
    val_ds = ImprovedTimeSeriesDataset(
        df_val,
        data['llm'],
        seq_len=INPUT_WINDOW,
        pred_horizon=PRED_HORIZON,
        feature_scaler=feature_scaler,
        is_train=True
    )
    if len(val_ds) > 0:
        val_datasets.append(val_ds)

train_dataset = ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = ConcatDataset(val_datasets) if val_datasets else None
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) if val_dataset else None

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset) if val_dataset else 0}")

print("\n Обучение модели")

model = ImprovedForecastModel(
    input_dim=INPUT_DIM,
    pred_horizon=PRED_HORIZON
).to(device)

print(f"Model on: {device}")

trained_model = train_model(
    model,
    train_loader,
    val_loader,
    epochs=10,
    lr=5e-4,
    patience=5
)

print("\n Генерация прогнозов")


def generate_forecast(model, df_ticker, llm_ticker, feature_scaler, date_t, clip_pct=0.05, smooth_window=3):
    """
    Генерирует прогноз на следующие PRED_HORIZON дней, используя последние INPUT_WINDOW дней.
    Применяется скользящее среднее сглаживание и ограничение экстремумов.
    """
    model.eval()

    df_cut = df_ticker[df_ticker['begin'] <= date_t].copy()
    if len(df_cut) < INPUT_WINDOW:
        return {'R': np.zeros(PRED_HORIZON), 'P': np.zeros(PRED_HORIZON), 'prices': np.zeros(PRED_HORIZON)}

    ds = ImprovedTimeSeriesDataset(
        df_cut,
        llm_ticker,
        seq_len=INPUT_WINDOW,
        pred_horizon=PRED_HORIZON,
        feature_scaler=feature_scaler,
        is_train=False
    )

    if len(ds) == 0:
        return {'R': np.zeros(PRED_HORIZON), 'P': np.zeros(PRED_HORIZON), 'prices': np.zeros(PRED_HORIZON)}

    X, _, _ = ds[-1]
    X = X.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_log_returns, pred_probs = model(X)

    pred_log_returns = pred_log_returns.cpu().numpy().flatten()

    if smooth_window > 1:
        pred_log_returns = pd.Series(pred_log_returns).rolling(window=smooth_window, min_periods=1,
                                                               center=True).mean().values

    pred_log_returns = np.clip(pred_log_returns, -clip_pct, clip_pct)

    last_price = df_cut['close'].iloc[-1]
    pred_prices = last_price * np.exp(np.cumsum(pred_log_returns))

    return {
        'R': pred_log_returns,
        'P': pred_probs.cpu().numpy().flatten(),
        'prices': pred_prices
    }


all_train_tickers = TRAIN_TICKERS
submission_rows = []

for ticker in tqdm(all_train_tickers, desc="Generating forecasts"):
    df_ticker = CANDLES[CANDLES['ticker'] == ticker].copy()
    llm_ticker = LLM_FEATURES_DF[LLM_FEATURES_DF['ticker'] == ticker].copy()

    df_cut = df_ticker[df_ticker['begin'] <= TRAIN_CUTOFF].copy()

    if len(df_cut) < INPUT_WINDOW:
        R = np.zeros(PRED_HORIZON)
    else:
        forecast = generate_forecast(
            trained_model,
            df_cut,
            llm_ticker,
            feature_scaler,
            TRAIN_CUTOFF
        )
        daily_log_r = forecast['R']
        R = np.exp(np.cumsum(daily_log_r)) - 1

    submission_rows.append([ticker] + R.tolist())

columns = ['ticker'] + [f'p{i + 1}' for i in range(PRED_HORIZON)]
submission_df = pd.DataFrame(submission_rows, columns=columns).set_index('ticker')

print(f"\n Прогнозы готовы для {len(submission_df)} тикеров")
print(submission_df.head())

submission_df.to_csv('forecast_results.csv', index=True)
print(f" Сохранено в: forecast_results.csv")
