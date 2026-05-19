# PLC Digital Twin

Цифровой двойник промышленного контроллера на Raspberry Pi 5. Магистерская работа.

Pi 5 в роли ПЛК: читает датчики, управляет реле и серво. Taipy-дашборд показывает телеметрию в реальном времени и принимает команды от оператора. AI-воркер в фоне даёт прогноз температуры и влажности на 15 мин - 24 ч. Раз в 5 минут хэш телеметрии уходит в Hedera HCS, само содержимое - в IPFS через Pinata: задним числом подделать журнал не получится.

## Стек

- Python 3.12 в `uv`-venv с `--system-site-packages` (нужен доступ к apt-пакетам `rpi-lgpio`, `troykahat`).
- Celery 5 + Redis: три воркера (`hw` solo, `ai` prefork, `ext` threads) + beat. APScheduler внутри hw-воркера для 1 Гц опроса датчиков.
- SQLite WAL по умолчанию, опционально Postgres (`DB_BACKEND=postgres`).
- Taipy 4.1 + plotly, тёмная тема.
- AI: `xgboost`, `lightgbm`, `prophet`, `lstm`, `nbeats`. Обучение на внешней машине, инференс на Pi.
- Pinata IPFS + Hedera Consensus Service (testnet).

## Железо

Raspberry Pi 5 8GB, Raspberry Pi OS Trixie. Troyka HAT (Amperka) поверх - STM32-расширитель ADC/PWM на I2C `0x2A`.

Датчики: BME280/BMP280 (`0x76`), BH1750 (`0x23`), INA219 (`0x40`), DS18B20 (1-Wire BCM 4), PIR HC-SR501 (BCM 23), MQ-2 (через STM32 A0). Актуаторы: 4 реле SRD-05VDC (HIGH-trigger), светофор из 3 LED, 2 серво SG90, RS-485 (MAX485 через UART0). Дисплей SSD1306 (`0x3C`).

Жёлтая колодка HAT нумеруется в WiringPi, в коде - BCM через `gpiozero` + `lgpio`. На Pi 5 `wiringpi` и `RPi.GPIO` не работают (чип RP1), вместо них `rpi-lgpio`. `troykahat` пропатчен под `smbus2` - без патча падает на импорте `wiringpi`. См. `scripts/patch_troykahat_for_pi5.sh`.

INA219 подключён high-side: HAT 5V -> реле -> VIN+ -> шунт -> VIN- -> нагрузка. Перепутаны VIN+/VIN- -> ток в логе уйдёт в минус, в коде `abs` не применять.

## Установка

```bash
pip install --user uv
uv python install 3.12
uv venv ~/.venv-plc-twin --python 3.12 --system-site-packages
echo "/usr/lib/python3/dist-packages" \
  > ~/.venv-plc-twin/lib/python3.12/site-packages/debian.pth
ln -sf ~/.venv-plc-twin .venv

uv sync --extra postgres --group ai --group dev
VENV=~/.venv-plc-twin bash scripts/patch_troykahat_for_pi5.sh

cp .env.example .env
# заполнить PINATA_JWT, HEDERA_OPERATOR_ID/KEY/TOPIC_ID

sudo apt install redis-server
```

`HEDERA_TOPIC_ID` создаётся один раз: `uv run python scripts/hedera_poc.py create_topic --memo "thesis"`.

## Запуск

```bash
sudo systemctl start redis-server
source ~/.venv-plc-twin/bin/activate
bash scripts/start_all.sh           # Redis-check + 3 воркера + beat + Taipy
# http://<pi-ip>:5000/overview
bash scripts/stop_all.sh            # SIGTERM -> 10s -> SIGKILL
```

PID-файлы лежат в `data/pids/`, логи в `data/logs/`. Оба каталога создаются автоматически.

## Разработка

```bash
uv run pytest tests/ -v
uv run ruff check . && uv run ruff format .

# Hedera
uv run python scripts/hedera_poc.py balance
uv run python -m scripts.check_proofs --limit 20 --verify-payload

# Smoke полного стека
uv run celery -A src.tasks.celery_app inspect ping

# Миграция SQLite -> Postgres
DB_BACKEND=postgres POSTGRES_DSN=postgresql://... \
  uv run python -m scripts.migrate_sqlite_to_postgres

# Новые зависимости
uv add <pkg>                       # main
uv add --group ai <pkg>            # AI-группа
uv add --extra postgres <pkg>      # postgres extras
```

## Модели

В `models/` лежит демо-набор (~4 МБ), обученный на реальной телеметрии стенда: `xgb_temp_real_v2`, `lightgbm_temp_real_v2`, `xgb_humidity_real_v1` (+ `.meta.json` к каждой). Этого достаточно, чтобы прогноз работал из коробки: ставим `AI_ENABLED=true` в `.env` и всё.

Полный набор артефактов (xgboost / lightgbm / prophet / lstm / nbeats / linear / persistence в разных вариантах) собирается своими руками - см. ниже.

## Обучение моделей (на внешней машине)

```bash
# на Pi: выгрузить датасет
uv run python scripts/export_training_data.py --out data/exports/<ds>.csv

# на машине с GPU
python training/train_xgboost.py --input <ds>.csv --target temp_c_next_15min
# артефакт <name>.joblib + <name>.meta.json -> копировать в models/ на Pi

# на Pi: валидация артефакта
uv run python scripts/validate_model_artifact.py models/<file>.joblib
# в .env: AI_ENABLED=true, опционально MODEL_ID=<id>
```

Тренировщики: `train_xgboost.py`, `train_lightgbm.py`, `train_lstm.py`, `train_prophet.py`, `train_nbeats.py`, `train_linear.py`, `train_persistence.py`. Train/test split строго по времени (временные ряды), `random_state=42`. На Pi обучение не запускать - слабовато.

## Структура

```
app/             # Taipy GUI: web.py, 6 страниц, state, refresh, theme, data_access
src/
  sensors/       # bme280, bh1750, ina219, ds18b20, mq2, pir
  actuators/     # relay, servo, leds
  comms/         # rs485, oled
  ai/            # loader, predictor, features, recommender, registry
  storage/       # sqlite / postgres backends + factory
  integrations/  # ipfs (pinata, web3_storage, local_node), hedera (client, topic, digest)
  tasks/         # celery_app, hardware, ai, external
  patches/       # gpio_expander под Pi 5
scripts/         # start/stop_all, hedera_poc, check_proofs, migrate, validate, ...
training/        # train_*.py - запускать на внешней машине
models/          # демо-артефакты .joblib + .meta.json (остальные в .gitignore)
configs/         # collector / celery / taipy / runtime.yml (горячая перезагрузка)
migrations/      # postgres/001_init.sql
tests/           # pytest без железа
```

## Безопасность и ограничения

- `.env` в `.gitignore` - секреты локально.
- `ALLOW_HARDWARE_RUNTIME=false` и `ALLOW_EXTERNAL_API_CALLS=false` по умолчанию. Пока флаги не выставлены: hw-воркер не трогает GPIO, ext-воркер не ходит в сеть.
- BCM 7 и BCM 8 не использовать - hardware pull-up, реле залипает.
- `gpio=12,16,24,25=op,dl` в `/boot/firmware/config.txt` - реле LOW при загрузке.
- Команды актуаторов rate-limited (>=2 секунды между нажатиями) + подтверждение модалкой в UI.
- Kill switch - отдельная очередь `hardware_priority` с priority=9.
- Дашборд читает из SQLite, а не напрямую с датчиков. Если collector упал - UI показывает last-known и не падает.
- После любого `pip install -U troykahat` - заново прогнать `scripts/patch_troykahat_for_pi5.sh`.
