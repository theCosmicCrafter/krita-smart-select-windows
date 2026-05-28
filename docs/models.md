# Models

## Model Strategy

Первая версия должна сфокусироваться на **SAM2 / SAM2.1 PyTorch-моделях**.

MVP не таргетирует:

- GGUF-модели;
- пользовательские кастомные модели;
- SAM3/SAM3.1 как default или обязательный backend.

SAM3 может быть будущим backend, но ни одна фича первой версии не должна от
него зависеть.

## Initial Curated Registry Candidates

Начальный curated registry должен включать варианты SAM2 вроде:

- SAM2 Hiera Tiny
- SAM2 Hiera Small
- SAM2 Hiera Base Plus
- SAM2 Hiera Large

Точный список нужно проверить по официальному состоянию релизов SAM2/SAM2.1 на
момент реализации.

## Hardware Profile

Hardware profile в `ModelRegistry` должен быть консервативной подсказкой для
пользователя и support flow, а не строгим запретом запуска.

Ориентировочные значения для MVP:

- SAM2/SAM2.1 Hiera Tiny: RAM 8 GB, VRAM 0-4 GB, быстрый/default вариант для
  CPU или слабого GPU.
- SAM2/SAM2.1 Hiera Small: RAM 8-16 GB, VRAM 4 GB, хороший default для GPU.
- SAM2/SAM2.1 Hiera Base Plus: RAM 16 GB, VRAM 6-8 GB, quality/performance
  вариант.
- SAM2/SAM2.1 Hiera Large: RAM 16-32 GB, VRAM 8-12 GB, quality/slow вариант.

Automatic mask generation для Base Plus и Large нужно помечать как тяжёлую
операцию. UI может предупреждать пользователя, если изображение большое или
выбранная модель явно не соответствует доступной памяти.

## ModelRegistry Contract

Метаданные моделей должны быть централизованы в `ModelRegistry`.

Каждая запись модели должна содержать как минимум:

- стабильный id модели;
- отображаемое имя;
- семейство модели;
- URL checkpoint;
- имя/путь config;
- ожидаемый размер файла;
- опциональный checksum;
- рекомендованный объём RAM;
- рекомендованный объём VRAM;
- профиль качества/производительности;
- поддерживаемые устройства;
- заметки для отображения в настройках.

Example shape:

```python
ModelSpec(
    id="sam2_hiera_tiny",
    title="SAM2 Hiera Tiny",
    family="sam2",
    checkpoint_url="https://...",
    config_name="configs/sam2/sam2_hiera_t.yaml",
    size_mb=155,
    recommended_ram_gb=8,
    recommended_vram_gb=0,
    profile="fast",
)
```

Окно настроек должно строиться на основе registry, а не через захардкоженные
UI-ветки.

## Download And Cache

Загрузка моделей должна быть устойчивой:

- Сначала скачивать во временный файл.
- Поддерживать resume там, где это практически возможно.
- Проверять размер файла и checksum, если он известен.
- Хранить модели в пользовательских данных/config Krita, а не внутри исходников
  плагина.
- Показывать понятные состояния загрузки, проверки и ошибки.
- Никогда не замораживать Krita во время загрузки.

## Current Implementation Status

Real `ModelRegistry` ещё не реализован. Текущий worker-backed mock flow
использует `model_id="mock"`.
