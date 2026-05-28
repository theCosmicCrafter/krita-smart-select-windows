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

Текущий readiness-stage registry активирует только тяжёлую default-модель:

- SAM2.1 Hiera Large (`sam2.1_hiera_large`)

Лёгкие модели остаются follow-up для будущего model picker:

- SAM2 Hiera Tiny
- SAM2 Hiera Small
- SAM2 Hiera Base Plus

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

Для readiness-stage default выбран **SAM2.1 Hiera Large**, потому что первый
real inference этап должен сразу проверять максимальное качество, а не быстрый
минимальный путь.

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
    id="sam2.1_hiera_large",
    title="SAM2.1 Hiera Large",
    family="sam2.1",
    checkpoint_url="https://huggingface.co/facebook/sam2.1-hiera-large/resolve/38e0b24f84dfe8a95d5b6cb53bc1b772cbe15dc2/sam2.1_hiera_large.pt?download=true",
    checkpoint_filename="sam2.1_hiera_large.pt",
    config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
    expected_bytes=898089053,
    etag='"5a2b7e292d0cbd3cce5f410502ced3eb9b92031a7b21fcea62cb97eed73de1d3"',
    recommended_ram_gb=32,
    recommended_vram_gb=12,
    profile="quality",
    requires_download=True,
)
```

Окно настроек должно строиться на основе registry, а не через захардкоженные
UI-ветки.

## Download And Cache

Загрузка моделей должна быть устойчивой:

- Сначала скачивать во временный файл.
- Поддерживать resume там, где это практически возможно.
- Ретраить transient network failures без потери `.partial`.
- Проверять свободное место перед продолжением загрузки.
- Проверять размер файла и checksum, если он известен.
- Хранить модели в пользовательских данных/config Krita, а не внутри исходников
  плагина.
- Показывать понятные состояния загрузки, проверки и ошибки.
- Никогда не замораживать Krita во время загрузки.

Для SAM2.1 Hiera Large используется pinned Hugging Face checkpoint:

- URL: `https://huggingface.co/facebook/sam2.1-hiera-large/resolve/38e0b24f84dfe8a95d5b6cb53bc1b772cbe15dc2/sam2.1_hiera_large.pt?download=true`
- expected size: `898089053` bytes
- ETag: `"5a2b7e292d0cbd3cce5f410502ced3eb9b92031a7b21fcea62cb97eed73de1d3"`

Официальный checksum в registry пока не используется, поэтому verification
использует exact byte size. ETag хранится как metadata и как future input для
более строгой проверки cache identity.

## Current Implementation Status

Реализованы:

- typed `ModelRegistry`;
- default model id `sam2.1_hiera_large`;
- download/cache/verify checkpoint-а в worker process;
- runtime diagnostics foundation.

Real SAM2 inference ещё не реализован. Worker пока возвращает mock segment map
после model readiness.
