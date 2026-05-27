# Дизайн Krita Smart Select

## Статус

Черновик. Этот документ фиксирует первые продуктовые и инженерные решения по
плагину Krita Smart Select.

## Рабочее Название

Название плагина: **Krita Smart Select**.

В интерфейсе Krita при необходимости можно использовать более короткие подписи:

- Smart Select
- Smart Select Settings

## Цель

Krita Smart Select - плагин для Krita, который помогает создавать выделения с
помощью AI. Основной сценарий: пользователь выбирает объект или сегмент прямо на
canvas Krita, видит интерактивную подсветку кандидата и затем применяет выбранный
сегмент как обычное выделение Krita.

Ожидаемый опыт должен быть ближе к нативному инструменту выделения, чем к
отдельному окну с миниатюрой изображения.

## Основное UX-Решение

Главное взаимодействие должно происходить поверх настоящего canvas Krita:

1. Пользователь активирует Smart Select tool.
2. Плагин проверяет кэш сегментов для текущего изображения.
3. Если валидного кэша нет, плагин запускает подготовку сегментов для всего
   изображения.
4. Пока сегменты готовятся, Krita остаётся отзывчивой, а tool показывает
   progress/loader прямо в canvas/tool options.
5. Когда сегменты готовы, при движении курсора над canvas подсвечивается
   сегмент-кандидат под указателем.
6. Клик применяет подсвеченный сегмент как выделение Krita.
7. Клавиши-модификаторы могут поддерживать режимы добавления, вычитания,
   пересечения или переключения сегмента в выделении.

Первая версия работает как **prepared segmentation tool**: сначала строится набор
сегментов, затем пользователь выбирает уже готовые сегменты мышью. Prompt-based
selection по клику/box и выбор на основании активного выделения не входят в MVP.

Отдельная команда "Generate all masks for image/layer" в MVP не нужна:
генерация сегментов запускается автоматически при активации Smart Select, если
для текущего изображения нет валидного кэша. Позже можно добавить явную команду
"Refresh segments" / "Rebuild smart select cache" для ручного пересчёта.

Создание нового слоя из сегмента полезно, но не должно быть основным режимом.
Это может быть дополнительной командой, например "Extract selection to new
layer".

## Стратегия По Моделям

Первая версия должна сфокусироваться на **SAM 2 / SAM 2.1 PyTorch-моделях**.

Мы не таргетируем GGUF-модели в первой реализации. Существующий PyTorch-путь для
SAM2 уже подходит для использования в Krita, а фокус на нём делает продукт
проще.

Мы также не таргетируем пользовательские кастомные модели в первой версии.
Пользователю должны предлагаться только curated-модели, совместимость которых
заранее известна.

### Начальное Семейство Моделей

Начальный curated registry должен включать варианты SAM2 вроде:

- SAM2 Hiera Tiny
- SAM2 Hiera Small
- SAM2 Hiera Base Plus
- SAM2 Hiera Large

Точный список нужно проверить по официальному состоянию релизов SAM2/SAM2.1 на
момент реализации.

### Начальный Hardware Profile

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

### Решение По SAM 3

SAM 3 / SAM 3.1 не должен быть основным backend первой версии.

Причины:

- Доступные checkpoints SAM3 значительно больше checkpoints SAM2.
- Публичное распространение сейчас gated на HuggingFace.
- Требования к зависимостям и железу выше.
- SAM3 ориентирован на concept segmentation, видео и object tracking, а это не
  требуется для начального сценария выделения на canvas.

Архитектура может оставить место для будущего SAM3 backend, но ни одна фича
первой версии не должна от него зависеть.

## Model Registry

Метаданные моделей должны быть централизованы в `ModelRegistry`.

Registry должен управлять и поведением backend, и UI настроек. Каждая запись
модели должна содержать как минимум:

- Стабильный id модели
- Отображаемое имя
- Семейство модели
- URL checkpoint
- Имя/путь config
- Ожидаемый размер файла
- Опциональный checksum
- Рекомендованный объём RAM
- Рекомендованный объём VRAM
- Профиль качества/производительности
- Поддерживаемые устройства
- Заметки для отображения в настройках

Пример формы:

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

Окно настроек должно строиться на основе этого registry, а не через
захардкоженные UI-ветки.

## Загрузка И Кэш

Загрузка моделей должна быть устойчивой:

- Сначала скачивать во временный файл.
- Поддерживать resume там, где это практически возможно.
- Проверять размер файла и checksum, если он известен.
- Хранить модели в пользовательских данных/config Krita, а не внутри исходников
  плагина.
- Показывать понятные состояния загрузки, проверки и ошибки.
- Никогда не замораживать Krita во время загрузки.

## Packaging И Target Krita

MVP таргетирует конкретную версию Krita, а не широкий диапазон версий.

Это решение принято из-за C++/hybrid canvas tool слоя: нативный tool использует
внутренние API Krita, и совместимость между версиями нужно проверять явно.

Начальный packaging может следовать подходу Krita Vision Tools:

- C++ часть собирается как native library/module под выбранную версию Krita.
- Python extension может выступать loader-обёрткой, которая подгружает native
  library и регистрирует tool внутри Krita.
- Release artifact должен явно указывать поддерживаемую версию Krita и платформу.

Поддержка нескольких версий Krita и кросс-версионная binary compatibility не
являются целью первого MVP.

## Runtime-Архитектура

Inference не должен блокировать UI Krita.

Предпочтительная архитектура:

- UI плагина и canvas integration остаются отзывчивыми.
- Тяжёлая загрузка моделей и сегментация выполняются в worker-процессе.
- Worker предоставляет небольшой протокол для:
  - загрузки/выгрузки модели
  - установки изображения
  - генерации automatic masks
  - отправки progress
  - cancellation
  - ошибок

Worker-процесс предпочтительнее in-process inference, потому что сбои PyTorch,
CUDA и модельного кода не должны крашить или замораживать Krita.

Prompted masks можно добавить в будущем, но MVP не должен зависеть от prompt
selection. Основной inference job первой версии - automatic segmentation всего
текущего изображения при активации Smart Select.

### IPC Между C++ Tool И Python Worker

MVP должен использовать `QProcess` + stdio JSON-RPC/NDJSON как control channel.

Причины:

- C++/Qt стороне удобно запускать и контролировать worker через `QProcess`.
- Worker lifecycle прост: старт, stdin commands, stdout events/results,
  stderr logs, exit code/crash handling.
- Это проще и кроссплатформеннее для MVP, чем local socket, named pipe, gRPC или
  shared memory.
- Протокол легко логировать и воспроизводить при диагностике.

JSON/NDJSON используется только для команд, progress, cancellation, ошибок и
metadata. Большие данные не должны передаваться через JSON.

Большие массивы передаются через временные файлы:

- C++ сторона сохраняет flattened snapshot во временный image/raw/npy-файл и
  передаёт worker путь.
- Python worker пишет `segment_map` и masks во временные `.npy`/`.npz` файлы и
  возвращает пути вместе с metadata.
- C++ tool читает результат в память, после чего временные файлы можно удалить.
- Runtime cache сегментов внутри tool-а остаётся in-memory.

Пример формы сообщений:

```json
{"id":1,"method":"prepare_segments","params":{"image_path":"/tmp/input.png","model_id":"sam2_hiera_tiny","fingerprint":"..."}}
{"id":1,"event":"progress","stage":"loading_model","progress":null}
{"id":1,"event":"progress","stage":"generating_segments","progress":0.42}
{"id":1,"result":{"segment_map_path":"/tmp/segment_map.npy","segments_path":"/tmp/segments.npz","metadata":[]}}
```

Shared memory можно рассмотреть позже, если временные файлы окажутся узким
местом. Для MVP она не нужна.

### Входное Изображение

MVP сегментирует flattened projection текущего документа по размеру canvas.

Правила:

- Входом является изображение целиком, а не активный слой.
- Используется flattened projection всех видимых слоёв.
- Невидимые слои не учитываются.
- Bounds равны размеру canvas.
- Прозрачность считается фоном, а не отдельным объектом.
- Для RGB-входа модели прозрачные пиксели нужно композитить на стабильный фон и
  дополнительно хранить alpha/background mask, чтобы не предлагать полностью
  прозрачные области как сегменты.

Режимы "sample current layer" и сегментация только активного выделения появятся
позже, после стабилизации основного workflow.

### Подготовка Сегментов И Loader

При активации Smart Select tool должен:

1. Получить snapshot всего текущего изображения.
2. Проверить, есть ли валидный кэш сегментов для этого snapshot.
3. Если кэш валиден, сразу включить hover/click interaction.
4. Если кэша нет, отправить snapshot в worker-процесс.
5. Получить от worker segment map, masks и метаданные сегментов.
6. Построить структуру для быстрого hit-test под курсором.
7. Перейти в интерактивный режим выбора сегментов.

Основная runtime-модель интерактивного выбора:

- C++ tool хранит prepared segment map и metadata по сегментам.
- Overlay не является отдельным floating window; это состояние отрисовки
  нативного Krita tool-а.
- По умолчанию overlay ничего не рисует.
- При движении курсора tool переводит позицию в image coordinates.
- По координате курсора tool читает `segment_id` из segment map.
- Если `segment_id` изменился, tool обновляет `hoverSegmentId` и запрашивает
  перерисовку старой и новой области сегмента.
- В `paint()` tool рисует полупрозрачную красную mask только для текущего
  hover-сегмента.
- Клик применяет текущий hover-сегмент как выделение Krita с учётом selection
  modifier mode.

Для MVP предпочтительно иметь label/segment map формата `pixel -> segment_id`,
чтобы hover hit-test был O(1). Перебор всех masks на каждое движение мыши
допустим только как временный прототип для маленьких изображений.

Segment map, masks и metadata в MVP хранятся в памяти. Дисковый кэш можно
добавить позже, если подготовка сегментов окажется достаточно дорогой для
частого пересчёта между сессиями.

### Selection Modifiers

Поведение должно соответствовать обычным инструментам выделения Krita:

- Click: replace текущего выделения.
- Shift + click: add.
- Alt/Ctrl + click: subtract/intersect/toggle в соответствии с нативными
  conventions Krita и доступными `SelectionAction`.

Точное сопоставление Alt/Ctrl нужно брать из Krita selection tool behavior, а не
изобретать отдельно.

Пока worker занят, UI не должен выглядеть зависшим:

- На canvas нужно показывать небольшой overlay или inline-состояние tool-а:
  "Preparing smart segments..." с progress, если он известен.
- В tool options стоит показывать текущий этап: loading model, reading image,
  generating segments, building hover index.
- Progress должен появляться быстро, даже если сначала он indeterminate.
- Должна быть кнопка Cancel.
- Если пользователь сменил tool, документ или закрыл изображение, job должен
  отменяться или уходить в безопасный background state без применения результата
  к неправильному документу.
- Ошибка подготовки должна показывать короткое понятное сообщение и действия из
  support flow: Diagnostic Report, full log, logs folder, GitHub issue.

### Кэш Сегментов

Кэш сегментов относится к изображению целиком. MVP не пытается отдельно
кэшировать активный слой или режимы "sample current layer".

Ключ кэша должен включать:

- Стабильный id документа, если его можно получить или сохранить.
- Размер изображения.
- Цветовую модель/depth, если доступно.
- Registry id выбранной модели.
- Версию алгоритма подготовки сегментов.
- Дешёвый fingerprint изображения.

Fingerprint изображения должен быть консервативным. Рекомендуемый MVP-подход:
сделать downscale snapshot всего изображения до небольшого размера, например
64x64 или 128x128, привести к RGB/RGBA U8 и посчитать hash байтов. Для более
мягкой эвристики можно дополнительно считать mean absolute difference между
предыдущим и новым downscaled snapshot.

Правило reuse должно быть осторожным:

- Если размеры изображения или model id изменились, кэш невалиден.
- Если fingerprint совпал, кэш можно использовать.
- Если fingerprint не совпал, кэш невалиден.
- Если есть сомнение, кэш нужно инвалидировать и пересчитать сегменты.

Follow-up после MVP: можно добавить более умные perceptual hashes, mean absolute
difference или Jaccard/IoU по карте изменённых пикселей, чтобы разрешать reuse
при небольших изменениях. В MVP similarity thresholds не используются.

## Observability И Диагностика

Плагин должен быть максимально прозрачен при ошибках и долгих операциях. Цель -
сделать поддержку возможной без угадывания по скриншотам и без просьб к
пользователю вручную искать файлы логов.

Логи должны покрывать:

- Версию плагина
- Версию Krita
- Версию Python в Krita
- Путь к venv/worker environment
- ОС и архитектуру
- Доступные устройства inference: CPU, CUDA и другие backend, если появятся
- Выбранную модель и её registry id
- Состояние model cache
- Этапы загрузки и проверки модели
- Старт, progress, cancellation и результат inference jobs
- Ошибки worker-процесса
- Ошибки Krita API при создании selection/layer

При любой пользовательской ошибке или внутреннем exception UI должен давать
понятные действия:

- Скопировать краткий отчёт об ошибке
- Скопировать полный лог
- Открыть папку с логами
- Открыть GitHub issue

Кнопка "Открыть GitHub issue" должна показывать pop-up с двумя вариантами:

- Manual
- Auto (requires GitHub CLI)

REST API GitHub для создания issue не рассматривается. Плагин не должен сам
получать, хранить или обновлять GitHub token.

Manual flow:

- Плагин открывает браузер с prefilled issue template.
- В шаблон можно передавать краткий отчёт через query params.
- Полный лог лучше предлагать скопировать отдельно, чтобы не упереться в лимиты
  URL и случайно не отправить лишние данные.
- Пользователь сам отправляет issue в браузере.

Auto flow через GitHub CLI:

- Плагин проверяет наличие `gh` в системе.
- Плагин проверяет авторизацию через `gh auth status`.
- Если `gh` не найден, плагин показывает ошибку: нужно установить GitHub CLI.
- Если `gh` найден, но не авторизован, плагин показывает ошибку: нужно выполнить
  `gh auth login` и повторить действие.
- Если `gh` найден и авторизован, плагин показывает preview issue body.
- После явного подтверждения пользователя плагин создаёт issue через
  `gh issue create --repo OWNER/REPO --title ... --body-file ...`.

Auto flow должен оставаться optional. Он удобен для пользователей, у которых уже
установлен и авторизован GitHub CLI, но не должен быть обязательным механизмом
support flow.

Diagnostic Report нужен не вместо полноценных логов, а как безопасный и быстрый
первый слой диагностики. Он должен быть достаточно коротким, чтобы его можно было
автоматически вставить в issue body или скопировать без риска утечки лишних
данных.

Краткий Diagnostic Report должен включать:

- Версию плагина
- Версию Krita
- ОС и архитектуру
- Версию Python в Krita
- Тип установки Krita, если его можно определить: AppImage, system package,
  Flatpak, Windows/macOS bundle
- Выбранную модель и её registry id
- Device/backend: CPU, CUDA, CUDA version, GPU name, если доступно
- Состояние модели: not downloaded, downloading, verified, load failed, loaded
- Последнее действие пользователя
- Последний failed event id
- Последний exception type и короткий message
- Путь к локальному log file, с редактированием домашней директории
- Несколько последних структурированных событий без stack trace

Полный лог остаётся отдельным артефактом и нужен для сложных случаев:

- Полный stack trace
- Подробный worker lifecycle
- Подробный download/inference progress
- stderr/stdout worker-процесса
- Детали model loading
- Ошибки Krita API

Логи не должны включать содержимое изображений, pixel data, masks или полные
пути к пользовательским файлам без явной необходимости. Перед копированием
отчёта в issue нужно по возможности редактировать чувствительные данные:

- Домашнюю директорию пользователя
- Абсолютные пути к документам
- Токены и переменные окружения с секретами

Для диагностики стоит использовать структурированные записи событий, чтобы
позже их можно было легко фильтровать:

- `setup.start`
- `setup.failed`
- `model.download.start`
- `model.download.failed`
- `worker.start`
- `worker.crashed`
- `inference.start`
- `inference.cancelled`
- `inference.failed`
- `selection.apply.failed`

В UI настроек или support dialog должен быть раздел диагностики с кнопками:

- Copy Diagnostic Report
- Copy Full Log
- Open Logs Folder
- Open GitHub Issue

## Интеграция С Canvas

Желаемый UX вдохновлён Krita Vision Tools, где сегментация реализована как
нативные инструменты Krita.

Для MVP canvas/tool слой должен быть C++/hybrid, а не Python-only overlay.

C++ часть отвечает за:

- Нативный Krita tool lifecycle.
- Получение canvas/image coordinates.
- Cursor и selection modifier behavior.
- Хранение текущего `hoverSegmentId`.
- Hit-test по prepared segment map под курсором.
- Отрисовку hover-подсветки готового сегмента в `paint()`.
- Применение mask как обычного выделения Krita.
- Инвалидацию/отмену текущего interaction state при смене документа, tool или
  изображения.

Python часть может отвечать за worker orchestration, model registry, download,
кэш и PyTorch/SAM2 inference. Граница между C++ tool и worker должна быть
небольшой: передать snapshot изображения на подготовку, получить список masks и
метаданные сегментов, затем применять выбранные masks через нативный Krita API.

Krita Vision Tools полезен как reference для:

- C++ plugin/module сборки под Krita.
- Регистрации нативных tools.
- Получения canvas/image coordinates.
- Применения pixel selection через Krita API.
- Отрисовки tool overlay через `paint()` и `updateCanvas()`.
- Python loader-обёртки для native library.

Krita Vision Tools не отвечает на несколько решений Krita Smart Select:

- IPC между C++ tool и отдельным Python/PyTorch worker.
- Prepared automatic segmentation всего изображения.
- Формат `segment_map` / `pixel -> segment_id`.
- Hover-подсветку сегмента в segmentation tool.
- Кэш segment maps/masks.
- SAM2 PyTorch dependency management.

## Исходные Проекты И Attribution

Проект может использовать идеи реализации из:

- Smart Segments
- Krita Vision Tools

Smart Segments полезен как reference для SAM2/PyTorch pipeline и связки Krita
pixels/selections.

Krita Vision Tools полезен как reference для нативного canvas/tool UX и
поведения selection tool.

В проекте должны быть корректные license notices и attribution для любого кода
или существенных implementation patterns, которые будут переиспользованы.

## Направление По Лицензии

Проект ожидается GPLv3-compatible, что соответствует практическим лицензионным
ожиданиям для разработки плагинов Krita и позволяет при необходимости быть
совместимыми с кодом Krita Vision Tools.

Точные license files и third-party notices нужно финализировать до первого
публичного релиза.

## Типизация И Тесты

Статическая типизация желательна, но вводить её нужно аккуратно.

Рекомендованный подход:

- Использовать строгую типизацию для чистых core-модулей.
- Описать типизированные границы вокруг Krita/PyQt API через protocols или
  локальные stubs.
- Динамический UI и Krita adapter code типизировать прагматично.
- Не заставлять mypy бороться с внутренностями PyQt/Krita до того, как
  архитектура станет чистой.

Тесты должны фокусироваться на поведении, которое можно проверять вне Krita:

- Валидация model registry
- Логика download/cache
- Hit-testing masks
- Маппинг координат
- Комбинирование masks
- Worker protocol
- Error handling и cancellation
- Image fingerprint/cache invalidation

Krita integration должна иметь отдельные smoke/integration tests там, где это
практически возможно.

## Не-Цели Первой Версии

- Загрузка пользовательских кастомных моделей
- Поддержка GGUF-моделей
- SAM3 как default или обязательный backend
- Prompt-based selection по point/box
- Выбор/генерация сегментов только на основании активного выделения
- Отдельный режим "sample current layer"; MVP работает с текущим изображением
  целиком
- Сложное generative editing или inpainting

## Открытые Вопросы

- Нет блокирующих открытых вопросов для перехода к планированию MVP.
