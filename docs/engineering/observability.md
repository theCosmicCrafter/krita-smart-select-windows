# Observability

## Goals

Плагин должен быть максимально прозрачен при ошибках и долгих операциях.

Цель observability:

- сделать поддержку возможной без угадывания по скриншотам;
- не просить пользователя вручную искать системные логи;
- иметь короткий безопасный Diagnostic Report;
- иметь полный лог для сложных случаев;
- не утекать содержимым изображений, pixel data, masks или секретами.

## Required Log Coverage

Логи должны покрывать:

- версию плагина;
- версию Krita;
- версию Python в Krita;
- путь к venv/worker environment;
- ОС и архитектуру;
- доступные устройства inference: CPU, CUDA и другие backend, если появятся;
- выбранную модель и её registry id;
- состояние model cache;
- этапы загрузки и проверки модели;
- start/progress/cancellation/result inference jobs;
- ошибки worker-процесса;
- stderr/stdout worker-процесса;
- ошибки Krita API при создании selection/layer.

## Structured Events

Нужно использовать структурированные записи событий, чтобы их можно было
фильтровать:

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

Stable event names важнее красивого текста. User-facing message может быть
отдельным полем.

## Diagnostic Report

Diagnostic Report нужен не вместо полноценных логов, а как безопасный и быстрый
первый слой диагностики. Он должен быть достаточно коротким, чтобы его можно
было автоматически вставить в issue body или скопировать без риска утечки
лишних данных.

Краткий Diagnostic Report должен включать:

- версию плагина;
- версию Krita;
- ОС и архитектуру;
- версию Python в Krita;
- тип установки Krita, если его можно определить: AppImage, system package,
  Flatpak, Windows/macOS bundle;
- выбранную модель и её registry id;
- device/backend: CPU, CUDA, CUDA version, GPU name, если доступно;
- состояние модели: not downloaded, downloading, verified, load failed, loaded;
- последнее действие пользователя;
- последний failed event id;
- последний exception type и короткий message;
- путь к локальному log file, с редактированием домашней директории;
- несколько последних структурированных событий без stack trace.

## Full Log

Полный лог остаётся отдельным артефактом и нужен для сложных случаев:

- полный stack trace;
- подробный worker lifecycle;
- подробный download/inference progress;
- stderr/stdout worker-процесса;
- детали model loading;
- ошибки Krita API.

## Redaction Rules

Логи не должны включать:

- содержимое изображений;
- pixel data;
- masks;
- tokens;
- секретные переменные окружения.

Перед копированием отчёта в issue нужно по возможности редактировать:

- домашнюю директорию пользователя;
- абсолютные пути к документам;
- значения переменных окружения с секретами.

## Support UI Actions

В UI настроек или support dialog должен быть раздел диагностики с кнопками:

- Copy Diagnostic Report
- Copy Full Log
- Open Logs Folder
- Open GitHub Issue

## GitHub Issue Flow

Кнопка `Open GitHub Issue` должна показывать pop-up с двумя вариантами:

- Manual
- Auto (requires GitHub CLI)

REST API GitHub для создания issue не рассматривается. Плагин не должен сам
получать, хранить или обновлять GitHub token.

### Manual Flow

- Плагин открывает браузер с prefilled issue template.
- В шаблон можно передавать краткий отчёт через query params.
- Полный лог лучше предлагать скопировать отдельно, чтобы не упереться в лимиты
  URL и случайно не отправить лишние данные.
- Пользователь сам отправляет issue в браузере.

### Auto Flow Через GitHub CLI

- Плагин проверяет наличие `gh` в системе.
- Плагин проверяет авторизацию через `gh auth status`.
- Если `gh` не найден, плагин показывает ошибку: нужно установить GitHub CLI.
- Если `gh` найден, но не авторизован, плагин показывает ошибку: нужно
  выполнить `gh auth login` и повторить действие.
- Если `gh` найден и авторизован, плагин показывает preview issue body.
- После явного подтверждения пользователя плагин создаёт issue через
  `gh issue create --repo OWNER/REPO --title ... --body-file ...`.

Auto flow должен оставаться optional.

## Current Implementation Status

Текущая реализация уже пишет structured events через `qInfo`, `qWarning` и
Python `print`.

Ещё не реализовано:

- persistent plugin log file;
- Diagnostic Report UI;
- Copy Full Log;
- Open Logs Folder;
- Open GitHub Issue.
