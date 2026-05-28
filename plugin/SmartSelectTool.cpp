#include "SmartSelectTool.h"

#include "SelectionApply.h"
#include "SmartSelectRuntimeConfig.h"
#include "WorkerClient.h"

#include <QCursor>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QIcon>
#include <QJsonArray>
#include <QLabel>
#include <QPainter>
#include <QProgressBar>
#include <QPushButton>

#include <KisViewManager.h>
#include <KisOptionCollectionWidget.h>
#include <KoPointerEvent.h>
#include <canvas/kis_canvas2.h>
#include <kis_canvas_resource_provider.h>
#include <kis_floating_message.h>
#include <kis_image.h>
#include <kis_selection_tool_config_widget_helper.h>
#include <klocalizedstring.h>

SmartSelectTool::SmartSelectTool(KoCanvasBase *canvas)
    : KisToolSelect(canvas, QCursor(Qt::CrossCursor), i18n("Smart Select"))
{
    setObjectName("tool_smart_select");
}

SmartSelectTool::~SmartSelectTool()
{
    qInfo() << "[SmartSelect] event=tool.destroy.enter"
            << "state" << int(m_state)
            << "activeJobId" << m_activeJobId;

    if (m_worker) {
        disconnect(m_worker, nullptr, this, nullptr);
        m_worker->stop();
    }

    qInfo() << "[SmartSelect] event=tool.destroy.exit";
}

void SmartSelectTool::activate(const QSet<KoShape *> &shapes)
{
    qInfo() << "[SmartSelect] event=tool.activate.enter"
            << "tool=SmartSelectTool"
            << "state" << int(m_state)
            << "activeJobId" << m_activeJobId
            << "hasStatusWidget" << !m_statusLabel.isNull();
    KisToolSelect::activate(shapes);
    qInfo() << "[SmartSelect] event=tool.activate.after_base";

    if (KisImage *image = currentImage().data()) {
        connect(image, SIGNAL(sigImageUpdated(QRect)), this, SLOT(updateImage(QRect)));
        qInfo() << "[SmartSelect] event=tool.activate.image_connected"
                << "bounds" << image->bounds();
    } else {
        qWarning() << "[SmartSelect] event=tool.activate.no_image";
    }

    qInfo() << "[SmartSelect] event=tool.activate.prepare_begin";
    prepareSegments();
    qInfo() << "[SmartSelect] event=tool.activate.exit"
            << "state" << int(m_state)
            << "activeJobId" << m_activeJobId;
}

void SmartSelectTool::deactivate()
{
    qInfo() << "[SmartSelect] event=tool.deactivate.enter"
            << "state" << int(m_state)
            << "activeJobId" << m_activeJobId
            << "hoverSegmentId" << m_hoverSegmentId;

    if (KisImage *image = currentImage().data()) {
        disconnect(image, SIGNAL(sigImageUpdated(QRect)), this, SLOT(updateImage(QRect)));
        qInfo() << "[SmartSelect] event=tool.deactivate.image_disconnected";
    }

    cancelActiveJob();

    const quint16 oldHover = m_hoverSegmentId;
    m_hoverSegmentId = 0;
    requestSegmentUpdate(oldHover);
    clearPreparedData();
    m_state = State::Idle;
    m_stageText.clear();
    m_progress = 0.0;
    m_hasProgress = false;
    m_errorMessage.clear();

    KisToolSelect::deactivate();
    qInfo() << "[SmartSelect] event=tool.deactivate.exit";
}

void SmartSelectTool::updateImage(QRect const &)
{
    qInfo() << "[SmartSelect] event=tool.image.updated";
    prepareSegments();
}

void SmartSelectTool::prepareSegments()
{
    cancelActiveJob();
    m_hoverSegmentId = 0;
    clearPreparedData();

    if (!currentImage()) {
        setState(State::Error, i18n("no active image"));
        return;
    }

    setState(State::Preparing);
    if (!startWorkerIfNeeded()) {
        return;
    }

    startMockPrepareJob();
}

bool SmartSelectTool::startWorkerIfNeeded()
{
    if (!m_worker) {
        m_worker = new WorkerClient(this);
        connect(m_worker, &WorkerClient::progress, this, &SmartSelectTool::onWorkerProgress);
        connect(m_worker, &WorkerClient::result, this, &SmartSelectTool::onWorkerResult);
        connect(m_worker, &WorkerClient::error, this, &SmartSelectTool::onWorkerError);
        connect(m_worker, &WorkerClient::stderrLine, this, &SmartSelectTool::onWorkerStderr);
        connect(m_worker, &WorkerClient::crashed, this, &SmartSelectTool::onWorkerCrashed);
    }

    const SmartSelectRuntimeConfig &config = SmartSelectRuntimeConfig::instance();
    const QString pythonExecutable = config.pythonExecutable();
    const QString workerScript = config.workerScriptPath();

    if (pythonExecutable.isEmpty() || !QFileInfo::exists(pythonExecutable)) {
        qWarning() << "[SmartSelect] event=worker.start.failed"
                   << "reason=python_executable_not_found"
                   << "pythonExecutable" << pythonExecutable;
        setState(State::Error, i18n("python executable not found"));
        return false;
    }

    if (workerScript.isEmpty() || !QFileInfo::exists(workerScript)) {
        qWarning() << "[SmartSelect] event=worker.start.failed"
                   << "reason=worker_script_not_found"
                   << "workerScript" << workerScript;
        setState(State::Error, i18n("worker script not found"));
        return false;
    }

    qInfo() << "[SmartSelect] event=worker.start"
            << "pythonExecutable" << pythonExecutable
            << "workerScript" << workerScript;
    if (!m_worker->start(pythonExecutable, workerScript)) {
        setState(State::Error, i18n("worker failed to start"));
        return false;
    }

    return true;
}

void SmartSelectTool::startMockPrepareJob()
{
    if (!currentImage() || !m_worker) {
        return;
    }

    const QRect bounds = currentImage()->bounds();
    m_activeJobId = m_worker->prepareSegments(QString(),
                                              QStringLiteral("mock"),
                                              QStringLiteral("mock-no-snapshot-v1"),
                                              bounds.size());
    qInfo() << "[SmartSelect] event=inference.start"
            << "jobId" << m_activeJobId
            << "modelId=mock"
            << "fingerprint=mock-no-snapshot-v1"
            << "canvasSize" << bounds.size();
}

void SmartSelectTool::cancelActiveJob()
{
    if (m_worker && m_activeJobId != 0) {
        qInfo() << "[SmartSelect] event=inference.cancelled"
                << "jobId" << m_activeJobId;
        m_worker->cancel(m_activeJobId);
        m_activeJobId = 0;
    }
}

void SmartSelectTool::setState(State state, const QString &message)
{
    qInfo() << "[SmartSelect] event=tool.state.change"
            << "from" << int(m_state)
            << "to" << int(state)
            << "message" << message
            << "hasStatusWidget" << !m_statusLabel.isNull();
    m_state = state;
    if (state == State::Preparing) {
        m_stageText.clear();
        m_progress = 0.0;
        m_hasProgress = false;
        m_errorMessage.clear();
    } else if (state == State::Error) {
        m_errorMessage = message;
    } else if (state == State::Ready) {
        m_errorMessage.clear();
    }

    updateOptionsUi();
}

void SmartSelectTool::updateOptionsUi()
{
    if (m_statusLabel) {
        QString text;
        switch (m_state) {
        case State::Idle:
            text = i18n("Smart Select");
            break;
        case State::Preparing:
            text = m_stageText.isEmpty()
                ? i18n("Preparing smart segments...")
                : i18n("Preparing smart segments: %1", m_stageText);
            break;
        case State::Ready:
            text = i18n("Smart Select: %1 segments", m_segmentMap.segmentCount());
            break;
        case State::Error:
            text = i18n("Smart Select error: %1", m_errorMessage);
            break;
        case State::Cancelled:
            text = i18n("Smart Select cancelled");
            break;
        }
        m_statusLabel->setText(text);
    }

    if (m_progressBar) {
        const bool preparing = m_state == State::Preparing;
        m_progressBar->setVisible(preparing);
        if (preparing) {
            if (m_hasProgress) {
                m_progressBar->setRange(0, 100);
                m_progressBar->setValue(qBound(0, int(m_progress * 100.0), 100));
            } else {
                m_progressBar->setRange(0, 0);
            }
        }
    }

    if (m_cancelButton) {
        m_cancelButton->setVisible(m_state == State::Preparing);
    }
}

void SmartSelectTool::clearPreparedData()
{
    m_segmentMap.clear();
}

void SmartSelectTool::cleanupTempFiles(const QStringList &paths)
{
    for (const QString &path : paths) {
        if (path.isEmpty()) {
            continue;
        }

        QFileInfo info(path);
        const QString parentPath = info.absolutePath();
        if (QFile::remove(path)) {
            qInfo() << "[SmartSelect] event=temp_file.cleaned" << "path" << path;
        } else if (QFileInfo::exists(path)) {
            qWarning() << "[SmartSelect] event=temp_file.cleanup_failed" << "path" << path;
        }

        QDir parent(parentPath);
        if (parent.dirName().startsWith(QStringLiteral("krita-smart-select-")) && parent.isEmpty()) {
            if (parent.rmdir(parentPath)) {
                qInfo() << "[SmartSelect] event=temp_dir.cleaned" << "path" << parentPath;
            }
        }
    }
}

void SmartSelectTool::onWorkerProgress(int id, const QString &stage, double progress, bool hasProgress)
{
    if (id != m_activeJobId) {
        return;
    }

    qInfo() << "[SmartSelect] event=inference.progress"
            << "jobId" << id
            << "stage" << stage
            << "progress" << progress
            << "hasProgress" << hasProgress;
    m_stageText = stage;
    m_progress = progress;
    m_hasProgress = hasProgress;
    updateOptionsUi();
}

void SmartSelectTool::onWorkerResult(int id, const QString &segmentMapPath, const QString &segmentsPath, const QJsonArray &metadata)
{
    QStringList cleanupPaths;
    cleanupPaths << segmentMapPath << segmentsPath;

    if (id != m_activeJobId) {
        qInfo() << "[SmartSelect] event=inference.result.ignored_stale"
                << "jobId" << id
                << "activeJobId" << m_activeJobId;
        cleanupTempFiles(cleanupPaths);
        return;
    }

    qInfo() << "[SmartSelect] event=inference.result"
            << "jobId" << id
            << "segmentMapPath" << segmentMapPath
            << "segmentsPath" << segmentsPath
            << "metadataCount" << metadata.size();

    QString error;
    SegmentMap loadedMap;
    if (!loadedMap.loadNpy(segmentMapPath, &error)) {
        cleanupTempFiles(cleanupPaths);
        m_activeJobId = 0;
        setState(State::Error, error);
        return;
    }

    m_segmentMap = loadedMap;
    m_activeJobId = 0;
    setState(m_segmentMap.isEmpty() ? State::Error : State::Ready,
             m_segmentMap.isEmpty() ? i18n("segment map is empty") : QString());

    if (currentImage()) {
        canvas()->updateCanvas(currentImage()->pixelToDocument(currentImage()->bounds()));
    }

    cleanupTempFiles(cleanupPaths);
}

void SmartSelectTool::onWorkerError(int id, const QString &type, const QString &message)
{
    if (id != 0 && id != m_activeJobId) {
        qInfo() << "[SmartSelect] event=inference.error.ignored_stale"
                << "jobId" << id
                << "type" << type
                << "message" << message;
        return;
    }

    qWarning() << "[SmartSelect] event=inference.failed"
               << "jobId" << id
               << "type" << type
               << "message" << message;
    m_activeJobId = 0;
    setState(State::Error, message);
}

void SmartSelectTool::onWorkerStderr(const QString &line)
{
    qWarning() << "[SmartSelect] event=worker.stderr" << "line" << line;
}

void SmartSelectTool::onWorkerCrashed(const QString &message)
{
    qWarning() << "[SmartSelect] event=worker.crashed" << "message" << message;
    if (m_state == State::Preparing || m_activeJobId != 0) {
        m_activeJobId = 0;
        setState(State::Error, message);
    }
}

void SmartSelectTool::mouseMoveEvent(KoPointerEvent *event)
{
    if (m_state != State::Ready || m_segmentMap.isEmpty()) {
        KisToolSelect::mouseMoveEvent(event);
        return;
    }

    const QPoint position = convertToImagePixelCoordFloored(event);
    setHoverSegment(m_segmentMap.segmentAt(position));
    KisToolSelect::mouseMoveEvent(event);
}

void SmartSelectTool::beginPrimaryAction(KoPointerEvent *event)
{
    KisToolSelectBase::beginPrimaryAction(event);
    if (isMovingSelection()) {
        return;
    }
    if (!selectionEditable()) {
        showNotEditableMessage();
        event->ignore();
        return;
    }

    if (m_hoverSegmentId == 0 || m_state != State::Ready) {
        event->ignore();
        return;
    }

    beginSelectInteraction();
    applySegmentSelection(canvas(),
                          currentImage(),
                          currentNode(),
                          m_segmentMap,
                          m_hoverSegmentId,
                          selectionAction(),
                          growSelection(),
                          featherSelection(),
                          antiAliasSelection());
}

void SmartSelectTool::endPrimaryAction(KoPointerEvent *event)
{
    if (isMovingSelection()) {
        KisToolSelectBase::endPrimaryAction(event);
        return;
    }

    Q_UNUSED(event);
    endSelectInteraction();
}

void SmartSelectTool::paint(QPainter &painter, const KoViewConverter &converter)
{
    Q_UNUSED(converter);
    if (m_hoverSegmentId == 0 || m_state != State::Ready) {
        return;
    }

    const QRect bbox = m_segmentMap.segmentBounds(m_hoverSegmentId);
    if (bbox.isEmpty()) {
        return;
    }

    const QImage overlay = m_segmentMap.overlayImage(m_hoverSegmentId, QColor(255, 0, 0, 96));
    if (overlay.isNull()) {
        return;
    }

    painter.save();
    painter.drawImage(pixelToView(bbox), overlay);
    painter.restore();
}

QWidget *SmartSelectTool::createOptionWidget()
{
    qInfo() << "[SmartSelect] event=tool.options.create.enter";
    KisToolSelectBase::createOptionWidget();
    KisSelectionOptions *selectionWidget = selectionOptionWidget();
    selectionWidget->setReferenceSectionVisible(false);

    KisOptionCollectionWidgetWithHeader *sectionStatus =
        new KisOptionCollectionWidgetWithHeader(i18n("Smart Select"));

    m_statusLabel = new QLabel;
    m_progressBar = new QProgressBar;
    m_cancelButton = new QPushButton(i18n("Cancel"));
    connect(m_cancelButton, &QPushButton::clicked, this, &SmartSelectTool::cancelActiveJob);

    sectionStatus->appendWidget(QStringLiteral("statusLabel"), m_statusLabel);
    sectionStatus->appendWidget(QStringLiteral("progressBar"), m_progressBar);
    sectionStatus->appendWidget(QStringLiteral("cancelButton"), m_cancelButton);
    selectionWidget->appendWidget(QStringLiteral("sectionSmartSelectStatus"), sectionStatus);
    updateOptionsUi();
    qInfo() << "[SmartSelect] event=tool.options.create.exit";
    return selectionWidget;
}

void SmartSelectTool::resetCursorStyle()
{
    if (selectionAction() == SELECTION_ADD) {
        useCursor(QCursor(Qt::CrossCursor));
    } else if (selectionAction() == SELECTION_SUBTRACT) {
        useCursor(QCursor(Qt::ForbiddenCursor));
    } else if (selectionAction() == SELECTION_INTERSECT) {
        useCursor(QCursor(Qt::PointingHandCursor));
    } else {
        KisToolSelect::resetCursorStyle();
    }
}

void SmartSelectTool::setHoverSegment(quint16 segmentId)
{
    if (m_hoverSegmentId == segmentId) {
        return;
    }

    const quint16 old = m_hoverSegmentId;
    m_hoverSegmentId = segmentId;
    requestSegmentUpdate(old);
    requestSegmentUpdate(m_hoverSegmentId);
}

void SmartSelectTool::requestSegmentUpdate(quint16 segmentId)
{
    if (!currentImage() || segmentId == 0) {
        return;
    }

    QRect bbox = m_segmentMap.segmentBounds(segmentId);
    if (bbox.isEmpty()) {
        return;
    }
    bbox = bbox.adjusted(-2, -2, 2, 2).intersected(currentImage()->bounds());
    canvas()->updateCanvas(currentImage()->pixelToDocument(bbox));
}

void SmartSelectTool::showNotEditableMessage()
{
    KisCanvas2 *kisCanvas = dynamic_cast<KisCanvas2 *>(canvas());
    if (!kisCanvas || !kisCanvas->viewManager()) {
        return;
    }

    kisCanvas->viewManager()->showFloatingMessage(i18n("Selection is not editable"),
                                                  QIcon(),
                                                  2000,
                                                  KisFloatingMessage::Medium,
                                                  Qt::AlignCenter);
}
