#include "SmartSelectTool.h"

#include "LassoRegionMask.h"
#include "SelectionApply.h"
#include "SelectionMask.h"
#include "SmartSelectLog.h"
#include "SmartSelectRuntimeConfig.h"
#include "SnapshotCapture.h"
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

#include <KisOptionCollectionWidget.h>
#include <KisViewManager.h>
#include <KoPointerEvent.h>
#include <canvas/kis_canvas2.h>
#include <kis_cursor.h>
#include <kis_floating_message.h>
#include <kis_image.h>
#include <kis_selection_tool_config_widget_helper.h>
#include <klocalizedstring.h>

namespace
{

const QString DEFAULT_MODEL_ID = QStringLiteral("sam2.1_hiera_large");
const QString ALGORITHM_VERSION = QStringLiteral("sam2.1-lasso-selection-hard-v1");

QStringList appendPath(QStringList paths, const QString& path)
{
    if (!path.isEmpty()) {
        paths.append(path);
    }
    return paths;
}

} // namespace

SmartSelectOutlineBase::SmartSelectOutlineBase(KoCanvasBase *canvas)
    : KisToolOutlineBase(canvas, KisToolOutlineBase::SELECT, QCursor(Qt::CrossCursor))
{
    setObjectName("tool_smart_select");
}

SmartSelectTool::SmartSelectTool(KoCanvasBase *canvas)
    : KisToolSelectBase<SmartSelectOutlineBase>(canvas, i18n("Smart Select"))
{
    setObjectName("tool_smart_select");
}

SmartSelectTool::~SmartSelectTool()
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.destroy"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("enter") },
            { QStringLiteral("tool.state"), int(m_state) },
            { QStringLiteral("job.active_id"), m_activeJobId } });

    if (m_worker) {
        disconnect(m_worker, nullptr, this, nullptr);
        m_worker->stop();
    }
    cleanupTempFiles(m_activeJobCleanupPaths);
    clearActiveJobState();

    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.destroy"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("exit") } });
}

void SmartSelectTool::activate(const QSet<KoShape *>& shapes)
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.activate"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("enter") },
            { QStringLiteral("tool.name"), QStringLiteral("SmartSelectTool") },
            { QStringLiteral("tool.state"), int(m_state) },
            { QStringLiteral("job.active_id"), m_activeJobId },
            { QStringLiteral("ui.status_widget.exists"), !m_statusLabel.isNull() } });

    KisToolSelectBase<SmartSelectOutlineBase>::activate(shapes);

    if (KisImage *image = currentImage().data()) {
        connect(image, SIGNAL(sigImageUpdated(QRect)), this, SLOT(updateImage(QRect)),
            Qt::UniqueConnection);
        logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.activate"),
            { { QStringLiteral("lifecycle.action"), QStringLiteral("image_connected") },
                { QStringLiteral("image.bounds"), image->bounds() } });
    } else {
        logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("tool.activate"),
            { { QStringLiteral("error.reason"), QStringLiteral("no_image") } });
    }

    setState(State::Idle);
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.activate"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("exit") },
            { QStringLiteral("tool.state"), int(m_state) },
            { QStringLiteral("job.active_id"), m_activeJobId } });
}

void SmartSelectTool::deactivate()
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.deactivate"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("enter") },
            { QStringLiteral("tool.state"), int(m_state) },
            { QStringLiteral("job.active_id"), m_activeJobId } });

    if (KisImage *image = currentImage().data()) {
        disconnect(image, SIGNAL(sigImageUpdated(QRect)), this, SLOT(updateImage(QRect)));
        logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.deactivate"),
            { { QStringLiteral("lifecycle.action"), QStringLiteral("image_disconnected") } });
    }

    cancelActiveJob();
    m_state = State::Idle;
    m_stageText.clear();
    m_progress = 0.0;
    m_hasProgress = false;
    m_errorMessage.clear();

    KisToolSelectBase<SmartSelectOutlineBase>::deactivate();
    updateOptionsUi();
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.deactivate"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("exit") } });
}

void SmartSelectTool::updateImage(QRect const&)
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.image.updated"));
    cancelActiveJob();
    setState(State::Idle);
}

bool SmartSelectTool::startWorkerIfNeeded()
{
    if (!m_worker) {
        m_worker = new WorkerClient(this);
        connect(m_worker, &WorkerClient::progress, this, &SmartSelectTool::onWorkerProgress);
        connect(m_worker, &WorkerClient::result, this, &SmartSelectTool::onWorkerResult);
        connect(m_worker, &WorkerClient::selectionResult, this,
            &SmartSelectTool::onWorkerSelectionResult);
        connect(m_worker, &WorkerClient::error, this, &SmartSelectTool::onWorkerError);
        connect(m_worker, &WorkerClient::stderrLine, this, &SmartSelectTool::onWorkerStderr);
        connect(m_worker, &WorkerClient::crashed, this, &SmartSelectTool::onWorkerCrashed);
    }

    const SmartSelectRuntimeConfig& config = SmartSelectRuntimeConfig::instance();
    const QString pythonExecutable = config.pythonExecutable();
    const QString workerScript = config.workerScriptPath();

    if (pythonExecutable.isEmpty() || !QFileInfo::exists(pythonExecutable)) {
        logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("worker.start.failed"),
            { { QStringLiteral("error.reason"), QStringLiteral("python_executable_not_found") },
                { QStringLiteral("python.executable.path"), pythonExecutable } });
        setState(State::Error, i18n("python executable not found"));
        return false;
    }

    if (workerScript.isEmpty() || !QFileInfo::exists(workerScript)) {
        logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("worker.start.failed"),
            { { QStringLiteral("error.reason"), QStringLiteral("worker_script_not_found") },
                { QStringLiteral("worker.script.path"), workerScript } });
        setState(State::Error, i18n("worker script not found"));
        return false;
    }

    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("worker.start"),
        { { QStringLiteral("python.executable.path"), pythonExecutable },
            { QStringLiteral("worker.script.path"), workerScript } });
    if (!m_worker->start(pythonExecutable, workerScript)) {
        setState(State::Error, i18n("worker failed to start"));
        return false;
    }

    return true;
}

void SmartSelectTool::finishOutline(const QVector<QPointF>& points)
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("lasso.finish"),
        { { QStringLiteral("lasso.points.count"), points.size() },
            { QStringLiteral("tool.state"), int(m_state) } });

    if (!selectionEditable()) {
        showNotEditableMessage();
        return;
    }
    if (points.size() < 3) {
        return;
    }

    startSelectionJob(points);
}

void SmartSelectTool::beginShape()
{
    beginSelectInteraction();
}

void SmartSelectTool::endShape()
{
    endSelectInteraction();
}

void SmartSelectTool::startSelectionJob(const QVector<QPointF>& points)
{
    cancelActiveJob();

    if (!currentImage()) {
        setState(State::Error, i18n("no active image"));
        return;
    }

    LassoRegionMask regionMask;
    QString error;
    if (!buildLassoRegionMask(points, currentImage()->bounds().size(), &regionMask, &error)) {
        logSmartSelectEvent(SmartSelectLogLevel::Warning,
            QStringLiteral("lasso.region_mask.failed"),
            { { QStringLiteral("error.reason"), error } });
        setState(State::Error, error);
        return;
    }

    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("lasso.region_mask.created"),
        { { QStringLiteral("region.bbox"), regionMask.bbox },
            { QStringLiteral("region.area"), regionMask.area } });

    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("snapshot.capture.start"),
        { { QStringLiteral("image.bounds"), currentImage()->bounds() } });
    SnapshotInput snapshot;
    if (!captureSnapshot(currentImage(), &snapshot, &error)) {
        logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("snapshot.capture.failed"),
            { { QStringLiteral("error.reason"), error } });
        setState(State::Error, error);
        return;
    }

    if (!snapshot.hasVisiblePixels) {
        setState(State::Error, i18n("image has no visible pixels"));
        return;
    }

    SnapshotFiles files;
    if (!writeSnapshotFiles(snapshot, &files, &error)) {
        logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("snapshot.files.failed"),
            { { QStringLiteral("error.reason"), error } });
        setState(State::Error, error);
        return;
    }

    const QString regionMaskPath
        = QDir(files.tempDirPath).filePath(QStringLiteral("region_mask.npy"));
    if (!writeLassoRegionMaskNpy(regionMaskPath, regionMask, &error)) {
        cleanupTempFiles(files.cleanupPaths);
        logSmartSelectEvent(SmartSelectLogLevel::Warning,
            QStringLiteral("lasso.region_mask.failed"),
            { { QStringLiteral("error.reason"), error } });
        setState(State::Error, error);
        return;
    }
    files.cleanupPaths = appendPath(files.cleanupPaths, regionMaskPath);

    if (!startWorkerIfNeeded()) {
        cleanupTempFiles(files.cleanupPaths);
        return;
    }

    const SmartSelectRuntimeConfig& config = SmartSelectRuntimeConfig::instance();
    const QString modelCacheDir = config.modelCacheDir();
    const QString runtimeCacheDir = config.runtimeCacheDir();
    if (modelCacheDir.isEmpty() || !QDir().mkpath(modelCacheDir)
        || !QFileInfo(modelCacheDir).isWritable()) {
        logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("model.cache.failed"),
            { { QStringLiteral("error.reason"), QStringLiteral("model_cache_not_writable") },
                { QStringLiteral("model.cache.dir"), modelCacheDir } });
        cleanupTempFiles(files.cleanupPaths);
        setState(State::Error, i18n("model cache is not writable"));
        return;
    }
    if (runtimeCacheDir.isEmpty() || !QDir().mkpath(runtimeCacheDir)
        || !QFileInfo(runtimeCacheDir).isWritable()) {
        logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("runtime.cache.failed"),
            { { QStringLiteral("error.reason"), QStringLiteral("runtime_cache_not_writable") },
                { QStringLiteral("runtime.cache.dir"), runtimeCacheDir } });
        cleanupTempFiles(files.cleanupPaths);
        setState(State::Error, i18n("runtime cache is not writable"));
        return;
    }

    PrepareSelectionRequest request;
    request.imagePath = files.imagePath;
    request.alphaMaskPath = files.alphaMaskPath;
    request.regionMaskPath = regionMaskPath;
    request.regionBbox = regionMask.bbox;
    request.modelId = DEFAULT_MODEL_ID;
    request.algorithmVersion = ALGORITHM_VERSION;
    request.fingerprint = snapshot.fingerprint;
    request.modelCacheDir = modelCacheDir;
    request.runtimeCacheDir = runtimeCacheDir;
    request.canvasSize = snapshot.canvasSize;

    m_activeSelectionOptions.action = selectionAction();
    m_activeSelectionOptions.grow = growSelection();
    m_activeSelectionOptions.feather = featherSelection();
    m_activeSelectionOptions.antiAlias = antiAliasSelection();

    setState(State::Preparing);
    m_activeJobId = m_worker->prepareSelection(request);
    m_activeJobCleanupPaths = files.cleanupPaths;
    m_activeJobImagePtr = reinterpret_cast<quintptr>(currentImage().data());

    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("selection.start"),
        { { QStringLiteral("job.id"), m_activeJobId },
            { QStringLiteral("job.type"), QStringLiteral("lasso_selection") },
            { QStringLiteral("model.id"), request.modelId },
            { QStringLiteral("algorithm.version"), request.algorithmVersion },
            { QStringLiteral("image.fingerprint.prefix"), fingerprintPrefix(request.fingerprint) },
            { QStringLiteral("image.size"), request.canvasSize },
            { QStringLiteral("region.bbox"), request.regionBbox } });
}

void SmartSelectTool::cancelActiveJob()
{
    if (m_worker && m_activeJobId != 0) {
        logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("selection.cancelled"),
            { { QStringLiteral("job.id"), m_activeJobId },
                { QStringLiteral("job.type"), QStringLiteral("lasso_selection") } });
        m_worker->cancel(m_activeJobId);
    }
    cleanupTempFiles(m_activeJobCleanupPaths);
    clearActiveJobState();
}

void SmartSelectTool::setState(State state, const QString& message)
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.state.change"),
        { { QStringLiteral("tool.state.from"), int(m_state) },
            { QStringLiteral("tool.state.to"), int(state) },
            { QStringLiteral("error.message"), message },
            { QStringLiteral("ui.status_widget.exists"), !m_statusLabel.isNull() } });
    m_state = state;
    if (state == State::Preparing) {
        m_stageText.clear();
        m_progress = 0.0;
        m_hasProgress = false;
        m_errorMessage.clear();
    } else if (state == State::Error) {
        m_errorMessage = message;
    } else if (state == State::Ready || state == State::Idle) {
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
            text = m_stageText.isEmpty() ? i18n("Preparing selection...")
                                         : friendlyStageText(m_stageText);
            break;
        case State::Ready:
            text = i18n("Smart Select: selection applied");
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

void SmartSelectTool::clearActiveJobState()
{
    m_activeJobId = 0;
    m_activeJobCleanupPaths.clear();
    m_activeJobImagePtr = 0;
}

QString SmartSelectTool::fingerprintPrefix(const QString& fingerprint) const
{
    return fingerprint.left(12);
}

QString SmartSelectTool::friendlyStageText(const QString& stage) const
{
    if (stage == QStringLiteral("checking_runtime")) {
        return i18n("Checking SAM2 runtime...");
    }
    if (stage == QStringLiteral("creating_runtime")) {
        return i18n("Creating SAM2 runtime...");
    }
    if (stage == QStringLiteral("installing_runtime_dependencies")) {
        return i18n("Installing SAM2 runtime...");
    }
    if (stage == QStringLiteral("validating_runtime")) {
        return i18n("Validating SAM2 runtime...");
    }
    if (stage == QStringLiteral("resolving_model")) {
        return i18n("Preparing model...");
    }
    if (stage == QStringLiteral("checking_model_cache")) {
        return i18n("Checking model cache...");
    }
    if (stage == QStringLiteral("downloading_model")) {
        return i18n("Downloading SAM2.1 Large...");
    }
    if (stage == QStringLiteral("verifying_model")) {
        return i18n("Verifying SAM2.1 Large...");
    }
    if (stage == QStringLiteral("reading_image")) {
        return i18n("Reading image...");
    }
    if (stage == QStringLiteral("preprocessing_image")) {
        return i18n("Preparing image...");
    }
    if (stage == QStringLiteral("loading_model")) {
        return i18n("Loading SAM2.1 Large...");
    }
    if (stage == QStringLiteral("running_sam")) {
        return i18n("Running SAM2.1 Large...");
    }
    if (stage == QStringLiteral("building_selection_mask")) {
        return i18n("Building selection...");
    }
    return stage;
}

void SmartSelectTool::cleanupTempFiles(const QStringList& paths)
{
    if (qEnvironmentVariableIsSet("SMART_SELECT_KEEP_DEBUG_FILES")) {
        for (const QString& path : paths) {
            if (!path.isEmpty() && QFileInfo::exists(path)) {
                logSmartSelectEvent(SmartSelectLogLevel::Info,
                    QStringLiteral("temp_file.kept_for_debug"),
                    { { QStringLiteral("file.path"), path } });
            }
        }
        return;
    }

    for (const QString& path : paths) {
        if (path.isEmpty()) {
            continue;
        }

        QFileInfo info(path);
        const QString parentPath = info.absolutePath();
        if (QFile::remove(path)) {
            logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("temp_file.cleaned"),
                { { QStringLiteral("file.path"), path } });
        } else if (QFileInfo::exists(path)) {
            logSmartSelectEvent(SmartSelectLogLevel::Warning,
                QStringLiteral("temp_file.cleanup_failed"),
                { { QStringLiteral("file.path"), path } });
        }

        QDir parent(parentPath);
        if (parent.dirName().startsWith(QStringLiteral("krita-smart-select-"))
            && parent.isEmpty()) {
            if (parent.rmdir(parentPath)) {
                logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("temp_dir.cleaned"),
                    { { QStringLiteral("file.path"), parentPath } });
            }
        }
    }
}

void SmartSelectTool::onWorkerProgress(
    int id, const QString& stage, double progress, bool hasProgress)
{
    if (id != m_activeJobId) {
        return;
    }

    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("selection.progress"),
        { { QStringLiteral("job.id"), id },
            { QStringLiteral("job.type"), QStringLiteral("lasso_selection") },
            { QStringLiteral("job.stage"), stage }, { QStringLiteral("job.progress"), progress },
            { QStringLiteral("job.progress.available"), hasProgress } });
    m_stageText = stage;
    m_progress = progress;
    m_hasProgress = hasProgress;
    updateOptionsUi();
}

void SmartSelectTool::onWorkerResult(
    int id, const QString& segmentMapPath, const QString& segmentsPath, const QJsonArray& metadata)
{
    Q_UNUSED(metadata);
    logSmartSelectEvent(SmartSelectLogLevel::Info,
        QStringLiteral("inference.result.ignored_legacy"),
        { { QStringLiteral("job.id"), id },
            { QStringLiteral("job.type"), QStringLiteral("legacy_prepare_segments") },
            { QStringLiteral("file.segment_map.path"), segmentMapPath } });
    cleanupTempFiles(QStringList { segmentMapPath, segmentsPath });
}

void SmartSelectTool::onWorkerSelectionResult(int id, const QString& selectionMaskPath,
    const QString& segmentsPath, const QJsonArray& metadata)
{
    const QStringList resultCleanupPaths { selectionMaskPath, segmentsPath };
    const bool staleImage = !currentImage()
        || reinterpret_cast<quintptr>(currentImage().data()) != m_activeJobImagePtr;

    if (id != m_activeJobId || staleImage) {
        logSmartSelectEvent(SmartSelectLogLevel::Info,
            QStringLiteral("selection.result.ignored_stale"),
            { { QStringLiteral("job.id"), id }, { QStringLiteral("job.active_id"), m_activeJobId },
                { QStringLiteral("job.type"), QStringLiteral("lasso_selection") },
                { QStringLiteral("image.stale"), staleImage } });
        cleanupTempFiles(resultCleanupPaths);
        return;
    }

    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("selection.result"),
        { { QStringLiteral("job.id"), id },
            { QStringLiteral("job.type"), QStringLiteral("lasso_selection") },
            { QStringLiteral("file.selection_mask.path"), selectionMaskPath },
            { QStringLiteral("file.segments.path"), segmentsPath },
            { QStringLiteral("metadata.count"), metadata.size() } });

    QString error;
    SelectionMask mask;
    if (!mask.loadNpy(selectionMaskPath, &error)) {
        cleanupTempFiles(resultCleanupPaths);
        cleanupTempFiles(m_activeJobCleanupPaths);
        clearActiveJobState();
        setState(State::Error, error);
        return;
    }

    const bool applied = applySelectionMask(canvas(), currentImage(), currentNode(), mask,
        m_activeSelectionOptions.action, m_activeSelectionOptions.grow,
        m_activeSelectionOptions.feather, m_activeSelectionOptions.antiAlias);

    cleanupTempFiles(m_activeJobCleanupPaths);
    clearActiveJobState();
    cleanupTempFiles(resultCleanupPaths);

    if (!applied) {
        setState(State::Error, i18n("selection is empty"));
        return;
    }

    setState(State::Ready);
}

void SmartSelectTool::onWorkerError(int id, const QString& type, const QString& message)
{
    if (id != 0 && id != m_activeJobId) {
        logSmartSelectEvent(SmartSelectLogLevel::Info,
            QStringLiteral("selection.error.ignored_stale"),
            { { QStringLiteral("job.id"), id }, { QStringLiteral("job.active_id"), m_activeJobId },
                { QStringLiteral("error.type"), type },
                { QStringLiteral("error.message"), message } });
        return;
    }

    logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("selection.failed"),
        { { QStringLiteral("job.id"), id },
            { QStringLiteral("job.type"), QStringLiteral("lasso_selection") },
            { QStringLiteral("error.type"), type }, { QStringLiteral("error.message"), message } });
    cleanupTempFiles(m_activeJobCleanupPaths);
    clearActiveJobState();
    setState(State::Error, message);
}

void SmartSelectTool::onWorkerStderr(const QString& line)
{
    logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("worker.stderr"),
        { { QStringLiteral("worker.stderr.line"), line } });
}

void SmartSelectTool::onWorkerCrashed(const QString& message)
{
    logSmartSelectEvent(SmartSelectLogLevel::Warning, QStringLiteral("worker.crashed"),
        { { QStringLiteral("error.message"), message } });
    if (m_state == State::Preparing || m_activeJobId != 0) {
        cleanupTempFiles(m_activeJobCleanupPaths);
        clearActiveJobState();
        setState(State::Error, message);
    }
}

void SmartSelectTool::paint(QPainter& painter, const KoViewConverter& converter)
{
    KisToolSelectBase<SmartSelectOutlineBase>::paint(painter, converter);
}

QWidget *SmartSelectTool::createOptionWidget()
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.options.create"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("enter") } });
    KisToolSelectBase<SmartSelectOutlineBase>::createOptionWidget();
    KisSelectionOptions *selectionWidget = selectionOptionWidget();
    selectionWidget->setReferenceSectionVisible(false);

    KisOptionCollectionWidgetWithHeader *sectionStatus
        = new KisOptionCollectionWidgetWithHeader(i18n("Smart Select"));

    m_statusLabel = new QLabel;
    m_progressBar = new QProgressBar;
    m_cancelButton = new QPushButton(i18n("Cancel"));
    connect(m_cancelButton, &QPushButton::clicked, this, &SmartSelectTool::cancelActiveJob);

    sectionStatus->appendWidget(QStringLiteral("statusLabel"), m_statusLabel);
    sectionStatus->appendWidget(QStringLiteral("progressBar"), m_progressBar);
    sectionStatus->appendWidget(QStringLiteral("cancelButton"), m_cancelButton);
    selectionWidget->appendWidget(QStringLiteral("sectionSmartSelectStatus"), sectionStatus);
    updateOptionsUi();
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("tool.options.create"),
        { { QStringLiteral("lifecycle.action"), QStringLiteral("exit") } });
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
        KisToolSelectBase<SmartSelectOutlineBase>::resetCursorStyle();
    }
}

bool SmartSelectTool::primaryActionSupportsHiResEvents() const
{
    return !isMovingSelection();
}

bool SmartSelectTool::alternateActionSupportsHiResEvents(KisTool::AlternateAction action) const
{
    Q_UNUSED(action);
    return !isMovingSelection();
}

void SmartSelectTool::showNotEditableMessage()
{
    KisCanvas2 *kisCanvas = dynamic_cast<KisCanvas2 *>(canvas());
    if (!kisCanvas || !kisCanvas->viewManager()) {
        return;
    }

    kisCanvas->viewManager()->showFloatingMessage(i18n("Selection is not editable"), QIcon(), 2000,
        KisFloatingMessage::Medium, Qt::AlignCenter);
}
