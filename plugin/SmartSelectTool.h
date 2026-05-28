#ifndef SMART_SELECT_TOOL_H_
#define SMART_SELECT_TOOL_H_

#include <KisToolOutlineBase.h>
#include <QPointer>
#include <QSize>
#include <QStringList>
#include <QVector>
#include <kis_tool_select_base.h>

class QLabel;
class QJsonArray;
class QProgressBar;
class QPushButton;
class WorkerClient;

class SmartSelectOutlineBase : public KisToolOutlineBase
{
    Q_OBJECT

public:
    explicit SmartSelectOutlineBase(KoCanvasBase *canvas);
};

class SmartSelectTool : public KisToolSelectBase<SmartSelectOutlineBase>
{
    Q_OBJECT

public:
    explicit SmartSelectTool(KoCanvasBase *canvas);
    ~SmartSelectTool() override;

    QWidget *createOptionWidget() override;
    void paint(QPainter& painter, const KoViewConverter& converter) override;

    void resetCursorStyle() override;
    bool primaryActionSupportsHiResEvents() const override;
    bool alternateActionSupportsHiResEvents(KisTool::AlternateAction action) const override;

protected:
    bool wantsAutoScroll() const override
    {
        return false;
    }

    bool isPixelOnly() const override
    {
        return true;
    }

    bool usesColorLabels() const override
    {
        return true;
    }

public Q_SLOTS:
    void activate(const QSet<KoShape *>& shapes) override;
    void deactivate() override;
    void updateImage(QRect const& rect);

protected:
    using KisToolSelectBase::m_widgetHelper;

private:
    enum class State
    {
        Idle,
        Preparing,
        Ready,
        Error,
        Cancelled
    };

    struct SelectionOptionsSnapshot
    {
        SelectionAction action = SELECTION_REPLACE;
        int grow = 0;
        int feather = 0;
        bool antiAlias = true;
    };

    void finishOutline(const QVector<QPointF>& points) override;
    void beginShape() override;
    void endShape() override;

    bool startWorkerIfNeeded();
    void startSelectionJob(const QVector<QPointF>& points);
    void cancelActiveJob();
    void setState(State state, const QString& message = QString());
    void updateOptionsUi();
    void cleanupTempFiles(const QStringList& paths);
    void clearActiveJobState();
    QString fingerprintPrefix(const QString& fingerprint) const;
    QString friendlyStageText(const QString& stage) const;

    void onWorkerProgress(int id, const QString& stage, double progress, bool hasProgress);
    void onWorkerResult(int id, const QString& segmentMapPath, const QString& segmentsPath,
        const QJsonArray& metadata);
    void onWorkerSelectionResult(int id, const QString& selectionMaskPath,
        const QString& segmentsPath, const QJsonArray& metadata);
    void onWorkerError(int id, const QString& type, const QString& message);
    void onWorkerStderr(const QString& line);
    void onWorkerCrashed(const QString& message);

    void showNotEditableMessage();

    State m_state = State::Idle;

    WorkerClient *m_worker = nullptr;
    int m_activeJobId = 0;
    QString m_stageText;
    double m_progress = 0.0;
    bool m_hasProgress = false;
    QString m_errorMessage;
    QPointer<QLabel> m_statusLabel;
    QPointer<QProgressBar> m_progressBar;
    QPointer<QPushButton> m_cancelButton;
    QStringList m_pendingCleanupPaths;
    QStringList m_activeJobCleanupPaths;
    quintptr m_activeJobImagePtr = 0;
    SelectionOptionsSnapshot m_activeSelectionOptions;
};

#endif // SMART_SELECT_TOOL_H_
