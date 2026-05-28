#ifndef SMART_SELECT_TOOL_H_
#define SMART_SELECT_TOOL_H_

#include "SegmentMap.h"

#include <QPointer>
#include <QStringList>
#include <kis_tool_select_base.h>

class QLabel;
class QJsonArray;
class QProgressBar;
class QPushButton;
class WorkerClient;

class SmartSelectTool : public KisToolSelect
{
    Q_OBJECT

public:
    explicit SmartSelectTool(KoCanvasBase *canvas);
    ~SmartSelectTool() override;

    QWidget *createOptionWidget() override;
    void paint(QPainter &painter, const KoViewConverter &converter) override;

    void mouseMoveEvent(KoPointerEvent *event) override;
    void beginPrimaryAction(KoPointerEvent *event) override;
    void endPrimaryAction(KoPointerEvent *event) override;
    void resetCursorStyle() override;

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
    void activate(const QSet<KoShape *> &shapes) override;
    void deactivate() override;
    void updateImage(QRect const &rect);

protected:
    using KisToolSelectBase::m_widgetHelper;

private:
    enum class State {
        Idle,
        Preparing,
        Ready,
        Error,
        Cancelled
    };

    void prepareSegments();
    bool startWorkerIfNeeded();
    void startMockPrepareJob();
    void cancelActiveJob();
    void setState(State state, const QString &message = QString());
    void updateOptionsUi();
    void clearPreparedData();
    void cleanupTempFiles(const QStringList &paths);

    void onWorkerProgress(int id, const QString &stage, double progress, bool hasProgress);
    void onWorkerResult(int id, const QString &segmentMapPath, const QString &segmentsPath, const QJsonArray &metadata);
    void onWorkerError(int id, const QString &type, const QString &message);
    void onWorkerStderr(const QString &line);
    void onWorkerCrashed(const QString &message);

    void setHoverSegment(quint16 segmentId);
    void requestSegmentUpdate(quint16 segmentId);
    void showNotEditableMessage();

    SegmentMap m_segmentMap;
    quint16 m_hoverSegmentId = 0;
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
};

#endif // SMART_SELECT_TOOL_H_
