#ifndef SMART_SELECT_WORKER_CLIENT_H_
#define SMART_SELECT_WORKER_CLIENT_H_

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QRect>
#include <QSize>

struct PrepareSelectionRequest
{
    QString imagePath;
    QString alphaMaskPath;
    QString regionMaskPath;
    QRect regionBbox;
    QString modelId;
    QString algorithmVersion;
    QString fingerprint;
    QString modelCacheDir;
    QString runtimeCacheDir;
    QSize canvasSize;
};

inline QJsonObject prepareSelectionRequestJson(int id, const PrepareSelectionRequest& request)
{
    QJsonObject params;
    params["image_path"] = request.imagePath;
    params["alpha_mask_path"] = request.alphaMaskPath;
    params["region_mask_path"] = request.regionMaskPath;

    QJsonArray regionBbox;
    regionBbox.push_back(request.regionBbox.x());
    regionBbox.push_back(request.regionBbox.y());
    regionBbox.push_back(request.regionBbox.width());
    regionBbox.push_back(request.regionBbox.height());
    params["region_bbox"] = regionBbox;

    params["model_id"] = request.modelId;
    params["algorithm_version"] = request.algorithmVersion;
    params["fingerprint"] = request.fingerprint;
    params["model_cache_dir"] = request.modelCacheDir;
    params["runtime_cache_dir"] = request.runtimeCacheDir;
    params["canvas_width"] = request.canvasSize.width();
    params["canvas_height"] = request.canvasSize.height();

    QJsonObject message;
    message["id"] = id;
    message["method"] = QStringLiteral("prepare_selection");
    message["params"] = params;
    return message;
}

class WorkerClient : public QObject
{
    Q_OBJECT

public:
    explicit WorkerClient(QObject *parent = nullptr);
    ~WorkerClient() override;

    bool start(const QString& pythonExecutable, const QString& workerScript);
    void stop();

    int prepareSelection(const PrepareSelectionRequest& request);
    void cancel(int jobId);

Q_SIGNALS:
    void progress(int id, const QString& stage, double progress, bool hasProgress);
    void selectionResult(int id, const QString& selectionMaskPath, const QJsonArray& metadata);
    void error(int id, const QString& type, const QString& message);
    void stderrLine(const QString& line);
    void crashed(const QString& message);

private Q_SLOTS:
    void readStdout();
    void readStderr();
    void processFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
    void send(const QJsonObject& message);
    void handleMessage(const QJsonObject& message);

    QProcess m_process;
    QByteArray m_stdoutBuffer;
    int m_nextId = 1;
};

#endif // SMART_SELECT_WORKER_CLIENT_H_
