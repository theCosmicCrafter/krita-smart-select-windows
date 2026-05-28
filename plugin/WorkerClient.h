#ifndef SMART_SELECT_WORKER_CLIENT_H_
#define SMART_SELECT_WORKER_CLIENT_H_

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QProcess>
#include <QSize>

struct PrepareSegmentsRequest
{
    QString imagePath;
    QString alphaMaskPath;
    QString modelId;
    QString algorithmVersion;
    QString fingerprint;
    QString modelCacheDir;
    QSize canvasSize;
};

inline QJsonObject prepareSegmentsRequestJson(int id, const PrepareSegmentsRequest& request)
{
    QJsonObject params;
    params["image_path"] = request.imagePath;
    params["alpha_mask_path"] = request.alphaMaskPath;
    params["model_id"] = request.modelId;
    params["algorithm_version"] = request.algorithmVersion;
    params["fingerprint"] = request.fingerprint;
    params["model_cache_dir"] = request.modelCacheDir;
    params["canvas_width"] = request.canvasSize.width();
    params["canvas_height"] = request.canvasSize.height();

    QJsonObject message;
    message["id"] = id;
    message["method"] = QStringLiteral("prepare_segments");
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

    int prepareSegments(const PrepareSegmentsRequest& request);
    void cancel(int jobId);

Q_SIGNALS:
    void progress(int id, const QString& stage, double progress, bool hasProgress);
    void result(int id, const QString& segmentMapPath, const QString& segmentsPath,
        const QJsonArray& metadata);
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
