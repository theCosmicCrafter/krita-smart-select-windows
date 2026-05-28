#ifndef SMART_SELECT_WORKER_CLIENT_H_
#define SMART_SELECT_WORKER_CLIENT_H_

#include <QJsonArray>
#include <QObject>
#include <QProcess>
#include <QSize>

class WorkerClient : public QObject
{
    Q_OBJECT

public:
    explicit WorkerClient(QObject *parent = nullptr);
    ~WorkerClient() override;

    bool start(const QString& pythonExecutable, const QString& workerScript);
    void stop();

    int prepareSegments(const QString& imagePath, const QString& modelId,
        const QString& fingerprint, const QSize& canvasSize);
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
