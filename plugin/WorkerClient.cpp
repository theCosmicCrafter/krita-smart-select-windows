#include "WorkerClient.h"

#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QProcessEnvironment>

WorkerClient::WorkerClient(QObject *parent)
    : QObject(parent)
{
    connect(&m_process, SIGNAL(readyReadStandardOutput()), this, SLOT(readStdout()));
    connect(&m_process, SIGNAL(readyReadStandardError()), this, SLOT(readStderr()));
    connect(&m_process, SIGNAL(finished(int, QProcess::ExitStatus)), this,
        SLOT(processFinished(int, QProcess::ExitStatus)));
}

WorkerClient::~WorkerClient()
{
    stop();
}

bool WorkerClient::start(const QString& pythonExecutable, const QString& workerScript)
{
    if (m_process.state() != QProcess::NotRunning) {
        return true;
    }

    m_stdoutBuffer.clear();
    m_process.setProcessChannelMode(QProcess::SeparateChannels);
    m_process.start(pythonExecutable, QStringList { workerScript });
    if (!m_process.waitForStarted(1000)) {
        Q_EMIT crashed(m_process.errorString());
        return false;
    }
    return true;
}

void WorkerClient::stop()
{
    if (m_process.state() == QProcess::NotRunning) {
        return;
    }
    m_process.terminate();
    if (!m_process.waitForFinished(1000)) {
        m_process.kill();
        m_process.waitForFinished(1000);
    }
}

int WorkerClient::prepareSegments(const PrepareSegmentsRequest& prepareRequest)
{
    const int id = m_nextId++;
    QJsonObject params;
    params["image_path"] = prepareRequest.imagePath;
    params["alpha_mask_path"] = prepareRequest.alphaMaskPath;
    params["model_id"] = prepareRequest.modelId;
    params["algorithm_version"] = prepareRequest.algorithmVersion;
    params["fingerprint"] = prepareRequest.fingerprint;
    params["canvas_width"] = prepareRequest.canvasSize.width();
    params["canvas_height"] = prepareRequest.canvasSize.height();

    QJsonObject request;
    request["id"] = id;
    request["method"] = "prepare_segments";
    request["params"] = params;
    send(request);
    return id;
}

void WorkerClient::cancel(int jobId)
{
    QJsonObject params;
    params["job_id"] = jobId;

    QJsonObject request;
    request["id"] = m_nextId++;
    request["method"] = "cancel";
    request["params"] = params;
    send(request);
}

void WorkerClient::readStdout()
{
    m_stdoutBuffer.append(m_process.readAllStandardOutput());

    while (true) {
        const int newline = m_stdoutBuffer.indexOf('\n');
        if (newline < 0) {
            break;
        }
        const QByteArray line = m_stdoutBuffer.left(newline).trimmed();
        m_stdoutBuffer.remove(0, newline + 1);
        if (line.isEmpty()) {
            continue;
        }

        QJsonParseError parseError;
        const QJsonDocument document = QJsonDocument::fromJson(line, &parseError);
        if (parseError.error != QJsonParseError::NoError || !document.isObject()) {
            Q_EMIT error(0, "ProtocolError", parseError.errorString());
            continue;
        }
        handleMessage(document.object());
    }
}

void WorkerClient::readStderr()
{
    const QList<QByteArray> lines = m_process.readAllStandardError().split('\n');
    for (const QByteArray& line : lines) {
        const QString text = QString::fromUtf8(line).trimmed();
        if (!text.isEmpty()) {
            Q_EMIT stderrLine(text);
        }
    }
}

void WorkerClient::processFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    if (exitStatus == QProcess::CrashExit || exitCode != 0) {
        Q_EMIT crashed(QStringLiteral("Worker exited with code %1").arg(exitCode));
    }
}

void WorkerClient::send(const QJsonObject& message)
{
    if (m_process.state() != QProcess::Running) {
        Q_EMIT error(
            message.value("id").toInt(), "WorkerNotRunning", "Smart Select worker is not running");
        return;
    }

    const QByteArray payload = QJsonDocument(message).toJson(QJsonDocument::Compact) + '\n';
    m_process.write(payload);
}

void WorkerClient::handleMessage(const QJsonObject& message)
{
    const int id = message.value("id").toInt();
    if (message.contains("event") && message.value("event").toString() == "progress") {
        const QJsonValue progressValue = message.value("progress");
        Q_EMIT progress(id, message.value("stage").toString(),
            progressValue.isDouble() ? progressValue.toDouble() : 0.0, progressValue.isDouble());
        return;
    }

    if (message.contains("result")) {
        const QJsonObject resultObject = message.value("result").toObject();
        Q_EMIT result(id, resultObject.value("segment_map_path").toString(),
            resultObject.value("segments_path").toString(),
            resultObject.value("metadata").toArray());
        return;
    }

    if (message.contains("error")) {
        const QJsonObject errorObject = message.value("error").toObject();
        Q_EMIT error(id, errorObject.value("type").toString("WorkerError"),
            errorObject.value("message").toString("Smart Select worker failed"));
    }
}
