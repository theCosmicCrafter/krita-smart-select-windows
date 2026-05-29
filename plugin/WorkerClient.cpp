#include "WorkerClient.h"

#include "SmartSelectLog.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QProcessEnvironment>

namespace
{
QProcessEnvironment sanitizedWorkerEnvironment()
{
    QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
    const QStringList inheritedPythonKeys {
        QStringLiteral("PYTHONHOME"),
        QStringLiteral("PYTHONPATH"),
        QStringLiteral("PYTHONSTARTUP"),
        QStringLiteral("PYTHONUSERBASE"),
        QStringLiteral("PYTHONEXECUTABLE"),
        QStringLiteral("PYTHONSAFEPATH"),
        QStringLiteral("PYTHONINSPECT"),
        QStringLiteral("PYTHONBREAKPOINT"),
        QStringLiteral("PYTHONPLATLIBDIR"),
    };
    for (const QString& key : inheritedPythonKeys) {
        environment.remove(key);
    }

    // Krita AppImage exports library paths for its embedded Python/runtime. The
    // worker is intentionally launched with the system Python, so inheriting
    // those paths can break stdlib modules such as urllib HTTPS support.
    environment.remove(QStringLiteral("LD_LIBRARY_PATH"));
    environment.insert(QStringLiteral("PYTHONNOUSERSITE"), QStringLiteral("1"));
    return environment;
}
}

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
    m_process.setProcessEnvironment(sanitizedWorkerEnvironment());
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("worker.environment.sanitized"),
        { { QStringLiteral("environment.removed_keys"),
              QStringList { QStringLiteral("PYTHONHOME"), QStringLiteral("PYTHONPATH"),
                  QStringLiteral("PYTHONUSERBASE"), QStringLiteral("LD_LIBRARY_PATH") } },
            { QStringLiteral("python.no_user_site"), true } });
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

int WorkerClient::prepareSelection(const PrepareSelectionRequest& prepareRequest)
{
    const int id = m_nextId++;
    send(prepareSelectionRequestJson(id, prepareRequest));
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
            logSmartSelectEvent(SmartSelectLogLevel::Warning,
                QStringLiteral("worker.protocol.invalid_line"),
                { { QStringLiteral("error.type"), QStringLiteral("ProtocolError") },
                    { QStringLiteral("error.message"), parseError.errorString() },
                    { QStringLiteral("protocol.line.prefix"),
                        QString::fromUtf8(line.left(240)) } });
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
        if (resultObject.contains(QStringLiteral("model"))) {
            logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("selection.result.model"),
                { { QStringLiteral("job.id"), id },
                    { QStringLiteral("model.payload"),
                        QString::fromUtf8(
                            QJsonDocument(resultObject.value(QStringLiteral("model")).toObject())
                                .toJson(QJsonDocument::Compact)) } });
        }
        if (resultObject.contains(QStringLiteral("runtime"))) {
            logSmartSelectEvent(SmartSelectLogLevel::Info,
                QStringLiteral("selection.result.runtime"),
                { { QStringLiteral("job.id"), id },
                    { QStringLiteral("runtime.payload"),
                        QString::fromUtf8(
                            QJsonDocument(resultObject.value(QStringLiteral("runtime")).toObject())
                                .toJson(QJsonDocument::Compact)) } });
        }
        const QString selectionMaskPath
            = resultObject.value(QStringLiteral("selection_mask_path")).toString();
        if (!selectionMaskPath.isEmpty()) {
            Q_EMIT selectionResult(id, selectionMaskPath,
                resultObject.value(QStringLiteral("metadata")).toArray());
            return;
        }

        Q_EMIT error(id, "ProtocolError", "Smart Select worker returned no selection mask");
        return;
    }

    if (message.contains("error")) {
        const QJsonObject errorObject = message.value("error").toObject();
        Q_EMIT error(id, errorObject.value("type").toString("WorkerError"),
            errorObject.value("message").toString("Smart Select worker failed"));
    }
}
