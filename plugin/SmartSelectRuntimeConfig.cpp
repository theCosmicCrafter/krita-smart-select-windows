#include "SmartSelectRuntimeConfig.h"

#include <QDir>
#include <QStandardPaths>

SmartSelectRuntimeConfig& SmartSelectRuntimeConfig::instance()
{
    static SmartSelectRuntimeConfig config;
    return config;
}

void SmartSelectRuntimeConfig::setPackageDir(const QString& path)
{
    m_packageDir = QDir::cleanPath(path);
}

void SmartSelectRuntimeConfig::setPythonExecutable(const QString& path)
{
    m_pythonExecutable = QDir::cleanPath(path);
}

QString SmartSelectRuntimeConfig::packageDir() const
{
    return m_packageDir;
}

QString SmartSelectRuntimeConfig::workerScriptPath() const
{
    if (m_packageDir.isEmpty()) {
        return QString();
    }

    return QDir(m_packageDir).filePath(QStringLiteral("worker/smart_select_worker.py"));
}

QString SmartSelectRuntimeConfig::pythonExecutable() const
{
    if (!m_pythonExecutable.isEmpty()) {
        return m_pythonExecutable;
    }

    return QStandardPaths::findExecutable(QStringLiteral("python3"));
}

QString SmartSelectRuntimeConfig::modelCacheDir() const
{
    QString base = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    if (base.isEmpty()) {
        base = QDir::homePath() + QStringLiteral("/.local/share/krita-smart-select");
    }

    return QDir(base).filePath(QStringLiteral("smart_select/models"));
}

QString SmartSelectRuntimeConfig::runtimeCacheDir() const
{
    QString base = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    if (base.isEmpty()) {
        base = QDir::homePath() + QStringLiteral("/.local/share/krita-smart-select");
    }

    return QDir(base).filePath(QStringLiteral("smart_select/runtime"));
}
