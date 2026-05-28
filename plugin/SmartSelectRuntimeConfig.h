#ifndef SMART_SELECT_RUNTIME_CONFIG_H_
#define SMART_SELECT_RUNTIME_CONFIG_H_

#include <QString>

class SmartSelectRuntimeConfig
{
public:
    static SmartSelectRuntimeConfig& instance();

    void setPackageDir(const QString& path);
    void setPythonExecutable(const QString& path);

    QString packageDir() const;
    QString workerScriptPath() const;
    QString pythonExecutable() const;
    QString modelCacheDir() const;

private:
    SmartSelectRuntimeConfig() = default;

    QString m_packageDir;
    QString m_pythonExecutable;
};

#endif // SMART_SELECT_RUNTIME_CONFIG_H_
