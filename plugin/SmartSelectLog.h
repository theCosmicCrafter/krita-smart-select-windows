#ifndef SMART_SELECT_LOG_H_
#define SMART_SELECT_LOG_H_

#include <QString>
#include <QVariantMap>

enum class SmartSelectLogLevel
{
    Debug,
    Info,
    Warning,
    Error,
};

void logSmartSelectEvent(
    SmartSelectLogLevel level, const QString& eventName, const QVariantMap& fields = QVariantMap());

#endif // SMART_SELECT_LOG_H_
