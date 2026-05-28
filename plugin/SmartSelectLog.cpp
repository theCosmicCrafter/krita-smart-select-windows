#include "SmartSelectLog.h"

#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonValue>
#include <QRect>
#include <QSize>

namespace
{

QString jsonCompact(const QJsonValue& value)
{
    if (value.isString()) {
        const QString text = value.toString();
        bool safe = !text.isEmpty();
        for (const QChar ch : text) {
            if (!(ch.isLetterOrNumber() || ch == '_' || ch == '-' || ch == '.' || ch == '/'
                    || ch == ':' || ch == '@' || ch == '+')) {
                safe = false;
                break;
            }
        }
        if (safe) {
            return text;
        }
    }

    if (value.isArray()) {
        return QString::fromUtf8(QJsonDocument(value.toArray()).toJson(QJsonDocument::Compact));
    }

    return QString::fromUtf8(QJsonDocument(QJsonArray { value }).toJson(QJsonDocument::Compact))
        .mid(1)
        .chopped(1);
}

QJsonValue variantToJsonValue(const QVariant& value)
{
    if (value.canConvert<QRect>()) {
        const QRect rect = value.toRect();
        return QJsonArray { rect.x(), rect.y(), rect.width(), rect.height() };
    }
    if (value.canConvert<QSize>()) {
        const QSize size = value.toSize();
        return QJsonArray { size.width(), size.height() };
    }
    const int type = value.userType();
    if (type == QMetaType::Bool) {
        return value.toBool();
    }
    if (type == QMetaType::Int || type == QMetaType::UInt || type == QMetaType::LongLong
        || type == QMetaType::ULongLong) {
        return value.toLongLong();
    }
    if (type == QMetaType::Float || type == QMetaType::Double) {
        return value.toDouble();
    }
    if (value.canConvert<QStringList>()) {
        QJsonArray array;
        for (const QString& item : value.toStringList()) {
            array.push_back(item);
        }
        return array;
    }
    return value.toString();
}

QString formatField(const QString& key, const QVariant& value)
{
    return key + QStringLiteral("=") + jsonCompact(variantToJsonValue(value));
}

void writeLog(SmartSelectLogLevel level, const QString& line)
{
    switch (level) {
    case SmartSelectLogLevel::Debug:
        qDebug().noquote() << line;
        return;
    case SmartSelectLogLevel::Info:
        qInfo().noquote() << line;
        return;
    case SmartSelectLogLevel::Warning:
        qWarning().noquote() << line;
        return;
    case SmartSelectLogLevel::Error:
        qCritical().noquote() << line;
        return;
    }
}

QString severityName(SmartSelectLogLevel level)
{
    switch (level) {
    case SmartSelectLogLevel::Debug:
        return QStringLiteral("debug");
    case SmartSelectLogLevel::Info:
        return QStringLiteral("info");
    case SmartSelectLogLevel::Warning:
        return QStringLiteral("warning");
    case SmartSelectLogLevel::Error:
        return QStringLiteral("error");
    }
    return QStringLiteral("info");
}

} // namespace

void logSmartSelectEvent(
    SmartSelectLogLevel level, const QString& eventName, const QVariantMap& fields)
{
    QStringList parts;
    parts << QStringLiteral("[SmartSelect]");
    parts << formatField(QStringLiteral("event.name"), eventName);
    parts << QStringLiteral("event.domain=smart_select");
    parts << formatField(QStringLiteral("severity"), severityName(level));
    parts << QStringLiteral("component=krita_plugin");
    for (auto it = fields.cbegin(); it != fields.cend(); ++it) {
        parts << formatField(it.key(), it.value());
    }

    writeLog(level, parts.join(QLatin1Char(' ')));
}
