#ifndef SMART_SELECT_PLUGIN_H_
#define SMART_SELECT_PLUGIN_H_

#include <QList>
#include <QObject>
#include <QString>
#include <QVariant>

class SmartSelectPlugin : public QObject
{
    Q_OBJECT

public:
    SmartSelectPlugin(QObject *parent, const QVariantList &args);
    ~SmartSelectPlugin() override;

    void injectTools();

private:
    QList<QString> m_toolIds;
};

#endif // SMART_SELECT_PLUGIN_H_

