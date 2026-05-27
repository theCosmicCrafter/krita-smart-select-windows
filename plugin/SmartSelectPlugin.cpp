#include "SmartSelectPlugin.h"

#include "SmartSelectRuntimeConfig.h"
#include "SmartSelectToolFactory.h"

#include <KoToolManager_p.h>
#include <KoToolRegistry.h>
#include <QDebug>
#include <kpluginfactory.h>

K_PLUGIN_FACTORY_WITH_JSON(SmartSelectPluginFactory, "smartselect.json", registerPlugin<SmartSelectPlugin>();)

SmartSelectPlugin::SmartSelectPlugin(QObject *parent, const QVariantList &)
    : QObject(parent)
{
    auto addTool = [this](KoToolFactoryBase *toolFactory) {
        qDebug() << "[SmartSelectPlugin] Registering tool factory:" << toolFactory->id();
        KoToolRegistry::instance()->add(toolFactory);
        m_toolIds.append(toolFactory->id());
    };

    addTool(new SmartSelectToolFactory());
}

SmartSelectPlugin::~SmartSelectPlugin() = default;

void SmartSelectPlugin::injectTools()
{
    KoToolRegistry *registry = KoToolRegistry::instance();
    KoToolManager::Private *p = KoToolManager::instance()->priv();

    for (const QString &id : registry->keys()) {
        if (m_toolIds.contains(id)) {
            qDebug() << "[SmartSelectPlugin] Injecting tool action for" << id;
            p->toolActionList.append(new KoToolAction(registry->value(id)));
        }
    }
}

extern "C" {

Q_DECL_EXPORT void load_smart_select_plugin_with_runtime(const char *packageDir, const char *pythonExecutable)
{
    qDebug() << "[SmartSelectPlugin] Loading Smart Select plugin via Python loader with runtime paths.";
    SmartSelectRuntimeConfig::instance().setPackageDir(QString::fromUtf8(packageDir ? packageDir : ""));
    SmartSelectRuntimeConfig::instance().setPythonExecutable(QString::fromUtf8(pythonExecutable ? pythonExecutable : ""));

    SmartSelectPlugin plugin(nullptr, QVariantList());
    plugin.injectTools();
}

Q_DECL_EXPORT void load_smart_select_plugin()
{
    qDebug() << "[SmartSelectPlugin] Loading Smart Select plugin via Python loader.";
    SmartSelectPlugin plugin(nullptr, QVariantList());
    plugin.injectTools();
}

} // extern "C"

#include "SmartSelectPlugin.moc"
