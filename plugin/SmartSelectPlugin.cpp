#include "SmartSelectPlugin.h"

#include "SmartSelectLog.h"
#include "SmartSelectRuntimeConfig.h"
#include "SmartSelectToolFactory.h"

#include <KoToolManager_p.h>
#include <KoToolRegistry.h>
#include <kpluginfactory.h>

K_PLUGIN_FACTORY_WITH_JSON(
    SmartSelectPluginFactory, "smartselect.json", registerPlugin<SmartSelectPlugin>();)

SmartSelectPlugin::SmartSelectPlugin(QObject *parent, const QVariantList&)
    : QObject(parent)
{
    auto addTool = [this](KoToolFactoryBase *toolFactory) {
        logSmartSelectEvent(SmartSelectLogLevel::Info,
            QStringLiteral("setup.tool_factory.register"),
            { { QStringLiteral("tool.id"), toolFactory->id() } });
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

    for (const QString& id : registry->keys()) {
        if (m_toolIds.contains(id)) {
            logSmartSelectEvent(SmartSelectLogLevel::Info,
                QStringLiteral("setup.tool_action.inject"), { { QStringLiteral("tool.id"), id } });
            p->toolActionList.append(new KoToolAction(registry->value(id)));
        }
    }
}

extern "C" {

Q_DECL_EXPORT void load_smart_select_plugin_with_runtime(
    const char *packageDir, const char *pythonExecutable)
{
    SmartSelectRuntimeConfig::instance().setPackageDir(
        QString::fromUtf8(packageDir ? packageDir : ""));
    SmartSelectRuntimeConfig::instance().setPythonExecutable(
        QString::fromUtf8(pythonExecutable ? pythonExecutable : ""));
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("setup.native_runtime"),
        { { QStringLiteral("package.dir"), SmartSelectRuntimeConfig::instance().packageDir() },
            { QStringLiteral("worker.script.path"),
                SmartSelectRuntimeConfig::instance().workerScriptPath() },
            { QStringLiteral("python.executable.path"),
                SmartSelectRuntimeConfig::instance().pythonExecutable() } });

    SmartSelectPlugin plugin(nullptr, QVariantList());
    plugin.injectTools();
}

Q_DECL_EXPORT void load_smart_select_plugin()
{
    logSmartSelectEvent(SmartSelectLogLevel::Info, QStringLiteral("setup.native_runtime"),
        { { QStringLiteral("setup.mode"), QStringLiteral("fallback_without_runtime_paths") } });
    SmartSelectPlugin plugin(nullptr, QVariantList());
    plugin.injectTools();
}

} // extern "C"

#include "SmartSelectPlugin.moc"
