#ifndef SMART_SELECT_TOOL_FACTORY_H_
#define SMART_SELECT_TOOL_FACTORY_H_

#include "SmartSelectTool.h"

#include <KisSelectionToolFactoryBase.h>
#include <kis_icon.h>
#include <klocalizedstring.h>

class SmartSelectToolFactory : public KisSelectionToolFactoryBase
{
public:
    SmartSelectToolFactory()
        : KisSelectionToolFactoryBase("SmartSelectTool")
    {
        setToolTip(i18n("Smart Select"));
        setSection(ToolBoxSection::Select);
        setIconName(koIconNameCStr("tool_smart_select"));
        setPriority(10);
        setActivationShapeId(KRITA_TOOL_ACTIVATION_ID);
    }

    KoToolBase *createTool(KoCanvasBase *canvas) override
    {
        return new SmartSelectTool(canvas);
    }
};

#endif // SMART_SELECT_TOOL_FACTORY_H_
