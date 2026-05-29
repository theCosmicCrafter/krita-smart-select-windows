#ifndef SMART_SELECT_SELECTION_APPLY_H_
#define SMART_SELECT_SELECTION_APPLY_H_

#include "SelectionMask.h"

#include <kis_selection_tool_config_widget_helper.h>
#include <kis_types.h>

class KoCanvasBase;

bool applySelectionMask(KoCanvasBase *canvas, KisImageSP image, KisNodeSP node,
    const SelectionMask& mask, SelectionAction action, int grow, int feather, bool antiAlias);

#endif // SMART_SELECT_SELECTION_APPLY_H_
