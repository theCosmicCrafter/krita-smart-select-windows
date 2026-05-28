#ifndef SMART_SELECT_SELECTION_APPLY_H_
#define SMART_SELECT_SELECTION_APPLY_H_

#include "SegmentMap.h"
#include "SelectionMask.h"

#include <kis_selection_tool_config_widget_helper.h>
#include <kis_types.h>

class KoCanvasBase;

bool applySegmentSelection(KoCanvasBase *canvas, KisImageSP image, KisNodeSP node,
    const SegmentMap& segmentMap, quint16 segmentId, SelectionAction action, int grow, int feather,
    bool antiAlias);

bool applySelectionMask(KoCanvasBase *canvas, KisImageSP image, KisNodeSP node,
    const SelectionMask& mask, SelectionAction action, int grow, int feather, bool antiAlias);

#endif // SMART_SELECT_SELECTION_APPLY_H_
