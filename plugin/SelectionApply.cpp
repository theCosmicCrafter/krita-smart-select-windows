#include "SelectionApply.h"

#include <KisCursorOverrideLock.h>
#include <KoCanvasBase.h>
#include <canvas/kis_canvas2.h>
#include <kis_command_utils.h>
#include <kis_cursor.h>
#include <kis_default_bounds.h>
#include <kis_paint_device.h>
#include <kis_pixel_selection.h>
#include <kis_processing_applicator.h>
#include <kis_selection_filters.h>
#include <kis_selection_tool_helper.h>
#include <kundo2magicstring.h>

namespace
{

void adjustSelection(KisPixelSelectionSP const& selection, int grow, int feather, bool antiAlias)
{
    if (grow > 0) {
        KisGrowSelectionFilter filter(grow, grow);
        filter.process(selection, selection->selectedRect().adjusted(-grow, -grow, grow, grow));
    } else if (grow < 0) {
        KisShrinkSelectionFilter filter(-grow, -grow, false);
        filter.process(selection, selection->selectedRect());
    }

    if (feather > 0) {
        KisFeatherSelectionFilter filter(feather);
        filter.process(
            selection, selection->selectedRect().adjusted(-feather, -feather, feather, feather));
    } else if (antiAlias) {
        KisAntiAliasSelectionFilter filter;
        filter.process(selection, selection->selectedRect());
    }
}

} // namespace

bool applySegmentSelection(KoCanvasBase *canvas, KisImageSP image, KisNodeSP node,
    const SegmentMap& segmentMap, quint16 segmentId, SelectionAction action, int grow, int feather,
    bool antiAlias)
{
    KisCanvas2 *kisCanvas = dynamic_cast<KisCanvas2 *>(canvas);
    if (!kisCanvas || !image || segmentId == 0) {
        return false;
    }

    QRect maskBounds;
    const QByteArray mask = segmentMap.selectionMask(segmentId, &maskBounds);
    if (mask.isEmpty() || maskBounds.isEmpty()) {
        return false;
    }

    KisPaintDeviceSP defaultDevice = image->projection();
    KisPixelSelectionSP selection
        = new KisPixelSelection(new KisSelectionDefaultBounds(defaultDevice));
    selection->writeBytes(reinterpret_cast<const quint8 *>(mask.constData()), maskBounds);
    adjustSelection(selection, grow, feather, antiAlias);
    selection->invalidateOutlineCache();

    KisCursorOverrideLock cursorLock(KisCursor::waitCursor());
    KisProcessingApplicator applicator(image, node, KisProcessingApplicator::NO_IMAGE_UPDATES,
        KisImageSignalVector(), kundo2_i18n("Smart Select"));

    KisSelectionToolHelper helper(kisCanvas, kundo2_i18n("Smart Select"));
    helper.selectPixelSelection(applicator, selection, action);
    applicator.end();
    return true;
}

bool applySelectionMask(KoCanvasBase *canvas, KisImageSP image, KisNodeSP node,
    const SelectionMask& mask, SelectionAction action, int grow, int feather, bool antiAlias)
{
    KisCanvas2 *kisCanvas = dynamic_cast<KisCanvas2 *>(canvas);
    if (!kisCanvas || !image || mask.isEmpty()) {
        return false;
    }

    const QRect maskBounds = mask.selectedBounds();
    const QByteArray bytes = mask.bytesForBounds(maskBounds);
    if (maskBounds.isEmpty() || bytes.isEmpty()) {
        return false;
    }

    KisPaintDeviceSP defaultDevice = image->projection();
    KisPixelSelectionSP selection
        = new KisPixelSelection(new KisSelectionDefaultBounds(defaultDevice));
    selection->writeBytes(reinterpret_cast<const quint8 *>(bytes.constData()), maskBounds);
    adjustSelection(selection, grow, feather, antiAlias);
    selection->invalidateOutlineCache();

    KisCursorOverrideLock cursorLock(KisCursor::waitCursor());
    KisProcessingApplicator applicator(image, node, KisProcessingApplicator::NO_IMAGE_UPDATES,
        KisImageSignalVector(), kundo2_i18n("Smart Select"));

    KisSelectionToolHelper helper(kisCanvas, kundo2_i18n("Smart Select"));
    helper.selectPixelSelection(applicator, selection, action);
    applicator.end();
    return true;
}
