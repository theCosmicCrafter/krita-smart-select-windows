#ifndef SMART_SELECT_TOOL_H_
#define SMART_SELECT_TOOL_H_

#include "SegmentMap.h"

#include <kis_tool_select_base.h>

class SmartSelectTool : public KisToolSelect
{
    Q_OBJECT

public:
    explicit SmartSelectTool(KoCanvasBase *canvas);

    QWidget *createOptionWidget() override;
    void paint(QPainter &painter, const KoViewConverter &converter) override;

    void mouseMoveEvent(KoPointerEvent *event) override;
    void beginPrimaryAction(KoPointerEvent *event) override;
    void endPrimaryAction(KoPointerEvent *event) override;
    void resetCursorStyle() override;

protected:
    bool wantsAutoScroll() const override
    {
        return false;
    }

    bool isPixelOnly() const override
    {
        return true;
    }

    bool usesColorLabels() const override
    {
        return true;
    }

public Q_SLOTS:
    void activate(const QSet<KoShape *> &shapes) override;
    void deactivate() override;
    void updateImage(QRect const &rect);

protected:
    using KisToolSelectBase::m_widgetHelper;

private:
    enum class State {
        Idle,
        Preparing,
        Ready,
        Error,
        Cancelled
    };

    void prepareFakeSegments();
    void setHoverSegment(quint16 segmentId);
    void requestSegmentUpdate(quint16 segmentId);
    void showNotEditableMessage();

    SegmentMap m_segmentMap;
    quint16 m_hoverSegmentId = 0;
    State m_state = State::Idle;
};

#endif // SMART_SELECT_TOOL_H_

