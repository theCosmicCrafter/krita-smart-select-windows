#include "SmartSelectTool.h"

#include "SelectionApply.h"

#include <QCursor>
#include <QDebug>
#include <QIcon>
#include <QLabel>
#include <QLayout>
#include <QPainter>
#include <QVBoxLayout>

#include <KisViewManager.h>
#include <KoPointerEvent.h>
#include <canvas/kis_canvas2.h>
#include <kis_canvas_resource_provider.h>
#include <kis_floating_message.h>
#include <kis_image.h>
#include <kis_selection_tool_config_widget_helper.h>
#include <klocalizedstring.h>

SmartSelectTool::SmartSelectTool(KoCanvasBase *canvas)
    : KisToolSelect(canvas, QCursor(Qt::CrossCursor), i18n("Smart Select"))
{
    setObjectName("tool_smart_select");
}

void SmartSelectTool::activate(const QSet<KoShape *> &shapes)
{
    qDebug() << "[SmartSelectTool] activate";
    KisToolSelect::activate(shapes);

    if (KisImage *image = currentImage().data()) {
        connect(image, SIGNAL(sigImageUpdated(QRect)), this, SLOT(updateImage(QRect)));
    }

    prepareFakeSegments();
}

void SmartSelectTool::deactivate()
{
    qDebug() << "[SmartSelectTool] deactivate";

    if (KisImage *image = currentImage().data()) {
        disconnect(image, SIGNAL(sigImageUpdated(QRect)), this, SLOT(updateImage(QRect)));
    }

    const quint16 oldHover = m_hoverSegmentId;
    m_hoverSegmentId = 0;
    requestSegmentUpdate(oldHover);
    m_segmentMap.clear();
    m_state = State::Idle;

    KisToolSelect::deactivate();
}

void SmartSelectTool::updateImage(QRect const &)
{
    prepareFakeSegments();
}

void SmartSelectTool::prepareFakeSegments()
{
    m_state = State::Preparing;
    m_hoverSegmentId = 0;

    if (!currentImage()) {
        m_segmentMap.clear();
        m_state = State::Error;
        return;
    }

    const QRect bounds = currentImage()->bounds();
    m_segmentMap.buildFake(bounds.width(), bounds.height());
    m_state = m_segmentMap.isEmpty() ? State::Error : State::Ready;
    canvas()->updateCanvas(currentImage()->pixelToDocument(bounds));
    qDebug() << "[SmartSelectTool] fake segment map ready" << bounds << "segments" << m_segmentMap.segmentCount();
}

void SmartSelectTool::mouseMoveEvent(KoPointerEvent *event)
{
    if (m_state != State::Ready || m_segmentMap.isEmpty()) {
        KisToolSelect::mouseMoveEvent(event);
        return;
    }

    const QPoint position = convertToImagePixelCoordFloored(event);
    setHoverSegment(m_segmentMap.segmentAt(position));
    KisToolSelect::mouseMoveEvent(event);
}

void SmartSelectTool::beginPrimaryAction(KoPointerEvent *event)
{
    KisToolSelectBase::beginPrimaryAction(event);
    if (isMovingSelection()) {
        return;
    }
    if (!selectionEditable()) {
        showNotEditableMessage();
        event->ignore();
        return;
    }

    if (m_hoverSegmentId == 0 || m_state != State::Ready) {
        event->ignore();
        return;
    }

    beginSelectInteraction();
    applySegmentSelection(canvas(),
                          currentImage(),
                          currentNode(),
                          m_segmentMap,
                          m_hoverSegmentId,
                          selectionAction(),
                          growSelection(),
                          featherSelection(),
                          antiAliasSelection());
}

void SmartSelectTool::endPrimaryAction(KoPointerEvent *event)
{
    if (isMovingSelection()) {
        KisToolSelectBase::endPrimaryAction(event);
        return;
    }

    Q_UNUSED(event);
    endSelectInteraction();
}

void SmartSelectTool::paint(QPainter &painter, const KoViewConverter &converter)
{
    Q_UNUSED(converter);
    if (m_hoverSegmentId == 0 || m_state != State::Ready) {
        return;
    }

    const QRect bbox = m_segmentMap.segmentBounds(m_hoverSegmentId);
    if (bbox.isEmpty()) {
        return;
    }

    const QImage overlay = m_segmentMap.overlayImage(m_hoverSegmentId, QColor(255, 0, 0, 96));
    if (overlay.isNull()) {
        return;
    }

    painter.save();
    painter.drawImage(pixelToView(bbox), overlay);
    painter.restore();
}

QWidget *SmartSelectTool::createOptionWidget()
{
    KisToolSelectBase::createOptionWidget();
    KisSelectionOptions *selectionWidget = selectionOptionWidget();
    selectionWidget->setReferenceSectionVisible(false);

    QWidget *container = selectionWidget;
    QLayout *layout = container->layout();
    if (!layout) {
        layout = new QVBoxLayout(container);
    }
    QLabel *label = new QLabel(i18n("Smart Select mock segments: %1").arg(m_segmentMap.segmentCount()), container);
    layout->addWidget(label);
    return selectionWidget;
}

void SmartSelectTool::resetCursorStyle()
{
    if (selectionAction() == SELECTION_ADD) {
        useCursor(QCursor(Qt::CrossCursor));
    } else if (selectionAction() == SELECTION_SUBTRACT) {
        useCursor(QCursor(Qt::ForbiddenCursor));
    } else if (selectionAction() == SELECTION_INTERSECT) {
        useCursor(QCursor(Qt::PointingHandCursor));
    } else {
        KisToolSelect::resetCursorStyle();
    }
}

void SmartSelectTool::setHoverSegment(quint16 segmentId)
{
    if (m_hoverSegmentId == segmentId) {
        return;
    }

    const quint16 old = m_hoverSegmentId;
    m_hoverSegmentId = segmentId;
    requestSegmentUpdate(old);
    requestSegmentUpdate(m_hoverSegmentId);
}

void SmartSelectTool::requestSegmentUpdate(quint16 segmentId)
{
    if (!currentImage() || segmentId == 0) {
        return;
    }

    QRect bbox = m_segmentMap.segmentBounds(segmentId);
    if (bbox.isEmpty()) {
        return;
    }
    bbox = bbox.adjusted(-2, -2, 2, 2).intersected(currentImage()->bounds());
    canvas()->updateCanvas(currentImage()->pixelToDocument(bbox));
}

void SmartSelectTool::showNotEditableMessage()
{
    KisCanvas2 *kisCanvas = dynamic_cast<KisCanvas2 *>(canvas());
    if (!kisCanvas || !kisCanvas->viewManager()) {
        return;
    }

    kisCanvas->viewManager()->showFloatingMessage(i18n("Selection is not editable"),
                                                  QIcon(),
                                                  2000,
                                                  KisFloatingMessage::Medium,
                                                  Qt::AlignCenter);
}
