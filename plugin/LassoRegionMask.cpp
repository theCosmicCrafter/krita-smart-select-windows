#include "LassoRegionMask.h"

#include "NpyWriter.h"

#include <QImage>
#include <QPainter>
#include <QPainterPath>

bool buildLassoRegionMask(const QVector<QPointF>& points, const QSize& canvasSize,
    LassoRegionMask *mask, QString *errorMessage)
{
    if (!mask) {
        return false;
    }
    *mask = LassoRegionMask();

    if (points.size() < 3) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("lasso region is too small");
        }
        return false;
    }
    if (!canvasSize.isValid() || canvasSize.width() <= 0 || canvasSize.height() <= 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("canvas size is invalid");
        }
        return false;
    }

    QPainterPath path;
    path.moveTo(points.first());
    for (int i = 1; i < points.size(); ++i) {
        path.lineTo(points[i]);
    }
    path.closeSubpath();

    QImage image(canvasSize, QImage::Format_Grayscale8);
    image.fill(0);

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.setPen(Qt::NoPen);
    painter.setBrush(Qt::white);
    painter.drawPath(path);
    painter.end();

    QByteArray bytes;
    bytes.resize(canvasSize.width() * canvasSize.height());

    QRect bbox;
    bool seen = false;
    int area = 0;
    for (int y = 0; y < canvasSize.height(); ++y) {
        const uchar *source = image.constScanLine(y);
        for (int x = 0; x < canvasSize.width(); ++x) {
            const uchar value = source[x] > 0 ? 255 : 0;
            bytes[y * canvasSize.width() + x] = char(value);
            if (value == 0) {
                continue;
            }
            const QRect pixelRect(QPoint(x, y), QSize(1, 1));
            bbox = seen ? bbox.united(pixelRect) : pixelRect;
            seen = true;
            ++area;
        }
    }

    if (bbox.isEmpty() || area == 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("lasso region is empty");
        }
        return false;
    }

    mask->canvasSize = canvasSize;
    mask->bbox = bbox;
    mask->area = area;
    mask->bytes = bytes;
    return true;
}

bool writeLassoRegionMaskNpy(
    const QString& path, const LassoRegionMask& mask, QString *errorMessage)
{
    return writeUint8Npy(path, mask.canvasSize, mask.bytes, errorMessage);
}
