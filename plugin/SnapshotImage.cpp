#include "SnapshotImage.h"

#include <QColor>

bool normalizeSnapshotImage(
    const QImage& sourceRgba, NormalizedSnapshot *snapshot, QString *errorMessage)
{
    if (!snapshot) {
        return false;
    }
    *snapshot = NormalizedSnapshot();

    if (sourceRgba.isNull() || sourceRgba.width() <= 0 || sourceRgba.height() <= 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("snapshot image is empty");
        }
        return false;
    }

    const QImage rgba = sourceRgba.convertToFormat(QImage::Format_RGBA8888);
    QImage rgb(rgba.size(), QImage::Format_RGB888);
    QByteArray alphaMask;
    alphaMask.resize(rgba.width() * rgba.height());
    bool hasVisiblePixels = false;

    for (int y = 0; y < rgba.height(); ++y) {
        const uchar *src = rgba.constScanLine(y);
        uchar *dst = rgb.scanLine(y);
        uchar *alpha = reinterpret_cast<uchar *>(alphaMask.data()) + y * rgba.width();
        for (int x = 0; x < rgba.width(); ++x) {
            const int offset = x * 4;
            const int r = src[offset];
            const int g = src[offset + 1];
            const int b = src[offset + 2];
            const int a = src[offset + 3];
            alpha[x] = uchar(a);
            hasVisiblePixels = hasVisiblePixels || a > 0;

            const int inverseAlpha = 255 - a;
            dst[x * 3] = uchar((r * a + 255 * inverseAlpha + 127) / 255);
            dst[x * 3 + 1] = uchar((g * a + 255 * inverseAlpha + 127) / 255);
            dst[x * 3 + 2] = uchar((b * a + 255 * inverseAlpha + 127) / 255);
        }
    }

    snapshot->canvasSize = rgba.size();
    snapshot->modelRgbImage = rgb;
    snapshot->alphaMask = alphaMask;
    snapshot->hasVisiblePixels = hasVisiblePixels;
    return true;
}
