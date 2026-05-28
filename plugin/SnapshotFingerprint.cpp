#include "SnapshotFingerprint.h"

#include <QCryptographicHash>

namespace
{

QByteArray compactRows(const QImage& image, int bytesPerPixel)
{
    QByteArray bytes;
    bytes.reserve(image.width() * image.height() * bytesPerPixel);
    for (int y = 0; y < image.height(); ++y) {
        bytes.append(
            reinterpret_cast<const char *>(image.constScanLine(y)), image.width() * bytesPerPixel);
    }
    return bytes;
}

} // namespace

QString exactSnapshotFingerprint(
    const QImage& modelRgbImage, const QByteArray& alphaMask, const QSize& fingerprintSize)
{
    if (modelRgbImage.isNull() || !fingerprintSize.isValid()) {
        return QString();
    }
    const int width = modelRgbImage.width();
    const int height = modelRgbImage.height();
    if (width <= 0 || height <= 0 || alphaMask.size() != width * height) {
        return QString();
    }

    const QImage normalizedRgb
        = modelRgbImage.convertToFormat(QImage::Format_RGB888)
              .scaled(fingerprintSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
              .convertToFormat(QImage::Format_RGB888);

    QImage alphaImage(reinterpret_cast<const uchar *>(alphaMask.constData()), width, height, width,
        QImage::Format_Grayscale8);
    const QImage normalizedAlpha
        = alphaImage.copy()
              .scaled(fingerprintSize, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
              .convertToFormat(QImage::Format_Grayscale8);

    const QByteArray rgbBytes = compactRows(normalizedRgb, 3);
    const QByteArray alphaBytes = compactRows(normalizedAlpha, 1);

    QCryptographicHash hash(QCryptographicHash::Sha256);
    const int fingerprintWidth = fingerprintSize.width();
    const int fingerprintHeight = fingerprintSize.height();
    hash.addData(reinterpret_cast<const char *>(&fingerprintWidth), sizeof(int));
    hash.addData(reinterpret_cast<const char *>(&fingerprintHeight), sizeof(int));
    hash.addData(reinterpret_cast<const char *>(&width), sizeof(int));
    hash.addData(reinterpret_cast<const char *>(&height), sizeof(int));
    hash.addData(rgbBytes.constData(), rgbBytes.size());
    hash.addData(alphaBytes.constData(), alphaBytes.size());

    return QString::fromLatin1(hash.result().toHex());
}

QString exactSnapshotFingerprint(const QImage& snapshot, const QSize& fingerprintSize)
{
    if (snapshot.isNull() || !fingerprintSize.isValid()) {
        return QString();
    }

    const QImage rgba = snapshot.convertToFormat(QImage::Format_RGBA8888);
    QImage rgb(rgba.size(), QImage::Format_RGB888);
    QByteArray alphaMask;
    alphaMask.resize(rgba.width() * rgba.height());

    for (int y = 0; y < rgba.height(); ++y) {
        const uchar *src = rgba.constScanLine(y);
        uchar *dst = rgb.scanLine(y);
        uchar *alpha = reinterpret_cast<uchar *>(alphaMask.data()) + y * rgba.width();
        for (int x = 0; x < rgba.width(); ++x) {
            const int srcOffset = x * 4;
            dst[x * 3] = src[srcOffset];
            dst[x * 3 + 1] = src[srcOffset + 1];
            dst[x * 3 + 2] = src[srcOffset + 2];
            alpha[x] = src[srcOffset + 3];
        }
    }

    return exactSnapshotFingerprint(rgb, alphaMask, fingerprintSize);
}
