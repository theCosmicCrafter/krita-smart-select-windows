#include "SnapshotFingerprint.h"

#include <QCryptographicHash>

QString exactSnapshotFingerprint(const QImage &snapshot, const QSize &fingerprintSize)
{
    if (snapshot.isNull() || !fingerprintSize.isValid()) {
        return QString();
    }

    const QImage normalized = snapshot
                                  .convertToFormat(QImage::Format_RGBA8888)
                                  .scaled(fingerprintSize,
                                          Qt::IgnoreAspectRatio,
                                          Qt::SmoothTransformation)
                                  .convertToFormat(QImage::Format_RGBA8888);

    QCryptographicHash hash(QCryptographicHash::Sha256);
    const int fingerprintWidth = fingerprintSize.width();
    const int fingerprintHeight = fingerprintSize.height();
    const int snapshotWidth = snapshot.width();
    const int snapshotHeight = snapshot.height();
    hash.addData(reinterpret_cast<const char *>(&fingerprintWidth), sizeof(int));
    hash.addData(reinterpret_cast<const char *>(&fingerprintHeight), sizeof(int));
    hash.addData(reinterpret_cast<const char *>(&snapshotWidth), sizeof(int));
    hash.addData(reinterpret_cast<const char *>(&snapshotHeight), sizeof(int));
    for (int y = 0; y < normalized.height(); ++y) {
        hash.addData(reinterpret_cast<const char *>(normalized.constScanLine(y)),
                     normalized.bytesPerLine());
    }

    return QString::fromLatin1(hash.result().toHex());
}
