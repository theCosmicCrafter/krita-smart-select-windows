#ifndef SMART_SELECT_SNAPSHOT_IMAGE_H_
#define SMART_SELECT_SNAPSHOT_IMAGE_H_

#include <QByteArray>
#include <QImage>
#include <QSize>
#include <QString>

struct NormalizedSnapshot
{
    QSize canvasSize;
    QImage modelRgbImage;
    QByteArray alphaMask;
    bool hasVisiblePixels = false;
};

bool normalizeSnapshotImage(
    const QImage& sourceRgba, NormalizedSnapshot *snapshot, QString *errorMessage = nullptr);

#endif // SMART_SELECT_SNAPSHOT_IMAGE_H_
