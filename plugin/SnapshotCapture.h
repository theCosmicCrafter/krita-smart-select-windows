#ifndef SMART_SELECT_SNAPSHOT_CAPTURE_H_
#define SMART_SELECT_SNAPSHOT_CAPTURE_H_

#include "SnapshotImage.h"

#include <QRect>
#include <QStringList>

#include <kis_types.h>

struct SnapshotInput
{
    QRect bounds;
    QSize canvasSize;
    QImage modelRgbImage;
    QByteArray alphaMask;
    QString fingerprint;
    bool hasVisiblePixels = false;
};

struct SnapshotFiles
{
    QString tempDirPath;
    QString imagePath;
    QString alphaMaskPath;
    QStringList cleanupPaths;
};

bool captureSnapshot(KisImageSP image, SnapshotInput *snapshot, QString *errorMessage = nullptr);
bool writeSnapshotFiles(
    const SnapshotInput& snapshot, SnapshotFiles *files, QString *errorMessage = nullptr);

#endif // SMART_SELECT_SNAPSHOT_CAPTURE_H_
