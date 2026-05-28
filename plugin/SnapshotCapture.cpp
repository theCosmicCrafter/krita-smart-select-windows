#include "SnapshotCapture.h"

#include "NpyWriter.h"
#include "SnapshotFingerprint.h"

#include <QDir>
#include <QTemporaryDir>

#include <kis_image.h>

bool captureSnapshot(KisImageSP image, SnapshotInput *snapshot, QString *errorMessage)
{
    if (!snapshot) {
        return false;
    }
    *snapshot = SnapshotInput();

    if (!image) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("no active image");
        }
        return false;
    }

    const QRect bounds = image->bounds();
    if (bounds.isEmpty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("image bounds are empty");
        }
        return false;
    }

    const QImage projection = image->convertToQImage(bounds, nullptr);
    NormalizedSnapshot normalized;
    if (!normalizeSnapshotImage(projection, &normalized, errorMessage)) {
        return false;
    }

    snapshot->bounds = bounds;
    snapshot->canvasSize = normalized.canvasSize;
    snapshot->modelRgbImage = normalized.modelRgbImage;
    snapshot->alphaMask = normalized.alphaMask;
    snapshot->hasVisiblePixels = normalized.hasVisiblePixels;
    snapshot->fingerprint
        = exactSnapshotFingerprint(snapshot->modelRgbImage, snapshot->alphaMask, QSize(128, 128));
    if (snapshot->fingerprint.isEmpty()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("snapshot fingerprint failed");
        }
        return false;
    }
    return true;
}

bool writeSnapshotFiles(const SnapshotInput& snapshot, SnapshotFiles *files, QString *errorMessage)
{
    if (!files) {
        return false;
    }
    *files = SnapshotFiles();

    if (snapshot.modelRgbImage.isNull() || snapshot.alphaMask.isEmpty()
        || !snapshot.canvasSize.isValid()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("snapshot data is empty");
        }
        return false;
    }

    QTemporaryDir tempDir(QDir::tempPath() + QStringLiteral("/krita-smart-select-input-XXXXXX"));
    if (!tempDir.isValid()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("could not create snapshot temp dir");
        }
        return false;
    }

    const QString imagePath = tempDir.filePath(QStringLiteral("snapshot.png"));
    const QString alphaMaskPath = tempDir.filePath(QStringLiteral("alpha_mask.npy"));

    if (!snapshot.modelRgbImage.save(imagePath, "PNG")) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("could not write snapshot image");
        }
        return false;
    }

    if (!writeUint8Npy(alphaMaskPath, snapshot.canvasSize, snapshot.alphaMask, errorMessage)) {
        return false;
    }

    tempDir.setAutoRemove(false);
    files->tempDirPath = tempDir.path();
    files->imagePath = imagePath;
    files->alphaMaskPath = alphaMaskPath;
    files->cleanupPaths = QStringList { imagePath, alphaMaskPath };
    return true;
}
