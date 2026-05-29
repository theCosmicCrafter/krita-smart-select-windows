#include "../LassoRegionMask.h"
#include "../NpyWriter.h"
#include "../SelectionMask.h"
#include "../SmartSelectLog.h"
#include "../SnapshotFingerprint.h"
#include "../SnapshotImage.h"
#include "../WorkerClient.h"

#include <QColor>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QTemporaryFile>
#include <QTest>
#include <QVector>

namespace
{

QString writeTempFile(QTemporaryDir& dir, const QByteArray& bytes)
{
    QTemporaryFile file(dir.filePath(QStringLiteral("selection_mask_XXXXXX.npy")));
    file.setAutoRemove(false);
    if (!file.open()) {
        qFatal("Could not create temporary .npy fixture: %s", qPrintable(file.errorString()));
    }
    file.write(bytes);
    const QString path = file.fileName();
    file.close();
    return path;
}

QByteArray makeUint8Npy(const QByteArray& descr, int height, int width, const QByteArray& payload,
    bool fortranOrder = false)
{
    QByteArray header = "{'descr': '" + descr
        + "', 'fortran_order': " + (fortranOrder ? "True" : "False") + ", 'shape': ("
        + QByteArray::number(height) + ", " + QByteArray::number(width) + "), }";
    const int padding = 16 - ((10 + header.size() + 1) % 16);
    header.append(QByteArray(padding, ' '));
    header.append('\n');

    QByteArray bytes = "\x93NUMPY";
    bytes.append(char(1));
    bytes.append(char(0));
    bytes.append(char(header.size() & 0xff));
    bytes.append(char((header.size() >> 8) & 0xff));
    bytes.append(header);
    bytes.append(payload);
    return bytes;
}

} // namespace

class SmartSelectCoreTest : public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void exactFingerprintIsStableAndContentSensitive();
    void fingerprintIncludesAlphaMask();
    void fingerprintRejectsInvalidAlphaMask();
    void snapshotNormalizationCompositesOverWhiteAndKeepsAlpha();
    void snapshotNormalizationDetectsFullyTransparentImage();
    void writeUint8NpyWritesValidHeader();
    void writeUint8NpyRejectsInvalidPayloadSize();
    void workerPrepareSelectionRequestIncludesRegionFields();
    void selectionMaskLoadNpyAcceptsUint8Mask();
    void lassoRegionMaskRasterizesBounds();
    void smartSelectLogFormatterUsesScalarStrings();
};

void SmartSelectCoreTest::exactFingerprintIsStableAndContentSensitive()
{
    QImage first(4, 4, QImage::Format_RGBA8888);
    first.fill(QColor(0, 0, 0, 255));

    QImage same = first;
    QImage changed = first;
    changed.setPixelColor(0, 0, QColor(255, 0, 0, 255));

    QImage differentSize(5, 4, QImage::Format_RGBA8888);
    differentSize.fill(QColor(0, 0, 0, 255));

    const QString firstHash = exactSnapshotFingerprint(first);
    QVERIFY(!firstHash.isEmpty());
    QCOMPARE(exactSnapshotFingerprint(same), firstHash);
    QVERIFY(exactSnapshotFingerprint(changed) != firstHash);
    QVERIFY(exactSnapshotFingerprint(differentSize) != firstHash);
    QVERIFY(exactSnapshotFingerprint(QImage()).isEmpty());
}

void SmartSelectCoreTest::fingerprintIncludesAlphaMask()
{
    QImage rgb(4, 4, QImage::Format_RGB888);
    rgb.fill(QColor(10, 20, 30));
    QByteArray alpha(16, char(255));
    QByteArray changedAlpha = alpha;
    changedAlpha[0] = char(0);

    const QString firstHash = exactSnapshotFingerprint(rgb, alpha);
    QVERIFY(!firstHash.isEmpty());
    QCOMPARE(exactSnapshotFingerprint(rgb, alpha), firstHash);
    QVERIFY(exactSnapshotFingerprint(rgb, changedAlpha) != firstHash);
}

void SmartSelectCoreTest::fingerprintRejectsInvalidAlphaMask()
{
    QImage rgb(4, 4, QImage::Format_RGB888);
    rgb.fill(Qt::black);

    QVERIFY(exactSnapshotFingerprint(rgb, QByteArray(15, char(255))).isEmpty());
}

void SmartSelectCoreTest::snapshotNormalizationCompositesOverWhiteAndKeepsAlpha()
{
    QImage source(2, 1, QImage::Format_RGBA8888);
    source.setPixelColor(0, 0, QColor(0, 0, 0, 0));
    source.setPixelColor(1, 0, QColor(0, 0, 0, 128));

    NormalizedSnapshot snapshot;
    QString error;
    QVERIFY2(normalizeSnapshotImage(source, &snapshot, &error), qPrintable(error));

    QCOMPARE(snapshot.canvasSize, QSize(2, 1));
    QCOMPARE(snapshot.alphaMask.size(), 2);
    QCOMPARE(uchar(snapshot.alphaMask[0]), uchar(0));
    QCOMPARE(uchar(snapshot.alphaMask[1]), uchar(128));
    QVERIFY(snapshot.hasVisiblePixels);
    QCOMPARE(snapshot.modelRgbImage.pixelColor(0, 0), QColor(255, 255, 255));
    QCOMPARE(snapshot.modelRgbImage.pixelColor(1, 0), QColor(127, 127, 127));
}

void SmartSelectCoreTest::snapshotNormalizationDetectsFullyTransparentImage()
{
    QImage source(2, 2, QImage::Format_RGBA8888);
    source.fill(QColor(255, 0, 0, 0));

    NormalizedSnapshot snapshot;
    QString error;
    QVERIFY2(normalizeSnapshotImage(source, &snapshot, &error), qPrintable(error));

    QVERIFY(!snapshot.hasVisiblePixels);
    QCOMPARE(snapshot.alphaMask, QByteArray(4, char(0)));
}

void SmartSelectCoreTest::writeUint8NpyWritesValidHeader()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = dir.filePath(QStringLiteral("alpha.npy"));

    QString error;
    QVERIFY2(writeUint8Npy(path, QSize(3, 2), QByteArray("\x00\x01\x02\x03\x04\x05", 6), &error),
        qPrintable(error));

    QFile file(path);
    QVERIFY(file.open(QIODevice::ReadOnly));
    const QByteArray bytes = file.readAll();
    QVERIFY(bytes.startsWith("\x93NUMPY"));
    QVERIFY(bytes.contains("'descr': '|u1'"));
    QVERIFY(bytes.contains("'shape': (2, 3)"));
    QVERIFY(bytes.endsWith(QByteArray("\x00\x01\x02\x03\x04\x05", 6)));
}

void SmartSelectCoreTest::writeUint8NpyRejectsInvalidPayloadSize()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QString error;
    QVERIFY(!writeUint8Npy(
        dir.filePath(QStringLiteral("bad.npy")), QSize(3, 2), QByteArray("\x00\x01", 2), &error));
    QCOMPARE(error, QStringLiteral("Invalid uint8 .npy payload size"));
}

void SmartSelectCoreTest::workerPrepareSelectionRequestIncludesRegionFields()
{
    PrepareSelectionRequest request;
    request.imagePath = QStringLiteral("/tmp/input.png");
    request.alphaMaskPath = QStringLiteral("/tmp/alpha.npy");
    request.regionMaskPath = QStringLiteral("/tmp/region.npy");
    request.regionBbox = QRect(1, 2, 3, 4);
    request.modelId = QStringLiteral("birefnet_hr_matting");
    request.algorithmVersion = QStringLiteral("birefnet-hr-matting-lasso-v1");
    request.fingerprint = QStringLiteral("fingerprint");
    request.modelCacheDir = QStringLiteral("/tmp/models");
    request.runtimeCacheDir = QStringLiteral("/tmp/runtime");
    request.canvasSize = QSize(64, 48);

    const QJsonObject message = prepareSelectionRequestJson(13, request);
    QCOMPARE(message.value(QStringLiteral("id")).toInt(), 13);
    QCOMPARE(
        message.value(QStringLiteral("method")).toString(), QStringLiteral("prepare_selection"));

    const QJsonObject params = message.value(QStringLiteral("params")).toObject();
    QCOMPARE(params.value(QStringLiteral("region_mask_path")).toString(), request.regionMaskPath);
    const QJsonArray bbox = params.value(QStringLiteral("region_bbox")).toArray();
    QCOMPARE(bbox.size(), 4);
    QCOMPARE(bbox.at(0).toInt(), 1);
    QCOMPARE(bbox.at(1).toInt(), 2);
    QCOMPARE(bbox.at(2).toInt(), 3);
    QCOMPARE(bbox.at(3).toInt(), 4);
    QCOMPARE(
        params.value(QStringLiteral("algorithm_version")).toString(), request.algorithmVersion);
    QCOMPARE(params.value(QStringLiteral("model_id")).toString(), request.modelId);
    QCOMPARE(params.value(QStringLiteral("model_cache_dir")).toString(), request.modelCacheDir);
    QCOMPARE(params.value(QStringLiteral("runtime_cache_dir")).toString(), request.runtimeCacheDir);
    QVERIFY(!params.contains(QStringLiteral("refiner_id")));
    QVERIFY(!params.contains(QStringLiteral("refiner_cache_dir")));
    QVERIFY(!params.contains(QStringLiteral("segments_path")));
}

void SmartSelectCoreTest::selectionMaskLoadNpyAcceptsUint8Mask()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QByteArray payload("\x00\x80\x00\x00\x40\xff", 6);
    const QString path = writeTempFile(dir, makeUint8Npy("|u1", 2, 3, payload));

    SelectionMask mask;
    QString error;
    QVERIFY2(mask.loadNpy(path, &error), qPrintable(error));

    QCOMPARE(mask.size(), QSize(3, 2));
    QCOMPARE(mask.selectedBounds(), QRect(1, 0, 2, 2));
    QCOMPARE(mask.bytesForBounds(mask.selectedBounds()), QByteArray("\x80\x00\x40\xff", 4));
}

void SmartSelectCoreTest::lassoRegionMaskRasterizesBounds()
{
    QVector<QPointF> points {
        QPointF(1, 1),
        QPointF(4, 1),
        QPointF(4, 4),
        QPointF(1, 4),
    };

    LassoRegionMask mask;
    QString error;
    QVERIFY2(buildLassoRegionMask(points, QSize(8, 8), &mask, &error), qPrintable(error));

    QVERIFY(mask.area > 0);
    QVERIFY(mask.bbox.intersects(QRect(1, 1, 4, 4)));
    QCOMPARE(mask.canvasSize, QSize(8, 8));
    QCOMPARE(mask.bytes.size(), 64);
}

void SmartSelectCoreTest::smartSelectLogFormatterUsesScalarStrings()
{
    const QString line = formatSmartSelectLogEvent(SmartSelectLogLevel::Info,
        QStringLiteral("selection.start"),
        { { QStringLiteral("model.id"), QStringLiteral("birefnet_hr_matting") },
            { QStringLiteral("region.bbox"), QRect(1, 2, 3, 4) },
            { QStringLiteral("environment.removed_keys"),
                QStringList { QStringLiteral("PYTHONHOME"), QStringLiteral("PYTHONPATH") } } });

    QVERIFY(line.contains(QStringLiteral("event.name=selection.start")));
    QVERIFY(!line.contains(QStringLiteral("event.name=[\"selection.start\"]")));
    QVERIFY(line.contains(QStringLiteral("severity=info")));
    QVERIFY(line.contains(QStringLiteral("model.id=birefnet_hr_matting")));
    QVERIFY(!line.contains(QStringLiteral("model.id=[\"birefnet_hr_matting\"]")));
    QVERIFY(line.contains(QStringLiteral("region.bbox=[1,2,3,4]")));
    QVERIFY(line.contains(QStringLiteral("environment.removed_keys=[\"PYTHONHOME\",\"PYTHONPATH\"]")));
}

QTEST_GUILESS_MAIN(SmartSelectCoreTest)

#include "SmartSelectCoreTest.moc"
