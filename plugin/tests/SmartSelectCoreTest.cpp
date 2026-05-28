#include "../NpyWriter.h"
#include "../SegmentMap.h"
#include "../SnapshotFingerprint.h"
#include "../SnapshotImage.h"
#include "../WorkerClient.h"

#include <QColor>
#include <QFile>
#include <QImage>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QTemporaryFile>
#include <QTest>
#include <QVector>

namespace
{

QByteArray makeNpy(const QByteArray& descr, int height, int width, const QVector<quint16>& values,
    bool fortranOrder = false)
{
    QByteArray payload;
    payload.reserve(values.size() * int(sizeof(quint16)));
    for (const quint16 value : values) {
        payload.append(char(value & 0xff));
        payload.append(char((value >> 8) & 0xff));
    }

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

QString writeTempFile(QTemporaryDir& dir, const QByteArray& bytes)
{
    QTemporaryFile file(dir.filePath(QStringLiteral("segment_map_XXXXXX.npy")));
    file.setAutoRemove(false);
    if (!file.open()) {
        qFatal("Could not create temporary .npy fixture: %s", qPrintable(file.errorString()));
    }
    file.write(bytes);
    const QString path = file.fileName();
    file.close();
    return path;
}

QVector<quint16> samplePixels()
{
    return QVector<quint16> {
        0,
        1,
        1,
        0,
        0,
        2,
        0,
        0,
        3,
        3,
        0,
        0,
    };
}

} // namespace

class SmartSelectCoreTest : public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void loadNpyAcceptsUint16SegmentMap();
    void loadNpyRejectsWrongDtype();
    void loadNpyRejectsFortranOrder();
    void loadNpyRejectsTruncatedPayload();
    void segmentSelectionMaskUsesTightBounds();
    void exactFingerprintIsStableAndContentSensitive();
    void fingerprintIncludesAlphaMask();
    void fingerprintRejectsInvalidAlphaMask();
    void snapshotNormalizationCompositesOverWhiteAndKeepsAlpha();
    void snapshotNormalizationDetectsFullyTransparentImage();
    void writeUint8NpyWritesValidHeader();
    void writeUint8NpyRejectsInvalidPayloadSize();
    void workerPrepareRequestIncludesModelCacheDir();
};

void SmartSelectCoreTest::loadNpyAcceptsUint16SegmentMap()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = writeTempFile(dir, makeNpy("<u2", 3, 4, samplePixels()));

    SegmentMap map;
    QString error;
    QVERIFY2(map.loadNpy(path, &error), qPrintable(error));

    QCOMPARE(map.width(), 4);
    QCOMPARE(map.height(), 3);
    QCOMPARE(map.bounds(), QRect(0, 0, 4, 3));
    QCOMPARE(map.segmentCount(), 3);
    QCOMPARE(map.segmentAt(QPoint(1, 0)), quint16(1));
    QCOMPARE(map.segmentAt(QPoint(1, 1)), quint16(2));
    QCOMPARE(map.segmentAt(QPoint(0, 2)), quint16(3));
    QCOMPARE(map.segmentAt(QPoint(-1, 0)), quint16(0));
    QCOMPARE(map.segmentAt(QPoint(4, 0)), quint16(0));
}

void SmartSelectCoreTest::loadNpyRejectsWrongDtype()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = writeTempFile(dir, makeNpy("<u4", 3, 4, samplePixels()));

    SegmentMap map;
    QString error;
    QVERIFY(!map.loadNpy(path, &error));
    QCOMPARE(error, QStringLiteral("Expected uint16 segment map"));
}

void SmartSelectCoreTest::loadNpyRejectsFortranOrder()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = writeTempFile(dir, makeNpy("<u2", 3, 4, samplePixels(), true));

    SegmentMap map;
    QString error;
    QVERIFY(!map.loadNpy(path, &error));
    QCOMPARE(error, QStringLiteral("Fortran-order .npy arrays are not supported"));
}

void SmartSelectCoreTest::loadNpyRejectsTruncatedPayload()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    QByteArray bytes = makeNpy("<u2", 3, 4, samplePixels());
    bytes.chop(1);
    const QString path = writeTempFile(dir, bytes);

    SegmentMap map;
    QString error;
    QVERIFY(!map.loadNpy(path, &error));
    QCOMPARE(error, QStringLiteral("Truncated .npy segment payload"));
}

void SmartSelectCoreTest::segmentSelectionMaskUsesTightBounds()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = writeTempFile(dir, makeNpy("<u2", 3, 4, samplePixels()));

    SegmentMap map;
    QString error;
    QVERIFY2(map.loadNpy(path, &error), qPrintable(error));

    QCOMPARE(map.segmentBounds(1), QRect(1, 0, 2, 1));
    QCOMPARE(map.segmentBounds(2), QRect(1, 1, 1, 1));
    QCOMPARE(map.segmentBounds(3), QRect(0, 2, 2, 1));
    QCOMPARE(map.segmentBounds(99), QRect());

    QRect maskBounds;
    const QByteArray mask = map.selectionMask(1, &maskBounds);
    QCOMPARE(maskBounds, QRect(1, 0, 2, 1));
    QCOMPARE(mask, QByteArray(2, char(255)));

    const QByteArray emptyMask = map.selectionMask(99, &maskBounds);
    QCOMPARE(maskBounds, QRect());
    QVERIFY(emptyMask.isEmpty());
}

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

void SmartSelectCoreTest::workerPrepareRequestIncludesModelCacheDir()
{
    PrepareSegmentsRequest request;
    request.imagePath = QStringLiteral("/tmp/input.png");
    request.alphaMaskPath = QStringLiteral("/tmp/alpha.npy");
    request.modelId = QStringLiteral("sam2.1_hiera_large");
    request.algorithmVersion = QStringLiteral("mock-segments-model-readiness-v1");
    request.fingerprint = QStringLiteral("fingerprint");
    request.modelCacheDir = QStringLiteral("/tmp/models");
    request.canvasSize = QSize(64, 48);

    const QJsonObject message = prepareSegmentsRequestJson(12, request);
    QCOMPARE(message.value(QStringLiteral("id")).toInt(), 12);
    QCOMPARE(
        message.value(QStringLiteral("method")).toString(), QStringLiteral("prepare_segments"));

    const QJsonObject params = message.value(QStringLiteral("params")).toObject();
    QCOMPARE(params.value(QStringLiteral("model_id")).toString(), request.modelId);
    QCOMPARE(
        params.value(QStringLiteral("algorithm_version")).toString(), request.algorithmVersion);
    QCOMPARE(params.value(QStringLiteral("model_cache_dir")).toString(), request.modelCacheDir);
    QCOMPARE(params.value(QStringLiteral("canvas_width")).toInt(), 64);
    QCOMPARE(params.value(QStringLiteral("canvas_height")).toInt(), 48);
}

QTEST_GUILESS_MAIN(SmartSelectCoreTest)

#include "SmartSelectCoreTest.moc"
