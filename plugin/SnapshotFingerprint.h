#ifndef SMART_SELECT_SNAPSHOT_FINGERPRINT_H_
#define SMART_SELECT_SNAPSHOT_FINGERPRINT_H_

#include <QImage>
#include <QSize>
#include <QString>

QString exactSnapshotFingerprint(const QImage &snapshot, const QSize &fingerprintSize = QSize(64, 64));

#endif // SMART_SELECT_SNAPSHOT_FINGERPRINT_H_

