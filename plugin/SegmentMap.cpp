#include "SegmentMap.h"

#include <QFile>
#include <QRegularExpression>
#include <QSize>
#include <QtGlobal>

#include <algorithm>
#include <cstring>

void SegmentMap::clear()
{
    m_width = 0;
    m_height = 0;
    m_pixels.clear();
    m_segments.clear();
}

bool SegmentMap::isEmpty() const
{
    return m_width <= 0 || m_height <= 0 || m_pixels.isEmpty();
}

int SegmentMap::width() const
{
    return m_width;
}

int SegmentMap::height() const
{
    return m_height;
}

QRect SegmentMap::bounds() const
{
    return QRect(0, 0, m_width, m_height);
}

int SegmentMap::segmentCount() const
{
    return m_segments.size();
}

void SegmentMap::buildFake(int width, int height)
{
    clear();
    if (width <= 0 || height <= 0) {
        return;
    }

    m_width = width;
    m_height = height;
    m_pixels.resize(width * height);
    std::fill(m_pixels.begin(), m_pixels.end(), 0);

    const QRect topLeft(qMax(0, width / 12),
                        qMax(0, height / 12),
                        qMax(8, width / 4),
                        qMax(8, height / 4));
    const QRect bottomRight(qMax(0, width - width / 3),
                            qMax(0, height - height / 3),
                            qMax(8, width / 4),
                            qMax(8, height / 4));

    for (int y = topLeft.top(); y <= topLeft.bottom() && y < height; ++y) {
        for (int x = topLeft.left(); x <= topLeft.right() && x < width; ++x) {
            putSegmentPixel(x, y, 1);
        }
    }

    const int radiusX = qMax(10, width / 8);
    const int radiusY = qMax(10, height / 8);
    const QPoint center(width / 2, height / 2);
    for (int y = qMax(0, center.y() - radiusY); y < qMin(height, center.y() + radiusY); ++y) {
        for (int x = qMax(0, center.x() - radiusX); x < qMin(width, center.x() + radiusX); ++x) {
            const double dx = double(x - center.x()) / double(radiusX);
            const double dy = double(y - center.y()) / double(radiusY);
            if (dx * dx + dy * dy <= 1.0) {
                putSegmentPixel(x, y, 2);
            }
        }
    }

    for (int y = bottomRight.top(); y <= bottomRight.bottom() && y < height; ++y) {
        for (int x = bottomRight.left(); x <= bottomRight.right() && x < width; ++x) {
            putSegmentPixel(x, y, 3);
        }
    }

    rebuildMetadata();
}

bool SegmentMap::loadNpy(const QString &path, QString *errorMessage)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    const QByteArray bytes = file.readAll();
    if (bytes.size() < 16 || !bytes.startsWith("\x93NUMPY")) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Not a NumPy .npy file");
        }
        return false;
    }

    const uchar major = uchar(bytes[6]);
    int headerOffset = 0;
    int headerLength = 0;
    if (major == 1) {
        headerOffset = 10;
        headerLength = uchar(bytes[8]) | (uchar(bytes[9]) << 8);
    } else if (major == 2) {
        headerOffset = 12;
        headerLength = uchar(bytes[8]) | (uchar(bytes[9]) << 8) | (uchar(bytes[10]) << 16) | (uchar(bytes[11]) << 24);
    } else {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Unsupported .npy version");
        }
        return false;
    }

    if (bytes.size() < headerOffset + headerLength) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Truncated .npy header");
        }
        return false;
    }

    const QString header = QString::fromLatin1(bytes.constData() + headerOffset, headerLength);
    if (!header.contains("'descr': '<u2'") && !header.contains("\"descr\": \"<u2\"")) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Expected uint16 segment map");
        }
        return false;
    }
    if (header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true")) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Fortran-order .npy arrays are not supported");
        }
        return false;
    }

    const QRegularExpression shapeRe(QStringLiteral("'shape'\\s*:\\s*\\((\\d+)\\s*,\\s*(\\d+)\\s*,?\\)"));
    QRegularExpressionMatch match = shapeRe.match(header);
    if (!match.hasMatch()) {
        const QRegularExpression jsonShapeRe(QStringLiteral("\"shape\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]"));
        match = jsonShapeRe.match(header);
    }
    if (!match.hasMatch()) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Could not parse .npy shape");
        }
        return false;
    }

    const int height = match.captured(1).toInt();
    const int width = match.captured(2).toInt();
    const int dataOffset = headerOffset + headerLength;
    const qint64 expectedBytes = qint64(width) * qint64(height) * qint64(sizeof(quint16));
    if (width <= 0 || height <= 0 || bytes.size() < dataOffset + expectedBytes) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Truncated .npy segment payload");
        }
        return false;
    }

    m_width = width;
    m_height = height;
    m_pixels.resize(width * height);
    memcpy(m_pixels.data(), bytes.constData() + dataOffset, expectedBytes);
    rebuildMetadata();
    return true;
}

quint16 SegmentMap::segmentAt(const QPoint &point) const
{
    if (point.x() < 0 || point.y() < 0 || point.x() >= m_width || point.y() >= m_height) {
        return 0;
    }
    return m_pixels[point.y() * m_width + point.x()];
}

const SegmentMetadata *SegmentMap::metadata(quint16 id) const
{
    for (const SegmentMetadata &segment : m_segments) {
        if (segment.id == id) {
            return &segment;
        }
    }
    return nullptr;
}

QRect SegmentMap::segmentBounds(quint16 id) const
{
    if (const SegmentMetadata *segment = metadata(id)) {
        return segment->bbox;
    }
    return QRect();
}

QImage SegmentMap::overlayImage(quint16 id, const QColor &color) const
{
    const QRect bbox = segmentBounds(id);
    if (bbox.isEmpty()) {
        return QImage();
    }

    QImage image(bbox.size(), QImage::Format_ARGB32_Premultiplied);
    image.fill(Qt::transparent);

    const QRgb overlay = color.rgba();
    for (int y = 0; y < bbox.height(); ++y) {
        QRgb *line = reinterpret_cast<QRgb *>(image.scanLine(y));
        for (int x = 0; x < bbox.width(); ++x) {
            const QPoint p = bbox.topLeft() + QPoint(x, y);
            if (segmentAt(p) == id) {
                line[x] = overlay;
            }
        }
    }

    return image;
}

QByteArray SegmentMap::selectionMask(quint16 id, QRect *maskBounds) const
{
    const QRect bbox = segmentBounds(id);
    if (maskBounds) {
        *maskBounds = bbox;
    }
    if (bbox.isEmpty()) {
        return QByteArray();
    }

    QByteArray bytes;
    bytes.resize(bbox.width() * bbox.height());
    uchar *out = reinterpret_cast<uchar *>(bytes.data());
    for (int y = 0; y < bbox.height(); ++y) {
        for (int x = 0; x < bbox.width(); ++x) {
            const QPoint p = bbox.topLeft() + QPoint(x, y);
            out[y * bbox.width() + x] = segmentAt(p) == id ? 255 : 0;
        }
    }

    return bytes;
}

void SegmentMap::putSegmentPixel(int x, int y, quint16 id)
{
    if (x < 0 || y < 0 || x >= m_width || y >= m_height) {
        return;
    }
    m_pixels[y * m_width + x] = id;
}

void SegmentMap::rebuildMetadata()
{
    m_segments.clear();

    struct MutableMetadata {
        QRect bbox;
        int area = 0;
        bool seen = false;
    };

    QVector<MutableMetadata> mutableSegments(4);
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            const quint16 id = m_pixels[y * m_width + x];
            if (id == 0) {
                continue;
            }
            if (id >= mutableSegments.size()) {
                mutableSegments.resize(id + 1);
            }

            MutableMetadata &segment = mutableSegments[id];
            const QPoint point(x, y);
            segment.bbox = segment.seen ? segment.bbox.united(QRect(point, QSize(1, 1)))
                                        : QRect(point, QSize(1, 1));
            segment.area += 1;
            segment.seen = true;
        }
    }

    for (int id = 1; id < mutableSegments.size(); ++id) {
        const MutableMetadata &segment = mutableSegments[id];
        if (!segment.seen) {
            continue;
        }
        m_segments.push_back(SegmentMetadata{quint16(id), segment.bbox, segment.area, 1.0});
    }
}
