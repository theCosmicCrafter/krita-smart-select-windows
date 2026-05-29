#include "SelectionMask.h"

#include <QFile>
#include <QRegularExpression>

#include <cstring>

void SelectionMask::clear()
{
    m_width = 0;
    m_height = 0;
    m_pixels.clear();
    m_selectedBounds = QRect();
}

bool SelectionMask::loadNpy(const QString& path, QString *errorMessage)
{
    clear();

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
        headerLength = uchar(bytes[8]) | (uchar(bytes[9]) << 8) | (uchar(bytes[10]) << 16)
            | (uchar(bytes[11]) << 24);
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
    if (!header.contains("'descr': '|u1'") && !header.contains("\"descr\": \"|u1\"")) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Expected uint8 selection mask");
        }
        return false;
    }
    if (header.contains("'fortran_order': True") || header.contains("\"fortran_order\": true")) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Fortran-order .npy arrays are not supported");
        }
        return false;
    }

    const QRegularExpression shapeRe(
        QStringLiteral("'shape'\\s*:\\s*\\((\\d+)\\s*,\\s*(\\d+)\\s*,?\\)"));
    QRegularExpressionMatch match = shapeRe.match(header);
    if (!match.hasMatch()) {
        const QRegularExpression jsonShapeRe(
            QStringLiteral("\"shape\"\\s*:\\s*\\[(\\d+)\\s*,\\s*(\\d+)\\]"));
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
    const qint64 expectedBytes = qint64(width) * qint64(height);
    if (width <= 0 || height <= 0 || bytes.size() < dataOffset + expectedBytes) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Truncated .npy selection payload");
        }
        return false;
    }

    m_width = width;
    m_height = height;
    m_pixels.resize(width * height);
    memcpy(m_pixels.data(), bytes.constData() + dataOffset, expectedBytes);
    rebuildSelectedBounds();
    return true;
}

bool SelectionMask::isEmpty() const
{
    return m_width <= 0 || m_height <= 0 || m_pixels.isEmpty() || m_selectedBounds.isEmpty();
}

QSize SelectionMask::size() const
{
    return QSize(m_width, m_height);
}

QRect SelectionMask::bounds() const
{
    return QRect(0, 0, m_width, m_height);
}

QRect SelectionMask::selectedBounds() const
{
    return m_selectedBounds;
}

QByteArray SelectionMask::bytesForBounds(const QRect& requestedBounds) const
{
    const QRect clipped = requestedBounds.intersected(bounds());
    if (clipped.isEmpty()) {
        return QByteArray();
    }

    QByteArray bytes;
    bytes.resize(clipped.width() * clipped.height());
    uchar *out = reinterpret_cast<uchar *>(bytes.data());
    for (int y = 0; y < clipped.height(); ++y) {
        const int sourceY = clipped.y() + y;
        for (int x = 0; x < clipped.width(); ++x) {
            const int sourceX = clipped.x() + x;
            out[y * clipped.width() + x] = m_pixels[sourceY * m_width + sourceX];
        }
    }
    return bytes;
}

void SelectionMask::rebuildSelectedBounds()
{
    m_selectedBounds = QRect();
    bool seen = false;
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            if (m_pixels[y * m_width + x] == 0) {
                continue;
            }
            const QRect pixelRect(QPoint(x, y), QSize(1, 1));
            m_selectedBounds = seen ? m_selectedBounds.united(pixelRect) : pixelRect;
            seen = true;
        }
    }
}
