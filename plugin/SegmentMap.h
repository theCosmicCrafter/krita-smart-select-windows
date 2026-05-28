#ifndef SMART_SELECT_SEGMENT_MAP_H_
#define SMART_SELECT_SEGMENT_MAP_H_

#include <QByteArray>
#include <QColor>
#include <QImage>
#include <QPoint>
#include <QRect>
#include <QString>
#include <QVector>

struct SegmentMetadata
{
    quint16 id = 0;
    QRect bbox;
    int area = 0;
    double score = 1.0;
};

class SegmentMap
{
public:
    void clear();
    void buildFake(int width, int height);
    bool loadNpy(const QString& path, QString *errorMessage = nullptr);

    bool isEmpty() const;
    int width() const;
    int height() const;
    QRect bounds() const;
    int segmentCount() const;

    quint16 segmentAt(const QPoint& point) const;
    const SegmentMetadata *metadata(quint16 id) const;
    QRect segmentBounds(quint16 id) const;

    QImage overlayImage(quint16 id, const QColor& color) const;
    QByteArray selectionMask(quint16 id, QRect *maskBounds) const;

private:
    void putSegmentPixel(int x, int y, quint16 id);
    void rebuildMetadata();

    int m_width = 0;
    int m_height = 0;
    QVector<quint16> m_pixels;
    QVector<SegmentMetadata> m_segments;
};

#endif // SMART_SELECT_SEGMENT_MAP_H_
