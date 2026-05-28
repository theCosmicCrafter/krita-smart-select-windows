#ifndef SMART_SELECT_SELECTION_MASK_H_
#define SMART_SELECT_SELECTION_MASK_H_

#include <QByteArray>
#include <QRect>
#include <QSize>
#include <QString>
#include <QVector>

class SelectionMask
{
public:
    void clear();
    bool loadNpy(const QString& path, QString *errorMessage = nullptr);

    bool isEmpty() const;
    QSize size() const;
    QRect bounds() const;
    QRect selectedBounds() const;
    QByteArray bytesForBounds(const QRect& bounds) const;

private:
    void rebuildSelectedBounds();

    int m_width = 0;
    int m_height = 0;
    QVector<quint8> m_pixels;
    QRect m_selectedBounds;
};

#endif // SMART_SELECT_SELECTION_MASK_H_
