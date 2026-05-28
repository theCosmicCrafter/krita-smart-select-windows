#ifndef SMART_SELECT_LASSO_REGION_MASK_H_
#define SMART_SELECT_LASSO_REGION_MASK_H_

#include <QByteArray>
#include <QPointF>
#include <QRect>
#include <QSize>
#include <QString>
#include <QVector>

struct LassoRegionMask
{
    QSize canvasSize;
    QRect bbox;
    int area = 0;
    QByteArray bytes;
};

bool buildLassoRegionMask(const QVector<QPointF>& points, const QSize& canvasSize,
    LassoRegionMask *mask, QString *errorMessage = nullptr);

bool writeLassoRegionMaskNpy(
    const QString& path, const LassoRegionMask& mask, QString *errorMessage = nullptr);

#endif // SMART_SELECT_LASSO_REGION_MASK_H_
