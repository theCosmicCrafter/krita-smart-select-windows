#ifndef SMART_SELECT_NPY_WRITER_H_
#define SMART_SELECT_NPY_WRITER_H_

#include <QByteArray>
#include <QList>
#include <QSize>
#include <QString>

QByteArray makeNpyBytes(
    const QByteArray& descr, const QList<int>& shape, const QByteArray& payload);

bool writeUint8Npy(const QString& path, const QSize& size, const QByteArray& bytes,
    QString *errorMessage = nullptr);

#endif // SMART_SELECT_NPY_WRITER_H_
