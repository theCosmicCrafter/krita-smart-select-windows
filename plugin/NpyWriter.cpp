#include "NpyWriter.h"

#include <QFile>

QByteArray makeNpyBytes(const QByteArray& descr, const QList<int>& shape, const QByteArray& payload)
{
    QByteArray shapeText;
    if (shape.size() == 1) {
        shapeText = "(" + QByteArray::number(shape[0]) + ",)";
    } else {
        shapeText = "(";
        for (int i = 0; i < shape.size(); ++i) {
            if (i > 0) {
                shapeText += ", ";
            }
            shapeText += QByteArray::number(shape[i]);
        }
        shapeText += ")";
    }

    QByteArray header
        = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shapeText + ", }";
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

bool writeUint8Npy(
    const QString& path, const QSize& size, const QByteArray& bytes, QString *errorMessage)
{
    if (!size.isValid() || size.width() <= 0 || size.height() <= 0) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Invalid uint8 .npy size");
        }
        return false;
    }

    const qsizetype expectedSize = qsizetype(size.width()) * qsizetype(size.height());
    if (bytes.size() != expectedSize) {
        if (errorMessage) {
            *errorMessage = QStringLiteral("Invalid uint8 .npy payload size");
        }
        return false;
    }

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly)) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    const QByteArray npy = makeNpyBytes("|u1", QList<int> { size.height(), size.width() }, bytes);
    if (file.write(npy) != npy.size()) {
        if (errorMessage) {
            *errorMessage = file.errorString();
        }
        return false;
    }

    return true;
}
