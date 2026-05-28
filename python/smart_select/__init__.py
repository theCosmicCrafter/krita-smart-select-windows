from krita import Krita

from .extension import SmartSelectExtension

Krita.instance().addExtension(SmartSelectExtension(Krita.instance()))
