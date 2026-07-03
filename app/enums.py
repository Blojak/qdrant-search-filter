"""Kontrollierte Vokabulare fuer filterbare Metadatenfelder.

Diese Enums bilden den erlaubten Wertebereich der Felder ``language``,
``doc_type`` und ``classification`` ab. Freitext ist fuer diese Felder bewusst
ausgeschlossen, damit Filter in Qdrant und Postgres verlaesslich funktionieren.
"""

from __future__ import annotations

import enum


class Language(str, enum.Enum):
    """Sprache eines Dokuments (ISO-639-1, Teilmenge)."""

    DE = "de"
    EN = "en"
    FR = "fr"
    ES = "es"
    IT = "it"
    UNKNOWN = "unknown"


class DocType(str, enum.Enum):
    """Grobe fachliche Dokumentart."""

    REPORT = "report"
    CONTRACT = "contract"
    INVOICE = "invoice"
    EMAIL = "email"
    ARTICLE = "article"
    MANUAL = "manual"
    NOTE = "note"
    OTHER = "other"


class Classification(str, enum.Enum):
    """Administrative Vertraulichkeitsstufe."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
