"""Cihaz profili yükleme, sınıf sayımı ve referans tablosu doğrulama."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "device_profiles.yaml"


def _norm(name: str) -> str:
    return re.sub(r"[\s_]+", "-", name.strip().lower())


def _aliases_set(aliases: list[str]) -> set[str]:
    return {_norm(a) for a in aliases}


@dataclass
class ComponentSpec:
    key: str
    count: int
    aliases: list[str]

    @property
    def norm_aliases(self) -> set[str]:
        return _aliases_set(self.aliases)


@dataclass
class DeviceProfile:
    key: str
    display_name: str
    device_markers: list[str]
    front_components: dict[str, ComponentSpec]
    label_components: dict[str, ComponentSpec]

    @property
    def marker_norms(self) -> set[str]:
        return _aliases_set(self.device_markers)


@dataclass
class CountMismatch:
    component: str
    expected: int
    actual: int
    section: str  # "front" | "labels"


@dataclass
class ValidationResult:
    ok: bool
    mismatches: list[CountMismatch] = field(default_factory=list)


def load_profiles(path: Path | None = None) -> dict[str, DeviceProfile]:
    path = path or CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    profiles: dict[str, DeviceProfile] = {}
    for dev_key, dev in raw["devices"].items():

        def _parse_components(section: dict[str, Any]) -> dict[str, ComponentSpec]:
            out: dict[str, ComponentSpec] = {}
            for comp_key, spec in section.items():
                out[comp_key] = ComponentSpec(
                    key=comp_key,
                    count=int(spec["count"]),
                    aliases=list(spec.get("aliases", [comp_key])),
                )
            return out

        profiles[dev_key] = DeviceProfile(
            key=dev_key,
            display_name=str(dev.get("display_name", dev_key)),
            device_markers=list(dev.get("device_markers", [dev_key])),
            front_components=_parse_components(dev.get("front_components", {})),
            label_components=_parse_components(dev.get("label_components", {})),
        )
    return profiles


def build_name_to_count(names_list: list[str], class_counts: dict[int, int]) -> dict[str, int]:
    """YOLO class index sayımlarını model sınıf adına çevirir."""
    by_name: dict[str, int] = {}
    for cid, cnt in class_counts.items():
        if cnt <= 0:
            continue
        name = names_list[cid]
        by_name[name] = by_name.get(name, 0) + cnt
    return by_name


def count_for_spec(by_name: dict[str, int], spec: ComponentSpec) -> int:
    total = 0
    norms = spec.norm_aliases
    for model_name, cnt in by_name.items():
        if _norm(model_name) in norms:
            total += cnt
    return total


def excluded_norms_for_front_id(profiles: dict[str, DeviceProfile]) -> set[str]:
    """Ön yüz tip tanımında kullanılmayan sınıflar (marker + etiket sınıfları)."""
    excluded: set[str] = set()
    for profile in profiles.values():
        excluded |= profile.marker_norms
        for spec in profile.label_components.values():
            excluded |= spec.norm_aliases
    return excluded


def filter_for_front_id(by_name: dict[str, int], profiles: dict[str, DeviceProfile]) -> dict[str, int]:
    excluded = excluded_norms_for_front_id(profiles)
    return {name: cnt for name, cnt in by_name.items() if _norm(name) not in excluded and cnt > 0}


def validate_section(
    by_name: dict[str, int],
    components: dict[str, ComponentSpec],
    section: str,
) -> ValidationResult:
    mismatches: list[CountMismatch] = []
    for comp_key, spec in components.items():
        actual = count_for_spec(by_name, spec)
        if actual != spec.count:
            mismatches.append(
                CountMismatch(component=comp_key, expected=spec.count, actual=actual, section=section)
            )
    return ValidationResult(ok=len(mismatches) == 0, mismatches=mismatches)


@dataclass
class DeviceGuess:
    key: str | None
    profile: DeviceProfile | None
    score: float = 0.0
    second_score: float = 0.0


def score_profile_front(by_name: dict[str, int], profile: DeviceProfile) -> float:
    """Ön yüz komponent sayılarına göre 0..1 uyum skoru (yalnızca front_components)."""
    score = 0.0
    total_weight = 0
    for spec in profile.front_components.values():
        actual = count_for_spec(by_name, spec)
        weight = max(1, spec.count)
        total_weight += weight
        if actual == spec.count:
            score += weight
        elif actual > 0:
            # Kısmi eşleşme düşük ağırlık — erken yanlış tip seçimini azaltır
            score += weight * 0.2
    return score / total_weight if total_weight else 0.0


def identify_device_from_components(
    by_name: dict[str, int],
    profiles: dict[str, DeviceProfile],
    *,
    min_score: float = 0.88,
    min_margin: float = 0.12,
) -> DeviceGuess:
    """
    Cihaz tipi yalnızca ön yüz komponent sayımlarından seçilir.
    DM100/XIO110 gibi marker sınıfları ve etiket sınıfları dikkate alınmaz.
    """
    filtered = filter_for_front_id(by_name, profiles)
    if not filtered:
        return DeviceGuess(None, None, 0.0, 0.0)

    ranked: list[tuple[str, DeviceProfile, float]] = []
    for key, profile in profiles.items():
        ranked.append((key, profile, score_profile_front(filtered, profile)))
    ranked.sort(key=lambda x: x[2], reverse=True)

    best_key, best_profile, best_score = ranked[0]
    second_score = ranked[1][2] if len(ranked) > 1 else 0.0

    if best_score < min_score or (best_score - second_score) < min_margin:
        return DeviceGuess(None, None, best_score, second_score)

    return DeviceGuess(best_key, best_profile, best_score, second_score)


def identify_device(
    by_name: dict[str, int],
    profiles: dict[str, DeviceProfile],
) -> tuple[str | None, DeviceProfile | None]:
    """Geriye uyumluluk: komponent tabanlı tanıma yönlendirir."""
    guess = identify_device_from_components(by_name, profiles)
    return guess.key, guess.profile


def format_raw_front_lines(by_name: dict[str, int], profiles: dict[str, DeviceProfile]) -> list[str]:
    """Cihaz tipi seçilmeden önce ham komponent sayıları."""
    filtered = filter_for_front_id(by_name, profiles)
    if not filtered:
        return ["Komponent: (yok)"]
    return [f"{name}: {cnt}" for name, cnt in sorted(filtered.items())]


def format_counts_line(by_name: dict[str, int], profile: DeviceProfile, section: str) -> list[str]:
    components = profile.front_components if section == "front" else profile.label_components
    lines = []
    for comp_key, spec in components.items():
        actual = count_for_spec(by_name, spec)
        mark = "OK" if actual == spec.count else "!"
        lines.append(f"{comp_key}: {actual}/{spec.count} {mark}")
    return lines


def format_mismatches(mismatches: list[CountMismatch]) -> list[str]:
    return [f"{m.section}:{m.component} {m.actual}/{m.expected}" for m in mismatches]
