"""Loading the definition layer off disk."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from featurepipe.spec import FeatureView


class DefinitionError(ValueError):
    """A definition file is unusable. Always names the file."""


class Registry(BaseModel):
    """Every view, with the checks that only make sense across views."""

    model_config = ConfigDict(frozen=True)

    views: tuple[FeatureView, ...]

    @model_validator(mode="after")
    def _names_are_unique_everywhere(self) -> Registry:
        view_names = [view.name for view in self.views]
        repeated = sorted({name for name in view_names if view_names.count(name) > 1})
        if repeated:
            raise ValueError(f"views declared twice: {', '.join(repeated)}")
        owners: dict[str, str] = {}
        for view in self.views:
            for spec in view.features:
                if spec.name in owners:
                    raise ValueError(
                        f"feature {spec.name!r} is declared in both {owners[spec.name]} "
                        f"and {view.name}; a training frame and a served row are both "
                        "flat, so the name has to be unique across views"
                    )
                owners[spec.name] = view.name
        return self

    @classmethod
    def from_directory(cls, directory: str | Path) -> Registry:
        path = Path(directory)
        if not path.is_dir():
            raise DefinitionError(f"{path} is not a directory")
        files = sorted(path.glob("*.toml"))
        if not files:
            raise DefinitionError(f"no .toml definitions in {path}")
        views = []
        for file in files:
            with file.open("rb") as handle:
                try:
                    raw = tomllib.load(handle)
                except tomllib.TOMLDecodeError as error:
                    raise DefinitionError(f"{file.name}: {error}") from error
            try:
                views.append(FeatureView.model_validate(raw))
            except ValueError as error:
                raise DefinitionError(f"{file.name}: {error}") from error
        try:
            return cls(views=tuple(views))
        except ValueError as error:
            raise DefinitionError(str(error)) from error

    def view(self, name: str) -> FeatureView:
        for view in self.views:
            if view.name == name:
                return view
        known = ", ".join(view.name for view in self.views)
        raise DefinitionError(f"no view named {name!r}; the registry holds {known}")

    def owner_of(self, feature_name: str) -> FeatureView:
        for view in self.views:
            for spec in view.features:
                if spec.name == feature_name:
                    return view
        raise DefinitionError(f"no feature named {feature_name!r} in any view")
