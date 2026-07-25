"""
base.py — Abstract base class for all ICH segmentation strategies.

Each strategy must implement: prepare_data(), train(), predict(),
and expose a Pydantic config model for type-safe parameter handling.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class ICHStrategy(ABC):
    """
    Abstract base for ICH (Intracranial Hemorrhage) segmentation strategies.

    Subclasses must define class-level metadata and implement all abstract
    methods. The strategy registry uses `name` as the unique identifier.
    """

    # ── Class-level metadata (override in subclasses) ──
    name: ClassVar[str]
    display_name: ClassVar[str]
    description: ClassVar[str]

    # ═════════════════════════════════════════════════════════════════════
    # Abstract Methods
    # ═════════════════════════════════════════════════════════════════════

    @abstractmethod
    def prepare_data(self) -> bool:
        """
        Prepare / preprocess raw data into the format this strategy requires.

        Returns:
            True if preparation succeeded, False otherwise.
        """
        ...

    @abstractmethod
    def train(self, config: BaseModel) -> bool:
        """
        Train the model using the given validated configuration.

        Args:
            config: Strategy-specific Pydantic config model instance.

        Returns:
            True if training completed successfully.
        """
        ...

    @abstractmethod
    def predict(self, study_dir: str) -> dict:
        """
        Run inference on a single DICOM study directory.

        Args:
            study_dir: Path to a directory containing DICOM slices (.dcm).

        Returns:
            Dictionary with 7 intermediate keys:
                V_EDH, V_SDH, V_IPH, V_SAH, V_IVH,
                fracture_prob, MLS_mm
        """
        ...

    @abstractmethod
    def get_config_class(self) -> type[BaseModel]:
        """
        Return the Pydantic config class for this strategy.

        Used by Streamlit to dynamically generate config forms and by
        ZenML steps for JSON deserialization / validation.
        """
        ...

    # ═════════════════════════════════════════════════════════════════════
    # Concrete Helpers
    # ═════════════════════════════════════════════════════════════════════

    def get_config_schema(self) -> dict:
        """
        Generate JSON Schema for this strategy's config model.

        Used by the Streamlit UI to render dynamic configuration forms
        without hard-coding field names or types.
        """
        schema = self.get_config_class().model_json_schema()
        return schema

    def get_default_config(self) -> dict:
        """
        Return default config values as a plain dict.

        Useful for pre-populating UI forms or fallback values.
        """
        config_class = self.get_config_class()
        return config_class().model_dump()

    def validate_config(self, config_dict: dict) -> BaseModel:
        """
        Validate and coerce a raw config dict into a typed Pydantic model.

        Raises:
            pydantic.ValidationError: if the dict fails validation.
        """
        return self.get_config_class().model_validate(config_dict)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
