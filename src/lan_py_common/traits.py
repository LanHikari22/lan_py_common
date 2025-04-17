from typing import Any, Dict, Protocol, Self

from result import Result

class TrToJson(Protocol):
    def to_json(self) -> str:
        """
        Converts instance to JSON
        """

class TrDescribe(Protocol):
    def describe(self) -> str:
        """
        Offers a description for the object
        """

class TrVerifyBySchema(Protocol):
    def verify_by_schema(self, schema: Any) -> Result[bool, any]:
        """
        Checks that the object upholds the schema contract
        """

class TrFromSchema(Protocol):
    @staticmethod
    def from_schema(schema: Any) -> Result[Self, Any]:
        """
        Creates the object from a schema
        """

class TrFromSchemaAndData(Protocol):
    @staticmethod
    def from_schema_and_data(schema: Any, data: dict) -> Result[Self, Any]:
        """
        Creates the object from a schema and initial data
        """

class TrMapSchema(Protocol):
    def map_schema(self, new_schema: Any, col_to_new_col: Dict[str, str]) -> Result[Self, Any]:
        """
        Maps an object from one schema to another, typically by considering only a subset of its
        columns and renaming the columns. This doesn't convert column types.
        """


class TrFromStr(Protocol):
    @staticmethod
    def from_str(s: str) -> Result[Self, Any]:
        """
        Creates the object from a schema
        """