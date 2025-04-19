import enum
from typing import Hashable, Self, Tuple, cast
import checkpipe as pipe
from dataclasses import dataclass
import classes
import pandas as pd # type: ignore
import numpy as np
import copy
from pandas.api.types import pandas_dtype, is_extension_array_dtype # type: ignore
from pandas.core.dtypes.base import ExtensionDtype # type: ignore
from .traits import *
from .common import *

DType = Union[np.dtype, ExtensionDtype]

def ty_s_to_nullable_alternative(ty_s: str) -> str:
    return (
        ty_s.capitalize()
            if ty_s.startswith('int') or ty_s.startswith('uint') else
        "boolean"
            if ty_s == "bool" else
        "string"
            if ty_s == "str" else
        ty_s
    )

def nullable_alternative_to_ty_s(ty_s: str) -> str:
    return (
        ty_s.lower()
            if ty_s.startswith('Int') or ty_s.startswith('Uint') else
        "bool"
            if ty_s == "boolean" else
        "str"
            if ty_s == "string" else
        ty_s
    )

def is_dtype_compatible(val: Any, dtype: str) -> bool:
    if val is None:
        return True  # Let nullability logic handle this
    try:
        # allow upcast
        if nullable_alternative_to_ty_s(dtype) == 'float':
            if val == int:
                val = float

        if np.issubdtype(val, np.dtype(nullable_alternative_to_ty_s(dtype))):
            return True
        else:
            pd.Series([val], dtype=pandas_dtype(dtype))
            return True
    except Exception:
        print(f"NOPE {val} {dtype}")
        return False

class FromStrForDfJsonSchemaErrTy(enum.Enum):
    FailedToParseJson = enum.auto()
    SchemaNotFlat = enum.auto()
    FailedToParseTyName = enum.auto()

FromStrForDfJsonSchemaErr = Tuple[FromStrForDfJsonSchemaErrTy, str]

class ImplTrFromStr_ForDfJsonSchema(Protocol):
    @staticmethod
    def from_str(s: str) -> Result['DfJsonSchema', FromStrForDfJsonSchemaErr]:
        import json

        # Make sure this is valid JSON
        try:
            s_jsn_col_to_val: dict = json.loads(s)
        except json.decoder.JSONDecodeError as e:
            return Err((FromStrForDfJsonSchemaErrTy.FailedToParseJson, e.msg))

        # Ensure flatness of the data and that it only describes purely columns
        def is_flat_fn() -> bool:
            return (
                s_jsn_col_to_val
                    .items()
                    .__iter__()
                    | pipe.OfIter[Tuple[str, Any]]
                    .all(pipe.tup2_unpack(lambda _, val:
                        isinstance(val, str)
                    ))
            )

        if not is_flat_fn():
            return Err((FromStrForDfJsonSchemaErrTy.SchemaNotFlat, ""))
        
        # Extract the special ty column
        ty_name_query = (
            s_jsn_col_to_val
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[str, Any]]
                .filter(pipe.tup2_unpack(lambda col, _:
                    col == "ty"
                ))
                | pipe.OfIter[Tuple[str, Any]]
                .map(pipe.tup2_unpack(lambda _, name:
                    name
                ))
                | pipe.OfIter[str]
                .to_list()
        )
        if len(ty_name_query) != 1:
            return Err((FromStrForDfJsonSchemaErrTy.FailedToParseTyName, ""))

        # exclude ty from further consideration
        s_jsn_col_to_val = (
            s_jsn_col_to_val
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[str, Any]]
                .filter(pipe.tup2_unpack(lambda col, _:
                    col != "ty"
                ))
                | pipe.OfIter[Tuple[str, Any]]
                .to_list()
                | pipe.Of[List[Tuple[str, Any]]]
                .map(lambda lst: dict(lst))
        )

        cols = (
            s_jsn_col_to_val
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[str, Any]]
                .map(pipe.tup2_unpack(lambda col, _:
                    col
                ))
                | pipe.OfIter[str]
                .to_list()
        )
        
        col_to_nullable = (
                s_jsn_col_to_val
                    .items()
                    .__iter__()
                    | pipe.OfIter[Tuple[str, Any]]
                    .map(pipe.tup2_unpack(lambda col, val:
                        (col, str(val).startswith('Optional[') and str(val).endswith(']'))
                    ))
                    | pipe.OfIter[Tuple[str, bool]]
                    .to_list()
                    | pipe.Of[List[Tuple[str, bool]]]
                    .map(lambda lst: dict(lst))
        )

        col_to_ty_s = (
                s_jsn_col_to_val
                    .items()
                    .__iter__()
                    | pipe.OfIter[Tuple[str, Any]]
                    .map(pipe.tup2_unpack(lambda col, ty:
                        (col, ty)
                            if not col_to_nullable[col] else
                        (col, str(ty)
                                .replace("Optional[", "")
                                .replace("]", "")
                                | pipe.Of[str]
                                .map(ty_s_to_nullable_alternative)
                        )
                    ))
                    | pipe.OfIter[Tuple[str, str]]
                    .to_list()
                    | pipe.Of[List[Tuple[str, str]]]
                    .map(lambda lst: dict(lst))
        )
        
        return Ok(DfJsonSchema(s, ty_name_query[0], cols, col_to_ty_s, col_to_nullable))

class ImplTrFromDict_ForDfJsonSchema(TrFromDict):
    @staticmethod
    def from_dict(d: Dict[Hashable, Any]) -> Result['DfJsonSchema', FromStrForDfJsonSchemaErr]:
        import json

        return DfJsonSchema.from_str(json.dumps(d))
    
@dataclass
class DfJsonSchema(
    ImplTrFromStr_ForDfJsonSchema,
    ImplTrFromDict_ForDfJsonSchema
):
    schema: str
    ty: str
    cols: List[str]
    col_to_ty_s: Dict[str, str]
    col_to_nullable: Dict[str, bool]


class ImplTrDescribe_ForDf(TrDescribe):
    def describe(self: 'Df') -> str: # type: ignore
        return f'Df with schema {self.schema}'

class VerifyBySchemaDfErrTy(enum.Enum):
    UnequalNumOfColumns = enum.auto()
    SchemaOrderError = enum.auto()
    WrongTypeError = enum.auto()
    UnequalSchemaTy = enum.auto()

VerifyBySchemaDfErr = Tuple[VerifyBySchemaDfErrTy, str]

class ImplTrVerifyBySchemaForDf(TrVerifyBySchema):
    def verify_by_schema(self: 'Df', schema: DfJsonSchema, verify_data=False) -> Result[None, VerifyBySchemaDfErr]: # type: ignore
        import numpy as np

        if self.schema.ty == schema.ty:
            return Err((VerifyBySchemaDfErrTy.UnequalSchemaTy, ""))

        if len(schema.col_to_ty_s.items()) != len(self.df.dtypes.items()):
            return Err((VerifyBySchemaDfErrTy.UnequalNumOfColumns, ""))

        def all_tys_matching() -> Result[None, VerifyBySchemaDfErr]:
            return (
                self.df.dtypes.items()
                    | pipe.OfIter[Tuple[str, np.dtype]]
                    .zip(schema.col_to_ty_s.items())
                    | pipe.OfIter[Tuple[Tuple[str, np.dtype], Tuple[str, str]]]
                    .map(pipe.tup2_tup2_flatten)
                    | pipe.OfIter[Tuple[str, np.dtype, str, str]]
                    .map(pipe.tup4_unpack(lambda df_col, df_ty, sch_col, sch_ty_s:
                        Err((VerifyBySchemaDfErrTy.SchemaOrderError, ""))
                            if df_col != sch_col else
                        Err((VerifyBySchemaDfErrTy.WrongTypeError, f"Expected {sch_ty_s} found {df_ty}"))
                            if df_ty != pandas_dtype(sch_ty_s) else
                        Ok(None)
                    ))
                    | pipe.OfResultIter[None, VerifyBySchemaDfErr]
                    .flatten()
                    | pipe.OfResult[List[None], VerifyBySchemaDfErr]
                    .map_ok(lambda _: None)
            )
        
        def check_rows_for_null() -> Result[None, VerifyBySchemaDfErr]:
            return (
                self.df
                    .to_dict(orient="records")
                    | pipe.OfIter[Dict[Hashable, Any]]
                    .map(lambda col_to_val:
                        col_to_val
                            .items()
                            .__iter__()
                            | pipe.OfIter[Tuple[Hashable, Any]]
                            .map(pipe.tup2_unpack(lambda col, val:
                                Err((CreateDfErrTy.InvalidNullValue, 
                                    f'expected non-null type {schema.col_to_ty_s[str(col)]} for col {col}'))
                                    if not schema.col_to_nullable[str(col)] and val is None else
                                Err((CreateDfErrTy.ColumnOfInvalidTy, 
                                    f'expected type {schema.col_to_ty_s[str(col)]} found {type(val)} for col {col}'))
                                    if val is not None and not is_dtype_compatible(type(val), schema.col_to_ty_s[str(col)]) else
                                Ok(None)
                            ))
                            | pipe.OfResultIter[None, CreateDfErr]
                            .flatten()
                    )
                    | pipe.OfResultIter[List[None], CreateDfErr]
                    .flatten()
                    | pipe.OfResult[List[List[None]], CreateDfErr]
                    .map_ok(lambda _: None)
            )

        return (
            all_tys_matching()
                .and_then(lambda _: 
                    Ok(None)
                        if not verify_data else
                    check_rows_for_null()
                )
        )


class CreateDfErrTy(enum.Enum):
    DatatypeNotUnderstood = enum.auto()
    InvalidDatarecords = enum.auto()
    ColumnOfInvalidTy = enum.auto()
    InvalidNullValue = enum.auto()
    RenameMapNotSchemaComplete = enum.auto()
    RenameMapNamesInvalidColumn = enum.auto()
    SchemaMapUnequalTys = enum.auto()

CreateDfErr = Tuple[CreateDfErrTy, str]

class ImplTrFromSchema_ForDf(TrFromSchema):
    @staticmethod
    def from_schema(schema: DfJsonSchema) -> Result['Df', CreateDfErr]:
        # Create an empty DataFrame with columns specified in the schema
        df = pd.DataFrame(columns=schema.col_to_ty_s.keys())

        # Convert the columns of the empty DataFrame to the specified data types
        try:
            df = df.astype(schema.col_to_ty_s)
        except TypeError as e:
            return Err((CreateDfErrTy.DatatypeNotUnderstood, str(e)))

        return Ok(Df(schema, df))

class ImplTrFromschemaAndData_ForDf(TrFromSchemaAndData):
    @staticmethod
    def from_schema_and_data(
        schema: DfJsonSchema, 
        col_to_val_per_row_union: Dict[str, Union[List[Any], np.ndarray[Any, Any]]]
        ) -> Result['Df', CreateDfErr]:

        init_df_res = Df.from_schema(schema)
        if init_df_res.is_err():
            return init_df_res
        init_df = init_df_res.unwrap()

        # First, map out any potential ndarrays to lists
        def val_to_list(val: Union[List[Any], np.ndarray[Any, Any]]) -> List[Any]:
            if isinstance(val, list):
                return val
            else:
                return cast(List[Any], val.tolist())

        col_to_val_per_row = (
            col_to_val_per_row_union
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[str, Union[List[Any], np.ndarray[Any, Any]]]]
                .map(pipe.tup2_unpack(lambda col, val:
                    (
                        col,
                        val_to_list(val)
                     )
                ))
                | pipe.OfIter[Tuple[str, List[Any]]]
                .to_list()
                | pipe.Of[List[Tuple[str, List[Any]]]]
                .map(lambda lst: dict(lst))
        )

        records_res = (
            col_to_val_per_row
                | pipe.to_records()
                | pipe.OfResult[List[Dict[str, Any]], str]
                .map_err(lambda e: (CreateDfErrTy.InvalidDatarecords, e))
        )
        if records_res.is_err():
            return Err(records_res.unwrap_err())
        records = records_res.unwrap()

        val_per_col_per_row_res = (
            records
                | pipe.OfIter[Dict[str, Any]]
                .map(lambda col_to_val: 
                    col_to_val
                        .items()
                        .__iter__()
                        | pipe.OfIter[Tuple[str, Any]]
                        .map(pipe.tup2_unpack(lambda col, val:
                            Err((CreateDfErrTy.InvalidNullValue, 
                                 f'expected non-null type {init_df.schema.col_to_ty_s[col]} for col {col}'))
                                if not init_df.schema.col_to_nullable[col] and val is None else
                            Err((CreateDfErrTy.ColumnOfInvalidTy, 
                                 f'expected type {init_df.schema.col_to_ty_s[col]} found {type(val)} for col {col}'))
                                if val is not None and not is_dtype_compatible(type(val), init_df.schema.col_to_ty_s[col])
                                else
                            Ok(val)
                        ))
                        | pipe.OfResultIter[Any, CreateDfErr]
                        .flatten()
                )
                | pipe.OfResultIter[List[Any], CreateDfErr]
                .flatten()
        )
        if val_per_col_per_row_res.is_err():
            return Err(val_per_col_per_row_res.unwrap_err())
        val_per_col_per_row = val_per_col_per_row_res.unwrap()

        # Add the data
        new_df = pd.DataFrame(val_per_col_per_row, columns=init_df.df.columns)
        new_df = new_df.astype(init_df.schema.col_to_ty_s)
        mut_df = copy.deepcopy(init_df)
        mut_df.df = new_df
        # for val_per_col in val_per_col_per_row:
        #     mut_df.df.loc[len(mut_df)] = val_per_col
        
        return Ok(mut_df)

class ImplTrFromSchemaAndCsv_ForDf(TrFromSchemaAndCsv):
    @staticmethod
    def from_schema_and_csv(schema: DfJsonSchema, csv_filename: str) -> Result['Df', CreateDfErr]:
        df = pd.read_csv(csv_filename)
        val_per_row_per_col = df.to_dict(orient="list")

        filtered_val_per_row_per_col = (
            val_per_row_per_col
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[Hashable, Any]]
                .filter(pipe.tup2_unpack(lambda col, _:
                    col in schema.cols
                ))
                | pipe.OfIter[Tuple[Hashable, Any]]
                .to_list()
                | pipe.Of[List[Tuple[Hashable, Any]]]
                .map(lambda lst: dict(lst))
        )

        return Df.from_schema_and_data(schema, filtered_val_per_row_per_col)

class ImplTrMapSchema_ForDf(TrMapSchema):
    def map_schema( # type: ignore
            self: 'Df',
            new_schema: DfJsonSchema, 
            col_to_new_col: Dict[str, str]
    ) -> Result['Df', CreateDfErr]:
        import json

        if len(new_schema.cols) != len(col_to_new_col.keys()):
            return Err((CreateDfErrTy.RenameMapNotSchemaComplete, 
                        f'rename dict is not complete with new schema: {len(new_schema.cols)} != {len(col_to_new_col.keys())}'))
        
        # Check that all of the columns in the dict keys exist in self.schema
        # Check that all of the columns in the dict vals exist in new_schema
        valid_keys_and_vals = (
            col_to_new_col
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[str, str]]
                .all(pipe.tup2_unpack(lambda col, new_col:
                    col in self.schema.cols and
                    new_col in new_schema.cols
                ))
        )
        if not valid_keys_and_vals:
            return Err((CreateDfErrTy.RenameMapNamesInvalidColumn, 
                        f'Column rename mapper naming invalid columns'))


        # Check that the columns between the maps correspond to the same type
        cols_of_corresponding_tys = (
            col_to_new_col
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[str, str]]
                .all(pipe.tup2_unpack(lambda col, new_col:
                    self.schema.col_to_ty_s[col] == new_schema.col_to_ty_s[new_col] and
                    self.schema.col_to_nullable[col] == new_schema.col_to_nullable[new_col]
                ))
        )
        if not cols_of_corresponding_tys:
            return Err((CreateDfErrTy.SchemaMapUnequalTys, 
                        f'Columns between schemas are not of identical types'))

        # Filter out any columns not in the dict val from the new df
        # Rename the remaining df columns according to the dict

        cols_to_filter = (
            col_to_new_col
                .items()
                .__iter__()
                | pipe.OfIter[Tuple[str, str]]
                .map(pipe.tup2_unpack(lambda col, _:
                    col
                ))
                | pipe.OfIter[str]
                .to_list()
        )

        new_df = (
            self.df[cols_to_filter]
                .rename(columns=col_to_new_col)
        )

        return Ok(Df(new_schema, new_df))

@dataclass
class Df(
    ImplTrDescribe_ForDf,
    ImplTrFromSchema_ForDf,
    ImplTrFromschemaAndData_ForDf,
    ImplTrMapSchema_ForDf,
    ImplTrFromSchemaAndCsv_ForDf,
):
    schema: DfJsonSchema
    df: pd.DataFrame