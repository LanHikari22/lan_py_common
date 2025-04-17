from typing import Any, Dict, List
import pytest
import pandas as pd
from lan_py_common.df import Df, DfJsonSchema, CreateDfErrTy, FromStrForDfJsonSchemaErrTy

@pytest.mark.parametrize("_, schema, xs, ys, opt_error_code", [
    (
        "1",
        '{"ty": "A", "x": "Optional[int64]", "y": "int64"}',
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        None,
    ),
    (
        "2",
        '{"ty": "A", "x": "int64", "y": "Optional[int64]"}',
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        None,
    ),
    (
        "2-1",
        '{"x": "int64", "y": "int64"}',
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        FromStrForDfJsonSchemaErrTy.FailedToParseTyName,
    ),
    (
        "3",
        '{"ty": "A", "x": "int64", "y": "int64"}',
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        CreateDfErrTy.InvalidNullValue,
    ),
    (
        "4",
        '{"ty": "A", "x": "int64", "y": "int64"}',
        [0, 1, 2, None, 4, 5],
        [1, 2, 3, 4, 5],
        CreateDfErrTy.InvalidDatarecords,
    ),
    (
        "5",
        '{"ty": "A", "x": "int64", "y": "int64"}',
        [0, 1, 2, '3', 4],
        [1, 2, 3, 4, 5],
        CreateDfErrTy.ColumnOfInvalidTy,
    ),
])
def test_creating_df_from_schema_and_data(
    _: str,
    schema: str, 
    xs: List[Any], 
    ys: List[Any], 
    opt_error_code: Any):

    df_json_schema_res = (
        DfJsonSchema
            .from_str(schema)
    )

    if opt_error_code is FromStrForDfJsonSchemaErrTy.FailedToParseTyName:
        assert df_json_schema_res.unwrap_err()[0] == opt_error_code
        return
    df_json_schema = df_json_schema_res.unwrap()

    res = (
        Df
            .from_schema_and_data(
                df_json_schema,
                {
                    "x": xs,
                    "y": ys,
                }
            )
    )

    if opt_error_code is not None:
        assert res.unwrap_err()[0] == opt_error_code
    else:
        res.unwrap()

@pytest.mark.parametrize("_, schema, new_schema, col_to_new_col, xs, ys, opt_error_code", [
    (
        "1",
        '{"ty": "A", "x": "Optional[int64]", "y": "int64"}',
        '{"ty": "B", "p": "Optional[int64]", "q": "int64"}',
        {'x': 'p', 'y': 'q'},
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        None,
    ),
    (
        "1-1",
        '{"ty": "A", "x": "Optional[int64]", "y": "int64"}',
        '{"ty": "B", "p": "Optional[int64]"}',
        {'x': 'p',},
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        None,
    ),
    (
        "2",
        '{"ty": "A", "x": "Optional[int64]", "y": "int64"}',
        '{"ty": "B", "p": "Optional[int64]", "q": "int64", "r": "int64"}',
        {'x': 'p', 'y': 'q'},
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        CreateDfErrTy.RenameMapNotSchemaComplete,
    ),
    (
        "3",
        '{"ty": "A", "x": "Optional[int64]", "y": "int64"}',
        '{"ty": "B", "p": "Optional[int64]", "q": "int64"}',
        {'x': 'z', 'y': 'q'},
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        CreateDfErrTy.RenameMapNamesInvalidColumn,
    ),
    (
        "4",
        '{"ty": "A", "x": "Optional[int64]", "y": "int64"}',
        '{"ty": "B", "p": "int64", "q": "int64"}',
        {'x': 'p', 'y': 'q'},
        [0, 1, 2, None, 4],
        [1, 2, 3, 4, 5],
        CreateDfErrTy.SchemaMapUnequalTys,
    ),
])
def test_mapping_schemas_for_df(
    _: str,
    schema: str, 
    new_schema: str, 
    col_to_new_col: Dict[str, str],
    xs: List[Any], 
    ys: List[Any], 
    opt_error_code: Any):

    df_json_schema = (
        DfJsonSchema
            .from_str(schema)
            .unwrap()
    )

    new_df_json_schema = (
        DfJsonSchema
            .from_str(new_schema)
            .unwrap()
    )

    df = (
        Df
            .from_schema_and_data(
                df_json_schema,
                {
                    "x": xs,
                    "y": ys,
                }
            )
            .unwrap()
    )

    res = df.map_schema(new_df_json_schema, col_to_new_col)

    if opt_error_code is not None:
        assert res.unwrap_err()[0] == opt_error_code
    else:
        new_df = res.unwrap()
        assert len(new_df.df.columns) == len(new_df_json_schema.cols)