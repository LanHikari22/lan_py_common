import pytest
import pandas as pd
from lan_py_common.df import Df, DfJsonSchema, CreateDfErrTy

def test_map_schema_happy_path():
    df = (
        Df.from_schema_and_data(
            DfJsonSchema
                .from_str('{"x": "Optional[int64]", "y": "int64" }')
                .unwrap(),
            {
                "x": [0, 1, 2, None, 4],
                "y": [1, 2, 3, 4, 5],
            }
        )
            .unwrap()
    )

    print(df.schema)
    print(df.df)
    assert False