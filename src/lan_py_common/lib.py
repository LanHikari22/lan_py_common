from .df import *
from .plotting import *
from .common import *

def add(a: int, b: int) -> int:
    return a + b

def test_df() -> None:
    df = (
        Df.from_schema_and_data(
            DfJsonSchema
                .from_str('{"ty": "SchA", "x": "Optional[int64]", "y": "int64" }')
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

if __name__ == '__main__':
    print('Test!')
    test_df()