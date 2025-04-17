from .df import *

def add(a: int, b: int) -> int:
    return a + b

def test_df():
    df = Df.new("")
    print(describe(5))
    print(add('a', 'b'))

if __name__ == '__main__':
    print('Test!')
    test_df()