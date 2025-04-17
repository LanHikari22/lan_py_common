import math
from typing import Any, Callable, List, Self, Tuple, Type, TypeVar, Generic, Union
import checkpipe as pipe
import sympy as sp # type: ignore

class Ord():
    def ord(self, right: Self) -> int:
        raise NotImplementedError()
    
    def __lt__(self, right: Self) -> bool:
        return (
            self.ord(right) == -1
        )

    def __le__(self, right: Self) -> bool:
        ord = (
            self.ord(right)
        )
        return (
            ord == -1 or ord == 0
        )

    def __eq__(self, right: Any) -> bool:
        ord = (
            self.ord(right)
        )
        return (
            ord == 0
        )

    def __gt__(self, right: Self) -> bool:
        ord = (
            self.ord(right)
        )
        return (
            ord == 1
        )

    def __ge__(self, right: Self) -> bool:
        ord = (
            self.ord(right)
        )
        return (
            ord == 0 or ord == 1
        )

class IntoInt():
    def into_int(self) -> int:
        raise NotImplementedError()

class IntoFloat():
    def into_float(self) -> float:
        raise NotImplementedError()

FromFloatT = TypeVar("FromFloatT", bound="FromFloat") # type: ignore
class FromFloat(Generic[FromFloatT]):
    @classmethod
    def from_float(cls: Type[FromFloatT], f: float) -> FromFloatT:
        raise NotImplementedError()

class IntoTup2Int():
    def into_tup2_int(self) -> Tuple[int, int]:
        raise NotImplementedError()

class IntoTup2Float():
    def into_tup2_float(self) -> Tuple[float, float]:
        raise NotImplementedError()

class IntoTup2x2Int():
    def into_tup2x2_int(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        raise NotImplementedError()

class IntoExpr():
    def into_expr(self) -> sp.Expr:
        raise NotImplementedError()

FromT = TypeVar("FromT", bound="FromExpr") # type: ignore
class FromExpr(Generic[FromT]):
    @classmethod
    def from_expr(cls: Type[FromT], expr: sp.Expr) -> FromT:
        raise NotImplementedError()


class Nat(IntoInt, IntoExpr, FromExpr["Nat"], Ord):
    def __init__(self, n: int):
        assert n > 0, f"Nat must be positive. {n} is not."
        self.val = n
    
    # Implements IntoInt
    def into_int(self) -> int:
        return (
            self.val
        )
    
    # Implements Ord
    def ord(self, right: Self) -> int:
        return (
            -1 if self.val < right.val else
             1 if self.val > right.val else
             0
        )

    # Implements IntoExpr
    def into_expr(self) -> sp.Expr:
        return (
            sp.sympify(self.into_int())
        )
    
    # Implements FromExpr
    @classmethod
    def from_expr(cls: Type[Self], expr: sp.Expr) -> Self:
        f = (
            expr.evalf()
        )

        assert f - int(f) == 0

        return (
            cls(int(f))
        )

class NatZ(IntoInt, IntoExpr, FromExpr["NatZ"], Ord):
    def __init__(self, n: int):
        assert n >= 0, f"NatZ must be non-zero. {n} is not."
        self.val = n
    
    # Implements IntoInt
    def into_int(self) -> int:
        return (
            self.val
        )
    
    # Implements Ord
    def ord(self, right: Self) -> int:
        return (
            -1 if self.val < right.val else
             1 if self.val > right.val else
             0
        )

    # Implements IntoExpr
    def into_expr(self) -> sp.Expr:
        return (
            sp.sympify(self.into_int())
        )
    
    # Implements FromExpr
    @classmethod
    def from_expr(cls: Type[Self], expr: sp.Expr) -> Self:
        f = (
            expr.evalf()
        )

        assert(f - int(f) == 0)

        return (
            cls(int(f))
        )


class Int(IntoInt, IntoExpr, FromExpr["Int"], Ord):
    def __init__(self, n: int):
        self.val = n
    
    # Implements IntoInt
    def into_int(self) -> int:
        return (
            self.val
        )

    # Implements IntoExpr
    def into_expr(self) -> sp.Expr:
        return (
            sp.sympify(self.into_int())
        )

    # Implements FromExpr
    @classmethod
    def from_expr(cls: Type[Self], expr: sp.Expr) -> Self:
        f = (
            expr.evalf()
        )

        assert(f - int(f) == 0)

        return (
            cls(int(f))
        )

    # Implements Ord
    def ord(self, right: Self) -> int:
        return (
            -1 if self.val < right.val else
             1 if self.val > right.val else
             0
        )

class Rat(IntoTup2Int, IntoFloat, FromFloat["Rat"], IntoExpr, FromExpr["Rat"], Ord):
    def __init__(self, n: int , m: int):
        assert(m != 0)

        self.n = n
        self.m = m
    
    # Implements IntoTup2Int
    def into_tup2_int(self) -> Tuple[int, int]:
        return (
            (self.n, self.m)
        )

    # Implements IntoFloat
    def into_float(self) -> float:
        return (
            float(self.n) /  float(self.m)
        )
    
    # Implements FromFloat
    @classmethod
    def from_float(cls: Type[Self], f: float) -> Self:
        # sp.Rational(sp.sympify("2**(1/12)").evalf())

        rat = (
            sp.Rational(f)
        )

        return (
            cls(rat.numerator, rat.denominator)
        )

    # Implements IntoExpr
    def into_expr(self) -> sp.Expr:
        (n, m) = (
            self.into_tup2_int()
        )
        return (
            sp.sympify(f'{n}/{m}')
        )

    # Implements FromExpr
    @classmethod
    def from_expr(cls: Type[Self], expr: sp.Expr) -> Self:
        def eval_numerator_denominator() -> Tuple[float, float]:
            tup: Tuple[float, float] = (
                expr.as_numer_denom()
                    | pipe.Of[Tuple[sp.Expr, sp.Expr]].to(pipe.tup2_unpack(lambda numerator, denominator:
                        (
                            numerator.evalf(),
                            denominator.evalf(),
                        )
                    ))
            )

            return (
                tup
            )

        (numerator, denominator) = (
            eval_numerator_denominator()
        )

        assert(numerator - int(numerator) == 0)
        assert(denominator - int(denominator) == 0)

        return (
            cls(int(numerator), int(denominator))
        )


    # Implements Ord
    def ord(self, right: Self) -> int:
        (n, m) = (
            (
                self.into_float(),
                right.into_float(),
            )
        )

        return (
            -1 if n < m else
             1 if n > m else
             0
        )

class IntoIntAndOrd(IntoInt, Ord):
    pass

IntervalT = TypeVar("IntervalT", bound=IntoIntAndOrd)
class Interval(Generic[IntervalT], IntoTup2Int):
    def __init__(self, n: IntervalT, m: IntervalT):
        assert(n <= m)

        self.n = n
        self.m = m

    # Implements IntoTup2Int
    def into_tup2_int(self) -> Tuple[int, int]:
        return (
            (self.n.into_int(), self.m.into_int())
        )

class IntervalRat(IntoTup2Float,IntoTup2x2Int):
    def __init__(self, n: Rat, m: Rat):
        assert(n <= m)

        self.n = n
        self.m = m

    # Implements IntoTup2x2Int
    def into_tup2x2_int(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            ((self.n.into_tup2_int(), self.m.into_tup2_int()))
        )


SymFnT = TypeVar("SymFnT", bound=Union[IntoInt, IntoTup2Int])
SymFnU = TypeVar("SymFnU")
class SymFn1(Generic[SymFnT, SymFnU]):
    def __init__(self, fn_expr: Callable[[sp.Expr], sp.Expr], fn_eval: Callable[[SymFnT], SymFnU]):
        self.fn_expr = fn_expr
        self.fn_eval = fn_eval
    
    def expr(self, arg1: sp.Expr) -> sp.Expr:
        return (
            self.fn_expr(arg1)
        )

    def eval(self, arg1: SymFnT) -> SymFnU:
        return (
            self.fn_eval(arg1)
        )

ListT = TypeVar("ListT")
def list_append(l: List[ListT], v: ListT) -> List[ListT]:
    l.append(v)

    return (
        l
    )


def square_fn() -> SymFn1[Rat, Rat]:
    fn_expr = (
        lambda x: x**2
    )

    fn_eval = (
        lambda x: Rat.from_float(x.into_float()**2)
    )

    return (
        SymFn1[Rat, Rat](fn_expr, fn_eval)
    )

def nth_power(n: int) -> SymFn1[Rat, Rat]:
    fn_expr = (
        lambda x: x**n
    )

    fn_eval = (
        lambda x: Rat.from_float(x.into_float()**n)
    )

    return (
        SymFn1[Rat, Rat](fn_expr, fn_eval)
    )

def sin_fn() -> SymFn1[Rat, Rat]:
    fn_expr = (
        lambda x: sp.sin(x)
    )
    fn_eval = (
        lambda x: Rat.from_float(math.sin(x.into_float()))
    )

    return (
        SymFn1[Rat, Rat](fn_expr, fn_eval)
    )

def exp_fn() -> SymFn1[Rat, Rat]:
    fn_expr = (
        lambda x: sp.exp(x)
    )
    fn_eval = (
        lambda x: Rat.from_float(math.exp(x.into_float()))
    )

    return (
        SymFn1[Rat, Rat](fn_expr, fn_eval)
    )

def taylor_series_expansion(fn: SymFn1[Rat, Rat], num_terms: int, debug: bool=False) -> SymFn1[Rat, Rat]:
    x: sp.Expr = (
        sp.Symbol("x")
    )

    polynomial: sp.Expr = (
        sp.series(fn.expr(x), x, 0, num_terms).removeO()
    )

    if debug:
        print(sp.latex(polynomial))

    fn_expr = (
        lambda arg: polynomial.replace(x, arg)
    )

    fn_eval = (
        lambda arg: Rat.from_float(polynomial.replace(x, arg.into_expr()).evalf())
    )

    return (
        SymFn1[Rat, Rat](fn_expr, fn_eval)
    )

def syms(s: str) -> List[sp.Expr]:
    symbols = (
        s.split(' ')
    )

    return (
        range(0, len(symbols))
            | pipe.OfIter[int].map(lambda i: 
                sp.symbols(symbols[i])
            )
            | pipe.OfIter[int].to_list()
    )