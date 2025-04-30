from dataclasses import dataclass
import math
from typing import Any, Callable, List, Optional, Protocol, Self, Tuple, Type, TypeVar, Generic, Union, cast
import checkpipe as pipe
import sympy as sp # type: ignore

T = TypeVar('T')
CT = TypeVar('CT', covariant=True)

class TrOrd(Protocol):
    def ord(self, right: Self) -> int:
        """
        Possible values are <(-1) ==(0) >(1)
        """
    
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

class TrIntoInt(Protocol):
    def into_int(self) -> int:
        """
        """

class TrFromInt(Protocol):
    @classmethod
    def from_int(cls: 'TrFromInt', n: int) -> 'TrFromInt':
        """ """

class TrIntoFloat(Protocol):
    def into_float(self) -> float:
        """ """

class TrFromFloat(Protocol):
    @classmethod
    def from_float(cls: 'TrFromFloat', f: float) -> 'TrFromFloat':
        """ """

class TrIntoTup2Int(Protocol):
    def into_tup2_int(self) -> Tuple[int, int]:
        """ """

class TrFromTup2Int(Protocol):
    @staticmethod
    def from_tup2_int(cls: 'TrFromTup2Int', tup: Tuple[int, int]) -> 'TrFromTup2Int':
        """ """

class TrIntoTup2Float(Protocol):
    def into_tup2_float(self) -> Tuple[float, float]:
        """ """

class TrIntoTup2x2Int(Protocol):
    def into_tup2x2_int(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """ """

class TrIntoExpr():
    def into_expr(self) -> sp.Expr:
        """ """

class TrFromExpr(Protocol):
    @classmethod
    def from_expr(cls: 'TrFromExpr', expr: sp.Expr) -> 'TrFromExpr':
        """ """

class TrIntervalT(TrIntoTup2Int, TrOrd):
    pass

IntervalT = TypeVar("IntervalT", bound=TrIntervalT)


class ImplTrIntoInt_ForNat(TrIntoInt):
    def into_int(self: 'Nat') -> int: # type: ignore
        return self.val

class ImplTrIntoInt_ForNatZ(TrIntoInt):
    def into_int(self: 'NatZ') -> int: # type: ignore
        return self.val

class ImplTrIntoInt_ForInt(TrIntoInt):
    def into_int(self: 'Int') -> int: # type: ignore
        return self.val


class ImplTrFromInt_ForNat(TrFromInt):
    @classmethod
    def from_int(cls: 'Nat', n: int) -> 'Nat': # type: ignore
        return Nat.new(n)

class ImplTrFromInt_ForNatZ(TrFromInt):
    @classmethod
    def from_int(cls: 'NatZ', n: int) -> 'NatZ': # type: ignore
        return NatZ.new(n)

class ImplTrFromInt_ForInt(TrFromInt):
    @classmethod
    def from_int(cls: 'Int', n: int) -> 'Int': # type: ignore
        return Int.new(n)


class ImplTrIntoTup2Int_ForRat(TrIntoTup2Int):
    def into_tup2_int(self: 'Rat') -> Tuple[int, int]: # type: ignore
        return (self.m, self.n)

class ImplTrFromTup2Int_ForRat(TrFromTup2Int):
    @staticmethod
    def from_tup2_int(cls: 'Rat', tup: Tuple[int, int]) -> 'Rat': # type: ignore
        return cls.new(tup[0], tup[1])

class ImplTrIntoTup2x2Int_ForInterval(Generic[IntervalT], TrIntoTup2x2Int):
    def into_tup2x2_int(self: 'Interval[IntervalT]') -> Tuple[Tuple[int, int], Tuple[int, int]]: # type: ignore
        return (
            self.m.into_tup2_int(), 
            self.n.into_tup2_int(),
        )

class ImplTrIntoFloat_ForRat(TrIntoFloat):
    def into_float(self: 'Rat') -> float: # type: ignore
        return float(self.m) /  float(self.n)

class ImplTrFromFloat_ForRat(TrFromFloat):
    @classmethod
    def from_float(cls, f: float) -> 'Rat': # type: ignore
        rat = sp.Rational(f)
        return cast(Rat, cls).new(rat.numerator, rat.denominator)

NatLike = TypeVar('NatLike', bound='TrIntoInt')
FromNatLike = TypeVar('FromNatLike', bound='TrFromInt')

class ImplTrIntoExpr_ForNatLike(Generic[NatLike], TrIntoExpr):
    def into_expr(self: NatLike) -> sp.Expr: # type: ignore
        return sp.sympify(self.into_int())

class ImplTrIntoExpr_ForRat(TrIntoExpr):
    def into_expr(self: 'Rat') -> sp.Expr: # type: ignore
        (n, m) = self.into_tup2_int()
        return sp.sympify(f'{n}/{m}')

class ImplTrFromExpr_ForNatLike(Generic[FromNatLike], TrFromExpr):
    @classmethod
    def from_expr(cls: FromNatLike, expr: sp.Expr) -> FromNatLike: # type: ignore
        f = expr.evalf()
        assert f - int(f) == 0

        return cls.from_int(int(f)) # type: ignore

class ImplTrFromExpr_ForRat(TrFromExpr):
    @classmethod
    def from_expr(cls: 'Rat', expr: sp.Expr) -> 'Rat': # type: ignore
        def eval_numerator_denominator() -> Tuple[float, float]:
            tup: Tuple[float, float] = (
                expr.as_numer_denom()
                    | pipe.Of[Tuple[sp.Expr, sp.Expr]]
                    .map(pipe.tup2_unpack(lambda numerator, denominator:
                        (
                            numerator.evalf(),
                            denominator.evalf(),
                        )
                    ))
            )

            return tup

        numerator, denominator = eval_numerator_denominator()

        assert(numerator - int(numerator) == 0)
        assert(denominator - int(denominator) == 0)

        return cls(int(numerator), int(denominator)) # type: ignore



class ImplTrOrd_ForNatLike(Generic[NatLike], TrOrd):
    def ord(self: NatLike, right: NatLike) -> int: # type: ignore
        return (
            -1 if self.into_int() < right.into_int() else
             1 if self.into_int() > right.into_int() else
             0
        )

class ImplTrOrd_ForRat(TrOrd):
    def ord(self: 'Rat', right: 'Rat') -> int: # type: ignore
        (n, m) = (
            self.into_float(),
            right.into_float(),
        )

        return (
            -1 if n < m else
             1 if n > m else
             0
        )

@dataclass
class Nat(
    ImplTrIntoInt_ForNat, 
    ImplTrFromInt_ForNat,
    ImplTrIntoExpr_ForNatLike['Nat'], 
    ImplTrFromExpr_ForNatLike['Nat'],
    ImplTrOrd_ForNatLike['Nat'],
):

    val: int

    @staticmethod
    def new(n: int) -> 'Nat':
        assert n > 0, f"Nat must be positive. {n} is not."
        return Nat(n)
    

@dataclass
class NatZ(
    ImplTrIntoInt_ForNatZ, 
    ImplTrFromInt_ForNatZ,
    ImplTrIntoExpr_ForNatLike['NatZ'], 
    ImplTrFromExpr_ForNatLike['NatZ'],
    ImplTrOrd_ForNatLike['NatZ'],
):
    val: int

    @staticmethod
    def new(n: int) -> 'NatZ':
        assert n >= 0, f"NatZ must be non-negative. {n} is not."
        return NatZ(n)
    

@dataclass
class Int(
    ImplTrIntoInt_ForInt, 
    ImplTrFromInt_ForInt,
    ImplTrIntoExpr_ForNatLike['Int'], 
    ImplTrFromExpr_ForNatLike['Int'],
    ImplTrOrd_ForNatLike['Int'],
):
    val: int

    @staticmethod
    def new(n: int) -> 'Int':
        return Int(n)
    

@dataclass
class Rat(
    ImplTrIntoTup2Int_ForRat, 
    ImplTrFromTup2Int_ForRat,
    ImplTrIntoFloat_ForRat, 
    ImplTrFromFloat_ForRat,
    ImplTrIntoExpr_ForRat,
    ImplTrFromExpr_ForRat,
    ImplTrOrd_ForRat,
):
    m: int
    n: int

    @staticmethod
    def new(m: int, n: int) -> 'Rat':
        assert(n != 0)
        return Rat(m, n)

@dataclass
class Interval(
    Generic[IntervalT], 
    ImplTrIntoTup2x2Int_ForInterval[IntervalT],
):
    m: IntervalT
    n: IntervalT

    @staticmethod
    def new(m: IntervalT, n: IntervalT) -> 'Interval[IntervalT]':
        assert(m <= n)
        return Interval[IntervalT](m, n)


class TrFitsEquation(Protocol):
    def fits_equation(self, variables: List[Any]) -> Optional[bool]:
        """
        True if substituting the variables yields symmetry, false if it yields asymmetry, and 
        None if fitting fails
        """

class ImplTrFitsEquation_ForProj3(TrFitsEquation):
    def fits_equation(self: 'Proj3', variables: List[Rat]) -> Optional[bool]: # type: ignore
        if len(variables) != 3:
            return None
        


@dataclass
class Proj3:
    a: Rat
    b: Rat
    c: Rat

    @staticmethod
    def new(a: Rat, b: Rat, c: Rat) -> 'Proj3':
        assert a.into_float() != 0 or b.into_float() != 0

        return Proj3(a, b, c)


SymFnT = TypeVar("SymFnT", bound=Union[TrIntoInt, TrIntoTup2Int])
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
    return l


def square_fn() -> SymFn1[Rat, Rat]:
    fn_expr = lambda x: x**2

    fn_eval = lambda x: Rat.from_float(cast(float, x.into_float())**2)

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