"""
Do a quick, sequential, numerical (not symbolic) exploration of some electronic
component values to propose solutions that use standard, inexpensive parts.
"""


from bisect import bisect_left
from itertools import islice
from math import log10, floor
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple, 
)


# See https://en.wikipedia.org/wiki/E_series_of_preferred_numbers
E3 = (1.0, 2.2, 4.7)

E6 = (1.0, 1.5, 2.2, 3.3, 4.7, 6.8)

E12 = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)

E24 = (
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
)

E48 = (
    1.00, 1.05, 1.10, 1.15, 1.21, 1.27, 1.33, 1.40, 1.47, 1.54, 1.62, 1.69,
    1.78, 1.87, 1.96, 2.05, 2.15, 2.26, 2.37, 2.49, 2.61, 2.74, 2.87, 3.01,
    3.16, 3.32, 3.48, 3.65, 3.83, 4.02, 4.22, 4.42, 4.64, 4.87, 5.11, 5.36,
    5.62, 5.90, 6.19, 6.49, 6.81, 7.15, 7.50, 7.87, 8.25, 8.66, 9.09, 9.53,
)

E96 = (
    1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30,
    1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74,
    1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21, 2.26, 2.32,
    2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09,
    3.16, 3.24, 3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
    4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23, 5.36, 5.49,
    5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32,
    7.50, 7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
)


def bisect_lower(a: Sequence[float], x: float) -> int:
    """
    Run bisect, but use one index before the return value of `bisect_left`
    @param a The sorted haystack
    @param x The needle
    @return The index of the array element that equals or is lesser than `x`
    """
    i = bisect_left(a, x)
    if (
        (i < len(a) and a[i] > x) or
        (i >= len(a) and a[i % len(a)]*10 > x)
    ):
        i -= 1
    return i


def approximate(x: float, series: Sequence[float]) -> (int, float):
    """
    Approximate a value by using the given series.
    @param x Any positive value
    @param series Any of E3 through E96
    @return An integer index into the series for the element lesser than or
             equal to the value's mantissa, and the value's decade - a power of
             ten
    """
    if x == float('inf'):
        return None, float('inf')

    decade = 10**floor(log10(x))
    mantissa = x / decade
    index = bisect_lower(series, mantissa)
    if index >= len(series):
        return 0, decade * 10
    return index, decade


def fmt_eng(x: float, unit: str, sig: int = 2) -> str:
    """
    Format a number in engineering (SI) notation
    @param x Any number
    @param unit The quantity unit (Hz, A, etc.)
    @param sig Number of significant digits to show
    @return The formatted string
    """
    if x == 0:
        p = 0
    elif x == float('inf'):
        return '∞'
    else:
        p = floor(log10(abs(x)))
    e = int(floor(p / 3))
    digs = max(0, sig - p%3 - 1)
    mantissa = x / 10**(3*e)

    if e == 0:
        prefix = ''
    elif 0 < e < 9:
        # See https://en.wikipedia.org/wiki/Metric_prefix
        prefix = ' kMGTPEZY'[e]
    elif 0 > e > -8:
        prefix = 'mμnpfazy'[-e-1]
    else:
        raise IndexError(f'Number out of SI range: {x:.1e}')

    fmt = '{:.%df} {:}{:}' % digs
    return fmt.format(mantissa, prefix, unit)


# a callable with any number of floating-point
# arguments, returning a float
class CalculateCall(Protocol):
    def __call__(self, *args: float) -> float:
        ...


class ComponentValue:
    """
    A value associated with a component - to track approximated values
    """

    def __init__(
        self,
        component: 'Component',
        decade: Optional[float] = None,
        index: Optional[int] = None,
        exact: Optional[float] = None,
    ):
        """
        Valid combinations:
          - exact - approximated value will be calculated
          - exact, index, decade - approximated value = series[index]*decade
          - index, decade - approximated value = series[index]*decade;
                            exact=approximate

        @param decade The quantity's power-of-ten
        @param index The integer index into the series for the quantity's
                      mantissa
        @param exact The exact quantity, if known
        """

        self.component = component

        if index is None:
            assert decade is None
            assert exact is not None
            self.exact = exact
            self.index, self.decade = approximate(exact, component.series)
        else:
            assert decade is not None
            self.index, self.decade = index, decade

        if self.decade == float('inf'):
            self.approx = float('inf')
        else:
            self.approx = component.series[self.index] * self.decade

        if index is not None:
            if exact is None:
                self.exact = self.approx
            else:
                self.exact = exact

    @property
    def error(self) -> float:
        return self.approx / self.exact - 1

    def get_other(self) -> Optional['ComponentValue']:
        """
        @return: If this approximated value is below its exact value, then the
                 next-highest E24 value; otherwise None
        """
        if self.approx >= self.exact:
            return None

        index, decade = self.index + 1, self.decade
        if index >= len(self.component.series):
            index = 0
            decade *= 10
        return ComponentValue(component=self.component, exact=self.exact,
                              index=index, decade=decade)

    def get_best(self) -> 'ComponentValue':
        other = self.get_other()
        if other is None:
            return self

        if self.error**2 < other.error**2:
            return self
        return other

    def __str__(self):
        return fmt_eng(self.approx, self.component.unit, self.component.digits)

    def fmt_exact(self) -> str:
        return fmt_eng(self.exact, self.component.unit, 4)


class Component:
    """
    A component, without knowledge of its value - only bounds and defining
    formula
    """

    def __init__(
        self,
        prefix: str,
        suffix: str,
        unit: str,
        series: Sequence[float] = E24,
        calculate: Optional[CalculateCall] = None,
        minimum: float = 0,
        maximum: Optional[float] = None,
        use_for_err: bool = True,
    ):
        """
        @param prefix i.e. R, C or L
        @param suffix Typically a number, i.e. the "2" in R2
        @param unit i.e. Hz, A, F, ...
        @param series One of E3 through E96
        @param calculate A callable that will be given all values of previous
                         components in the calculation sequence. These values
                         are floats, and the return must be a float.
                         If this callable is None, the component will be
                         interpreted as a degree of freedom.
        @param minimum Min allowable value; the return of calculate will be
                        checked against this and failures will be silently
                        dropped.
                        Must be at least zero, or greater than zero if
                        calculate is not None.
        @param maximum Max allowable value; the return of calculate will be
                        checked against this and failures will be silently
                        dropped.
        @param use_for_err If True, error from this component's ideal to
                            approximated value will influence the solution rank.
        """
        (
            self.prefix, self.suffix, self.unit, self.series,
            self.calculate, self.min, self.max, self.use_for_err,
        ) = (
            prefix, suffix, unit, series, calculate, minimum, maximum,
            use_for_err,
        )

        assert minimum >= 0
        assert maximum is None or maximum >= minimum

        if calculate:
            self.values = self._calculate_values
        elif minimum > 0:
            self.start_index, self.start_decade = approximate(minimum, series)
            self.values = self._iter_values

        self.digits: int = 3 if len(series) > 24 else 2

        self.fmt_field: Callable[[str], str] = (
            ('{:>%d}' % (4 + self.digits)).format
        )

    def __str__(self):
        return self.name

    @property
    def name(self) -> str:
        return f'{self.prefix}{self.suffix}'

    def _calculate_values(
        self, prev: Sequence[ComponentValue]
    ) -> Iterable[ComponentValue]:

        def values():
            # Get the value based on exact values first
            from_exact_val = self.calculate(*(p.exact for p in prev))
            if from_exact_val <= 0:
                return

            from_exact = ComponentValue(self, exact=from_exact_val)
            yield from_exact
            other = from_exact.get_other()
            if other:
                yield other

            # See if there's a difference when calculating against approximated
            # values
            from_approx_val = self.calculate(*(p.approx for p in prev))
            if from_approx_val > 0:
                from_approx = ComponentValue(self, exact=from_approx_val)
                if from_approx.exact != from_exact.exact:
                    yield from_approx
                    other = from_approx.get_other()
                    if other:
                        yield other

        for v in values():
            if (
                self.min <= v.exact and
                (self.max is None or self.max >= v.exact)
            ):
                yield v

    def _all_values(self) -> Iterable[Tuple[int, float]]:
        decade = self.start_decade
        for index in range(self.start_index, len(self.series)):
            yield index, decade
        while True:
            decade *= 10
            for index in range(len(self.series)):
                yield index, decade

    def _iter_values(
        self, prev: Sequence[ComponentValue],
    ) -> Iterable[ComponentValue]:
        for index, decade in self._all_values():
            value = ComponentValue(self, index=index, decade=decade)
            if value.approx > self.max:
                return
            yield value


class Resistor(Component):
    def __init__(
        self,
        suffix: str,
        series: Sequence[float] = E24,
        calculate: Optional[CalculateCall] = None,
        minimum: float = 0,
        maximum: Optional[float] = None,
        use_for_err: bool = False,
    ):
        super().__init__('R', suffix, 'Ω', series, calculate, minimum, maximum,
                         use_for_err)


class Capacitor(Component):
    def __init__(
        self,
        suffix: str,
        series: Sequence[float] = E24,
        calculate: Optional[CalculateCall] = None,
        minimum: float = 0,
        maximum: Optional[float] = None,
        use_for_err: bool = True,
    ):
        super().__init__('C', suffix, 'F', series, calculate, minimum, maximum,
                         use_for_err)


class Output:
    """
    A calculated parameter - potentially but not necessarily a circuit output -
    to be calculated and checked for error in the solution ranking process.
    """

    def __init__(
        self, name: str, unit: str, expected: float,
        calculate: CalculateCall,
    ):
        """
        @param name i.e. Vout
        @param unit i.e. V, A, Hz...
        @param expected The value that this parameter would assume under ideal
                         circumstances
        @param calculate A callable accepting a sequence of floats - one per
                          component, in the same order as they were passed to
                          the Solver constructor; returning a float.
        """
        self.name, self.unit, self.expected, self.calculate = (
            name, unit, expected, calculate,
        )

    def error(self, value: float) -> float:
        """
        @return Absolute error, since the expected value might be 0
        """
        return value - self.expected

    def __str__(self):
        return self.name


class Solver:
    """
    Basic recursive solver class that does a brute-force search through some
    component values.
    """

    def __init__(
        self,
        components: Sequence[Component],
        outputs: Sequence[Output],
        threshold: Optional[float] = 1e-3,
    ):
        """
        @param components A sequence of Component instances. The order of this
                          sequence determines the order of parameters passed to
                          Output.calculate and Component.calculate.
        @param outputs A sequence of Output instances - can be empty.
        @param threshold Maximum error above which solutions will be discarded
        """
        self.components, self.outputs = components, outputs
        self.candidates: List[Tuple[
            float,                     # error
            Sequence[float],           # output values
            Sequence[ComponentValue],  # component values to get the above
        ]] = []
        self.approx_seen: Set[Tuple[float, ...]] = set()
        self.threshold = threshold

    def _recurse(self, values: List[Optional[ComponentValue]], index: int = 0):
        if index >= len(self.components):
            self._evaluate(values)
        else:
            comp = self.components[index]
            for v in comp.values(values[:index]):
                values[index] = v
                self._recurse(values, index+1)

    def solve(self):
        """
        Recurse through all of the components, doing a brute-force search.
        Results are stored in self.candidates and sorted in order of increasing
        error.
        """
        values = [None]*len(self.components)
        self._recurse(values)
        self.candidates.sort(key=lambda v: v[0])

    def _evaluate(self, values: Sequence[ComponentValue]):
        approx = tuple(v.approx for v in values)
        if approx in self.approx_seen:
            return

        outputs = tuple(
            o.calculate(*approx)
            for o in self.outputs
        )
        err = sum(
            o.error(v)**2
            for o, v in zip(self.outputs, outputs)
        ) + sum(
            v.error**2
            for c, v in zip(self.components, values)
            if c.use_for_err
        )
        if self.threshold is None or err < self.threshold:
            self.candidates.append((err, outputs, tuple(values)))
            self.approx_seen.add(approx)

    def print(self, top: int = 10):
        """
        Print a table of all component values, output values and output error.
        @param top Row limit.
        """

        print(' '.join(
            comp.fmt_field(comp.name)
            for comp in self.components
        ), end=' ')
        print(' '.join(
            f'{output.name:>10} {"Err":>8}'
            for output in self.outputs
        ))

        for err, outputs, values in islice(self.candidates, top):
            print(' '.join(
                value.component.fmt_field(str(value))
                for value in values
            ), end=' ')
            print(' '.join(
                f'{fmt_eng(value, output.unit, 4):>10} '
                f'{output.error(value):>8.1e}'
                for value, output in zip(outputs, self.outputs)
            ))
