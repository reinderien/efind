from efind import (
    Solver, Output, Capacitor, Resistor, ComponentValue, fmt_eng,
    E12, E24, E96,
)


def opamp():
    # https://electronics.stackexchange.com/a/562046/10008
    # Op-amp to scale 4.5-6.5V -> 3-10V

    Vref = Vcc = 12

    def Vo(Vi: float, R1: float, R3: float, R4: float) -> float:
        I3 = (Vref - Vi) / R3
        I4 = Vi / R4
        I1 = I3 - I4
        V1 = I1 * R1
        return Vi - V1

    def Vol(R1: float, R3: float, R4: float) -> float:
        return Vo(4.5, R1, R3, R4)
    def Voh(R1: float, R3: float, R4: float) -> float:
        return Vo(6.5, R1, R3, R4)

    svout = Solver(
        components=(
            Resistor(
                suffix='1', series=E96, minimum=50e3, maximum=500e3,
            ),
            Resistor(
                suffix='3', series=E96, calculate=lambda R1: R1*16/17,
            ),
            Resistor(
                suffix='4', series=E96, calculate=lambda R1, R3: R1*16/23,
            ),
        ),
        outputs=(
            Output('Vol', unit='V', expected=3, calculate=Vol),
            Output('Voh', unit='V', expected=10, calculate=Voh),
        ),
        threshold=1e-2,
    )

    svout.solve()
    svout.print()


def opamp2():
    # https://electronics.stackexchange.com/questions/564165
    # Op-amp to scale 3.4-4.1V -> 1-5V

    Vref = 4.096  # or maybe 12V
    Vi1, Vo1 = 3.4, 1
    Vi2, Vo2 = 4.1, 5
    b = (
        ((Vref - Vi1)*(Vi2 - Vo2) - (Vref - Vi2)*(Vi1 - Vo1)) /
        ((Vref - Vi2)*Vi1 - (Vref - Vi1)*Vi2)
    )
    a = (Vi1*(b + 1) - Vo1)/(Vref - Vi1)

    def Vo(Vi: float, R3: float, R2: float, R1: float) -> float:
        I1 = (Vref - Vi)/R1
        I2 = Vi/R2
        I3 = I1 - I2
        V3 = I3*R3
        vo = Vi - V3
        return vo

    def Vol(R3: float, R2: float, R1: float) -> float:
        return Vo(3.4, R3, R2, R1)

    def Voh(R3: float, R2: float, R1: float) -> float:
        return Vo(4.1, R3, R2, R1)

    svout = Solver(
        components=(
            Resistor(
                suffix='3', series=E96, minimum=50e3, maximum=500e3,
            ),
            Resistor(
                suffix='2', series=E96, calculate=lambda R3: R3/b,
            ),
            Resistor(
                suffix='1', series=E96, calculate=lambda R3, R2: R3/a,
            ),
        ),
        outputs=(
            Output('Vol', unit='V', expected=1, calculate=Vol),
            Output('Voh', unit='V', expected=5, calculate=Voh),
        ),
        threshold=1e-2,
    )

    svout.solve()
    svout.print()


def opamp3():
    # https://electronics.stackexchange.com/questions/573296
    # Op-amp level shifter, 0-3.3V -> 0 - -7V with hysteresis
    # Account for common-mode-friendly input symmetric within Vss
    Vdd = 0  # or 3.3 or 5
    Vss = -7
    Vih = 3.3
    Vi1, Vi2 = 0.2*Vih, 0.8*Vih
    gain = -Vss/(Vi2 - Vi1)
    offset = -gain*Vi2
    amin = -Vih/Vss  # to avoid clipping
    a = R1R2 = 1 - Vih/Vss   # to center the input wave on Vss/2
    b = R4R3 = (gain + offset/Vss - 1) / (1 - Vdd/Vss)

    def Vi(Vo: float, R1: float, R2: float, R3: float, R4: float, R5: float) -> float:
        R3pR4 = 1 / (1/R3 + 1/R4)
        R3pR5 = 1 / (1/R3 + 1/R5)
        R4pR5 = 1 / (1/R4 + 1/R5)
        Vp_Vdd = Vdd * R4pR5 / (R4pR5 + R3)
        Vp_Vo = Vo * R3pR5 / (R3pR5 + R4)
        Vp_Vss = Vss * R3pR4 / (R3pR4 + R5)
        Vp = Vp_Vdd + Vp_Vo + Vp_Vss
        I1 = (Vp - Vss)/R2
        return Vp + I1*R1

    def Vilact(*args: float) -> float:
        return Vi(Vss, *args)

    def Vihact(*args: float) -> float:
        return Vi(0, *args)

    def Vpmean(R1: float, R2: float, R3: float, R4: float, R5: float) -> float:
        Vpmax = (Vih - Vss) / (1 + R1/R2) + Vss
        Vpmin = (0 - Vss) / (1 + R1/R2) + Vss
        return (Vpmax + Vpmin) / 2

    def getR5(R1: float, R2: float, R3: float, R4: float) -> float:
        c = R4R5 = gain * (1 + R1/R2) - R4/R3 - 1
        return R4/R4R5

    svout = Solver(
        components=(
            Resistor(
                suffix='1', series=E24, minimum=10e3, maximum=100e3,
            ),
            Resistor(
                suffix='2', series=E24, calculate=lambda R1: R1/R1R2,
            ),
            Resistor(
                suffix='3', series=E24, minimum=10e3, maximum=100e3,
            ),
            Resistor(
                suffix='4', series=E24, calculate=lambda R1, R2, R3: R3*R4R3,
            ),
            Resistor(
                suffix='5', series=E24, calculate=getR5,
            ),
        ),
        outputs=(
            Output('Vil', unit='V', expected=Vi1, calculate=Vilact),
            Output('Vih', unit='V', expected=Vi2, calculate=Vihact),
            Output('Vpmean', unit='V', expected=Vss/2, calculate=Vpmean),
        ),
        threshold=1e-2,
    )

    svout.solve()
    svout.print()


def buck():
    # https://electronics.stackexchange.com/a/562550/10008
    # Convert down to 3V using the device described in
    # https://fscdn.rohm.com/en/products/databook/datasheet/ic/power/switching_regulator/bd9e302efj-e.pdf
    # page 30

    Vout = 3.0
    Vref = 0.8
    R12max = 700e3
    # 700e3 / R2 * Vref = Vout at limit
    R2max = Vref / Vout * R12max

    svout = Solver(
        components=(
            Resistor(suffix='2', series=E96, minimum=R2max/10, maximum=R2max),
            Resistor(
                suffix='1', series=E96, calculate=lambda R2: R2*(Vout/Vref - 1),
            ),
        ),
        outputs=(
            Output(
                'Vout', unit='V', expected=Vout,
                calculate=lambda R2, R1: Vref*(1 + R1/R2),
            ),
        ),
    )
    svout.solve()
    svout.print()


def complex_smps():
    # A real(ish) SMPS calculation for the AZ34063 converting 24V to 5V

    # For the LRS-100-24
    Vinnom = 24
    loadreg = 5e-3
    loadtol = 1e-2
    Vinmin = Vinnom*(1 - loadreg - loadtol)

    # Following AN920-D Step−Down Switching Regulator Design Example
    # but targeting the AZ34063
    Voutnom = 5
    Vripple = 5e-3 * Voutnom
    fmin = 38e3
    Iout = 0.15
    Ipksw = 2*Iout

    # Vce(sat) for Darlington connection, typ. 1-1.3V from the table.
    # Figure 6 shows closer to 875mV.
    Vsat = 0.875
    Vref = 1.25
    Vdiff = Vinmin - Vsat - Voutnom

    # For the SB140TA. Probably even less than this.
    Vf = 0.3

    tmax = 1 / fmin
    ton_toff = (Voutnom + Vf)/Vdiff
    toff = tmax/(1 + ton_toff)
    ton = tmax - toff
    assert ton/(ton + toff) < 6/7

    Gt = 2.86e-5  # from AZ34063 figure 4
    Ct = Gt*ton
    Comin = Ipksw*tmax/8/Vripple
    Lmin = Vdiff*ton/Ipksw

    Ipkswnom = (Vinnom - Vsat - Voutnom)*ton/Lmin
    Idivmin = 100e-6

    Vsense = 0.3
    Rsc = Vsense/Ipkswnom
    R1max = Vref/Idivmin

    def Vo(R1: float, R2: float) -> float:
        return Vref*(1 + R2/R1)

    svout = Solver(
        components=(
            Resistor(suffix='1', series=E96, minimum=R1max/10, maximum=R1max),
            Resistor(
                suffix='2', series=E96, calculate=lambda R1: R1*(Voutnom/Vref - 1),
            ),
        ),
        outputs=(
            Output('Vout', unit='V', expected=Voutnom, calculate=Vo),
        ),
    )
    svout.solve()

    Ct_approx = ComponentValue(exact=Ct, component=Capacitor('t', E12))
    Rsc_approx = ComponentValue(
        exact=Rsc, component=Resistor('sc', E24)
    ).get_best()

    Rsc2 = Rsc_approx.approx
    _, _, (R12, R22) = svout.candidates[0]
    Co2 = 470e-6
    Ct2 = Ct_approx.approx
    L2 = 470e-6  # the RLB0914-471KL, Rdc < 1.3Ω
    ton2 = Ct2/Gt
    toff2 = ton2/ton_toff
    t2 = ton2 + toff2
    f2 = 1/t2
    Vripple2 = Ipksw*t2/Co2/8
    Vout2 = Vo(R12.approx, R22.approx)

    svout.print(1)

    print(
        f'\n Rsc = {Rsc_approx.fmt_exact()} ~ {Rsc_approx}'
        f'\n  R1 < {fmt_eng(R1max, "Ω", 4)} -> {R12}'
        f'\n  R2 = {R22}'
        f'\n  Co > {fmt_eng(Comin, "F", 4)} -> {fmt_eng(Co2, "F")}'
        f'\n  Ct < {Ct_approx.fmt_exact()} ~ {Ct_approx}'
        f'\n   L > {fmt_eng(Lmin, "H", 4)} -> {fmt_eng(L2, "H")}'
        f'\n ton < {fmt_eng(ton, "s", 4)} -> {fmt_eng(ton2, "s", 4)}'
        f'\ntoff < {fmt_eng(toff, "s", 4)} -> {fmt_eng(toff2, "s", 4)}'
        f'\n   f > {fmt_eng(fmin, "Hz", 4)} -> {fmt_eng(f2, "Hz", 4)}'
        f'\nVrip < {fmt_eng(Vripple, "V", 4)} -> {fmt_eng(Vripple2, "V", 4)}'
        f'\nVout = {fmt_eng(Voutnom, "V", 4)} -> {fmt_eng(Vout2, "V", 4)}'
        f'\n'
    )


opamp3()
