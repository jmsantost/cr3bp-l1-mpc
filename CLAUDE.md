# CLAUDE.md: cr3bp-l1-mpc

Contexto del proyecto para Claude Code. Léelo completo antes de escribir código.

## 1. Qué es y para qué

Proyecto de autoestudio en astrodinámica de José Miguel Santos Terán (ingeniero mecatrónico, fuerte en MPC/MPPI y estimación). Va enlazado en su CV y en su research statement para un Ph.D. en spacecraft GNC (X-Bai Research Group, Rutgers). Por eso:

- El código debe ser correcto, limpio y verificable, no solo "funcionar".
- José debe entender cada línea. Explica las decisiones matemáticas al implementar, no solo el código.
- Nada de resultados inventados: todo número del README sale de una corrida reproducible.
- El repositorio se llama `cr3bp-l1-mpc` y se publica en `github.com/jmsantost/cr3bp-l1-mpc`.

Texto que ya aparece en su CV (el proyecto debe cumplirlo literalmente):

> Implemented a planar Earth-Moon CR3BP toolkit in Python: Lagrange points, an L1 Lyapunov orbit family via differential correction and continuation, invariant manifolds from the monodromy matrix, and LTV MPC station-keeping under navigation noise and bounded thrust.

Existe una implementación de referencia en un solo archivo, `reference/cr3bp_l1_mpc.py`, ya probada. El trabajo es convertirla en un paquete modular con tests, entender cada parte y luego extenderla.

## 2. Estructura objetivo del repositorio

```
cr3bp-l1-mpc/
├── CLAUDE.md
├── README.md
├── requirements.txt          # numpy, scipy, matplotlib, cvxpy, pytest
├── reference/cr3bp_l1_mpc.py # implementación original, no se modifica
├── src/cr3bp/
│   ├── __init__.py
│   ├── constants.py          # MU, unidades
│   ├── dynamics.py           # ecuaciones de movimiento, Jacobiano, STM, constante de Jacobi
│   ├── lagrange.py           # puntos colineales L1, L2, L3
│   ├── periodic_orbits.py    # corrección diferencial y continuación (Lyapunov)
│   ├── manifolds.py          # monodromía, autovalores, variedades estable e inestable
│   ├── mpc.py                # modelo LTV discreto y controlador MPC
│   └── plotting.py
├── scripts/run_all.py        # genera todas las figuras y un results.json
├── tests/                    # pytest, ver sección 6
└── figures/
```

## 3. Modelo

Problema circular restringido de tres cuerpos, plano, sistema Tierra-Luna, marco rotatorio centrado en el baricentro, unidades adimensionales.

| Constante | Valor |
| --- | --- |
| μ (masa Luna / masa total) | 0.012150585 |
| Unidad de longitud | 384,400 km |
| Unidad de tiempo | 375,190 s (≈ 4.34 días) |
| Unidad de velocidad | L/T ≈ 1024.5 m/s |
| Unidad de aceleración | V/T ≈ 2.73e-3 m/s² |

Tierra en (−μ, 0), Luna en (1−μ, 0). Estado `s = [x, y, vx, vy]`.

```
r1 = sqrt((x+μ)² + y²)        r2 = sqrt((x−1+μ)² + y²)
U  = (x²+y²)/2 + (1−μ)/r1 + μ/r2

ẍ − 2ẏ = ∂U/∂x + ux
ÿ + 2ẋ = ∂U/∂y + uy
```

`u = [ux, uy]` es aceleración de control (solo en MPC).

Constante de Jacobi: `C = 2U − (vx² + vy²)`. Se conserva sin control y sirve para validar el integrador.

Linealización: `δṡ = A(t) δs`, con

```
A = [[0,   0,   1, 0],
     [0,   0,   0, 1],
     [Uxx, Uxy, 0, 2],
     [Uxy, Uyy, −2, 0]]
```

La matriz de transición de estados (STM) cumple `Φ̇ = A(t) Φ`, `Φ(0) = I`. Se integra junto al estado (20 ecuaciones).

Integrador: `scipy.integrate.solve_ivp` (RK45 o DOP853), `rtol=1e-11`, `atol=1e-12` para órbitas periódicas.

## 4. Algoritmos

### 4.1 Puntos de Lagrange
Raíces de ∂U/∂x = 0 sobre y = 0 con `brentq`, en los intervalos (0.5, 1−μ), (1−μ, 1.5) y (−1.5, −μ).

### 4.2 Órbitas de Lyapunov en L1 (corrección diferencial)
1. Autovalores de A en L1: un par real ±λ (silla) y un par imaginario ±iω (centro). El autovector del modo centro, normalizado con componente x real, da la condición inicial lineal: `x0 = xL1 + ε Re(v_x)`, `vy0 = ε Re(v_vy)`, con ε = −1e-3.
2. Por simetría respecto al eje x, basta con integrar media órbita: desde `[x0, 0, 0, vy0]` hasta el primer cruce de y = 0 (evento con `direction=-1`, terminal). En el cruce se exige `vx = 0`.
3. Con x0 fijo, se corrige vy0 (Newton):

```
δvy0 = −vx_f / ( Φ[2,3] − (ẍ_f / vy_f) · Φ[1,3] )
```

   El término con Φ[1,3] corrige el hecho de que el tiempo de cruce también cambia. Iterar hasta |vx_f| < 1e-11. Periodo T = 2 · t_cruce.
4. Continuación natural: x0 se aleja de L1 en pasos de 0.003. La estimación de vy0 para la siguiente órbita se extrapola con la secante de las dos anteriores (con una sola, escalado lineal con la amplitud). Pasos más grandes hicieron fallar la convergencia en las pruebas.

### 4.3 Monodromía y variedades
1. Integrar una órbita completa con STM: M = Φ(T).
2. Autovalores de M: 1, 1, λu, λs con λu · λs ≈ 1. λu ≫ 1 mide la inestabilidad.
3. Para N puntos de la órbita (t = τ), transportar el autovector: `v(τ) = Φ(τ) v0`, normalizar por su parte de posición, perturbar `d = 50 km` en ambos sentidos.
4. Variedad inestable: integrar hacia adelante. Estable: hacia atrás. Tiempo de flujo ≈ 3.5.

### 4.4 MPC de mantenimiento de órbita
- Referencia: la órbita de Lyapunov más grande de la familia, periódica, interpolada con `dense_output`.
- Paso de control: dt = T/150. Horizonte N = 25.
- Modelo de predicción LTV discreto del error δs = s − s_ref:
  - `Ad_k` = STM a lo largo de la referencia durante un paso.
  - `Bd_k ≈ ½ (Ad_k + I) Bc dt`, con `Bc = [0; I]` (aproximación trapezoidal de retención de orden cero; documentarla como aproximación).
  - Se precalculan para un periodo y se indexan con `k mod 150`.
- Problema de optimización (cvxpy, solver CLARABEL):

```
min  Σ_{j=1..N} δs_jᵀ Q δs_j + Σ_{j=0..N−1} u_jᵀ R u_j
s.a. δs_{j+1} = Ad δs_j + Bd u_j,   ||u_j||∞ ≤ umax,   δs_0 = estimación
Q = diag(1, 1, 0.1, 0.1) · 1e4,   R = I,   umax = 2e-4 m/s² (convertido a adimensional)
```

- Usar `cp.Parameter` para δs_0, Ad y Bd, y construir el problema una sola vez.
- Planta: CR3BP no lineal completo con u constante durante cada paso.
- Escenario: error de inyección de magnitud fija (20 km en posición, 5 cm/s en velocidad, dirección aleatoria, semilla 0). Ruido de navegación gaussiano sobre el estado que recibe el MPC (σ = 1 km y 1 cm/s). Se aplica solo el primer control (horizonte deslizante).
- Comparación contra el caso sin control. La simulación se corta si el error supera 0.3 (escape).
- Métricas: error de posición en km vs tiempo y Δv total = Σ|u| dt en m/s.

## 5. Resultados de referencia (para validar)

La implementación de referencia, con los parámetros por defecto, da:

| Cantidad | Valor |
| --- | --- |
| x de L1 / L2 / L3 | 0.836915 / 1.155682 / −1.005063 |
| Distancia L1 a la Luna | ≈ 58,019 km |
| Periodo de la órbita de Lyapunov más pequeña | 2.6918 (≈ 11.69 días) |
| Periodo de la más grande (x0 = 0.81492) | 2.8657 (≈ 12.44 días) |
| λu de la órbita más grande | ≈ 1797 |
| Sin control | escapa en ≈ 24.5 días |
| MPC | error de 20 km a ≈ 1 km (piso del ruido de navegación); Δv ≈ 4.2 m/s en ≈ 37 días |

Si el paquete modular no reproduce estos valores (salvo diferencias pequeñas por tolerancias o solver), hay un error.

## 6. Tests obligatorios (pytest)

- Puntos de Lagrange: gradiente del potencial ≈ 0 (< 1e-12).
- Jacobiano: coincide con diferencias finitas del campo vectorial (< 1e-6).
- STM: coincide con diferencias finitas de la propagación (< 1e-6 relativo).
- det(Φ(t)) = 1 (flujo que conserva volumen) con error < 1e-8.
- Constante de Jacobi conservada en una órbita sin control (variación < 1e-10).
- Periodicidad: ||s(T) − s(0)|| < 1e-8 para cada órbita de la familia.
- Monodromía: λu · λs ≈ 1 y dos autovalores ≈ 1.
- MPC: con ruido cero y sin error inicial, u ≈ 0 y el error se mantiene ≈ 0.

## 7. Plan de trabajo por fases

Cada fase termina con tests pasando y un commit. En cada fase, explicar a José la matemática antes del código.

1. **Base:** estructura del repo, `constants.py`, `dynamics.py`, `lagrange.py` y sus tests.
2. **Órbitas periódicas:** corrección diferencial, continuación, figura de la familia.
3. **Variedades:** monodromía, figura de variedades estable e inestable alrededor de L1 y la Luna.
4. **MPC:** modelo LTV, controlador, comparación sin control, figura de error y Δv.
5. **Publicación:** `scripts/run_all.py` genera figuras y `results.json`; README con resultados reales, instrucciones y limitaciones.

## 8. Extensiones (después de publicar la versión base)

En orden de valor para la aplicación:

1. **EKF en el lazo:** reemplazar el ruido sobre el estado por mediciones ruidosas (por ejemplo rango y ángulo desde la Tierra) y un filtro de Kalman extendido. Conecta con el trabajo de estimación del grupo.
2. **MPC con restricciones que dependen de la covarianza:** tubo o corredor alrededor de la referencia que se ajusta con la covarianza del EKF. Es la idea del research statement, en versión simplificada.
3. **Comparación LQR periódico vs MPC** en Δv y en error para varios niveles de ruido (barrido de Monte Carlo).
4. **Órbitas halo** (CR3BP espacial, 6 estados).
5. **Animación GIF** de la órbita controlada y sin control para el README.

## 9. Lo que José debe poder explicar sin el código (preparación de entrevista)

- Por qué el marco rotatorio produce los términos de Coriolis 2ẏ y −2ẋ.
- Qué significa la constante de Jacobi y qué son las curvas de velocidad cero.
- Por qué L1 es inestable (autovalores silla × centro).
- La fórmula de corrección diferencial, incluido el término de corrección por el tiempo de cruce.
- Por qué la simetría permite integrar solo media órbita.
- Qué es la matriz de monodromía, por qué λu · λs = 1 y qué implica λu ≈ 1800.
- Por qué las variedades sirven para transferencias de bajo costo.
- Por qué usar un modelo LTV y no uno LTI para el MPC, y qué aproxima Bd.
- Por qué el error se estabiliza alrededor de 1 km (piso del ruido) y cómo cambiaría el Δv con mejor navegación.

## 10. Convenciones

- Python 3.10+, type hints, docstrings breves con las ecuaciones relevantes.
- Unidades adimensionales en todo el núcleo; conversión a km, días y m/s solo en salida y figuras.
- Nada de números mágicos sin nombre: constantes en `constants.py` o como argumentos con valor por defecto.
- En documentos y README: sin rayas largas (em dashes).
