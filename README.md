# TeR-Twin — Digital Twin & Vehicle Dynamics Simulation Suite

Suite de simulación física diferenciable, correlación experimental de telemetría y gemelo digital en tiempo real para los monoplazas eléctricos **TeR26** y **TeR27** de Tecnun eRacing.

## Arquitectura del Proyecto

- `src/ter_twin/core/`: Motores de integración temporal (Heun RK2, GLRK-4) y operadores suaves en JAX.
- `src/ter_twin/models/`: Dinámica planar (15 DOF), modelos Pacejka MF5.2 acoplados, mapas aero y cinemática de suspensiones.
- `src/ter_twin/control/`: VCU virtual (Torque Vectoring por QP, Traction Control).
- `src/ter_twin/validation/`: Banco de correlación cuantitativa (RMSE, NMAE, $R^2$) frente a datos de Calspan (TTC) y pista.
- `src/ter_twin/realtime/`: Buffer circular de telemetría y servidor WebSocket para monitorización en boxes.

## Instalación

```bash
pip install -e ".[dev,realtime]"

