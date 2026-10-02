from pathlib import Path

# 1. Definición completa de carpetas
DIRECTORIES = [
    ".github/workflows",
    "config/vehicles/ter26",
    "config/vehicles/ter27",
    "config/subsystems/aerodynamics",
    "config/subsystems/powertrain",
    "config/subsystems/suspension",
    "config/subsystems/tires",
    "config/subsystems/brakes",
    "config/events",
    "config/tracks/centerlines",
    "config/tracks/surface_friction",
    "config/can/dbc",
    "config/can/mappings",
    "data/raw/ttc",
    "data/raw/telemetry",
    "data/raw/cfd",
    "data/raw/kinematics",
    "data/raw/dyno",
    "data/processed/ttc_binned",
    "data/processed/clean_telemetry",
    "data/processed/aero_surfaces",
    "data/synthetic",
    "src/ter_twin/core/integrators",
    "src/ter_twin/core/math",
    "src/ter_twin/models/tires",
    "src/ter_twin/models/aerodynamics",
    "src/ter_twin/models/suspension",
    "src/ter_twin/models/powertrain",
    "src/ter_twin/models/brakes",
    "src/ter_twin/models/full_vehicle",
    "src/ter_twin/control/torque_vectoring",
    "src/ter_twin/control/traction_control",
    "src/ter_twin/control/driver",
    "src/ter_twin/control/drs",
    "src/ter_twin/surrogates/architectures",
    "src/ter_twin/surrogates/training",
    "src/ter_twin/validation/metrics",
    "src/ter_twin/validation/benchmarks_subsystems",
    "src/ter_twin/validation/benchmarks_vehicle",
    "src/ter_twin/validation/reporting",
    "src/ter_twin/realtime/ingest",
    "src/ter_twin/realtime/state",
    "src/ter_twin/realtime/pipeline",
    "src/ter_twin/realtime/api",
    "scripts/calibration",
    "scripts/studies",
    "scripts/realtime",
    "apps/desktop/widgets",
    "apps/web/src",
    "tests/unit",
    "tests/integration",
    "tests/regression",
    "docs/engineering_design",
    "docs/references",
    "reports/correlation",
    "reports/studies",
]

# 2. Archivos esqueleto a inicializar
FILES = [
    # CI/CD
    ".github/workflows/ci_tests.yml",
    ".github/workflows/correlation_check.yml",
    ".github/workflows/code_quality.yml",
    # Config
    "config/vehicles/base_vehicle.py",
    "config/vehicles/ter26/vehicle.yaml",
    "config/vehicles/ter26/hardpoints.yaml",
    "config/vehicles/ter26/aero_map.yaml",
    "config/vehicles/ter27/vehicle.yaml",
    "config/vehicles/ter27/hardpoints.yaml",
    "config/vehicles/ter27/aero_map.yaml",
    "config/vehicles/ter27/limits.yaml",
    "config/events/acceleration_75m.yaml",
    "config/events/skidpad_fsg.yaml",
    "config/events/autocross_default.yaml",
    "config/events/endurance_stint.yaml",
    # Core JAX
    "src/ter_twin/core/coordinates.py",
    "src/ter_twin/core/integrators/explicit.py",
    "src/ter_twin/core/integrators/implicit.py",
    "src/ter_twin/core/math/differentiators.py",
    "src/ter_twin/core/math/interpolators.py",
    "src/ter_twin/core/math/smooth_operators.py",
    # Models
    "src/ter_twin/models/tires/base.py",
    "src/ter_twin/models/tires/pacejka_52.py",
    "src/ter_twin/models/tires/brush.py",
    "src/ter_twin/models/tires/relaxation.py",
    "src/ter_twin/models/tires/thermal.py",
    "src/ter_twin/models/aerodynamics/quasi_static_map.py",
    "src/ter_twin/models/aerodynamics/drs_actuator.py",
    "src/ter_twin/models/aerodynamics/aero_moments.py",
    "src/ter_twin/models/suspension/lookups.py",
    "src/ter_twin/models/suspension/springs_dampers.py",
    "src/ter_twin/models/suspension/arb.py",
    "src/ter_twin/models/suspension/roll_centers.py",
    "src/ter_twin/models/powertrain/motor_inverter.py",
    "src/ter_twin/models/powertrain/accumulator.py",
    "src/ter_twin/models/powertrain/thermal_derating.py",
    "src/ter_twin/models/brakes/hydraulics.py",
    "src/ter_twin/models/brakes/thermal_disc.py",
    "src/ter_twin/models/full_vehicle/point_mass.py",
    "src/ter_twin/models/full_vehicle/single_track.py",
    "src/ter_twin/models/full_vehicle/planar_15dof.py",
    "src/ter_twin/models/full_vehicle/spatial_multibody.py",
    # Control
    "src/ter_twin/control/torque_vectoring/feedforward.py",
    "src/ter_twin/control/torque_vectoring/feedback_pid.py",
    "src/ter_twin/control/torque_vectoring/qp_allocator.py",
    "src/ter_twin/control/traction_control/slip_controller.py",
    "src/ter_twin/control/driver/pure_pursuit.py",
    "src/ter_twin/control/driver/optimal_preview.py",
    "src/ter_twin/control/drs/logic_gate.py",
    # Surrogates
    "src/ter_twin/surrogates/architectures/hnn.py",
    "src/ter_twin/surrogates/architectures/phnn.py",
    "src/ter_twin/surrogates/architectures/pinn.py",
    "src/ter_twin/surrogates/architectures/passive_hnet.py",
    "src/ter_twin/surrogates/training/loss_functions.py",
    "src/ter_twin/surrogates/training/trainer.py",
    # Validation
    "src/ter_twin/validation/metrics/statistical.py",
    "src/ter_twin/validation/metrics/frequency.py",
    "src/ter_twin/validation/benchmarks_subsystems/evaluate_tires.py",
    "src/ter_twin/validation/benchmarks_subsystems/evaluate_aero.py",
    "src/ter_twin/validation/benchmarks_subsystems/evaluate_kinematics.py",
    "src/ter_twin/validation/benchmarks_subsystems/evaluate_powertrain.py",
    "src/ter_twin/validation/benchmarks_vehicle/replay_simulation.py",
    "src/ter_twin/validation/benchmarks_vehicle/telemetry_overlay.py",
    "src/ter_twin/validation/benchmarks_vehicle/gg_diagram_match.py",
    "src/ter_twin/validation/reporting/generate_scores.py",
    "src/ter_twin/validation/reporting/export_dashboard.py",
    # Realtime
    "src/ter_twin/realtime/ingest/udp_receiver.py",
    "src/ter_twin/realtime/ingest/serial_receiver.py",
    "src/ter_twin/realtime/ingest/can_decoder.py",
    "src/ter_twin/realtime/state/ring_buffer.py",
    "src/ter_twin/realtime/state/observer.py",
    "src/ter_twin/realtime/pipeline/live_runner.py",
    "src/ter_twin/realtime/api/websocket_server.py",
    "src/ter_twin/realtime/api/payloads.py",
    # Scripts
    "scripts/calibration/fit_pacejka_from_ttc.py",
    "scripts/calibration/fit_aero_surfaces.py",
    "scripts/studies/wheel_rim_width_study.py",
    "scripts/studies/drs_laptime_sensitivity.py",
    "scripts/studies/arb_tuning_sweep.py",
    "scripts/run_correlation_suite.py",
    "scripts/run_telemetry_bridge.py",
    # Apps y Tests
    "apps/desktop/main.py",
    "apps/web/package.json",
    "tests/unit/test_pacejka_derivatives.py",
    "tests/unit/test_integrators.py",
    "tests/unit/test_can_decoding.py",
    "tests/integration/test_planar_step.py",
    "tests/integration/test_torque_vectoring_loop.py",
    "tests/regression/test_baseline_laptime.py",
    # Docs
    "docs/engineering_design/tire_modeling_report.md",
    "docs/engineering_design/suspension_kinematics.md",
    "docs/engineering_design/correlation_methodology.md",
    # Gitkeeps
    "data/raw/ttc/.gitkeep",
    "data/raw/telemetry/.gitkeep",
    "data/processed/.gitkeep",
    "reports/.gitkeep",
]

# 1. Crear carpetas
for d in DIRECTORIES:
    Path(d).mkdir(parents=True, exist_ok=True)

# 2. Crear __init__.py en src/ter_twin y tests/
for p in Path("src/ter_twin").rglob("*"):
    if p.is_dir():
        (p / "__init__.py").touch(exist_ok=True)
Path("src/ter_twin/__init__.py").touch(exist_ok=True)

for p in Path("tests").rglob("*"):
    if p.is_dir():
        (p / "__init__.py").touch(exist_ok=True)
Path("tests/__init__.py").touch(exist_ok=True)

# 3. Crear archivos
for f in FILES:
    p = Path(f)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)

# 4. Crear .gitignore
Path(".gitignore").write_text(
"""__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
build/
dist/
*.egg-info/
.installed.cfg
*.egg

.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

data/raw/ttc/*.mat
data/raw/telemetry/*.mf4
data/raw/telemetry/*.log
data/raw/telemetry/*.bin
data/processed/*.npz
data/processed/*.mat
!data/raw/ttc/.gitkeep
!data/raw/telemetry/.gitkeep
!data/processed/.gitkeep

reports/*
!reports/.gitkeep

.vscode/*
!.vscode/settings.json
.idea/
*.swp
*~
"""
)

# 5. Crear pyproject.toml
Path("pyproject.toml").write_text(
"""[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ter-twin"
version = "0.1.0"
description = "TeR-Twin: Suite de dinamica vehicular, correlacion y telemetria en tiempo real para Formula Student"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    { name = "Tecnun eRacing", email = "eracing@tecnun.es" }
]
dependencies = [
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "cantools>=39.0.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-xdist>=3.3.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]
realtime = [
    "websockets>=11.0",
    "pyserial>=3.5",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
strict = false
"""
)

# 6. Crear README.md
Path("README.md").write_text(
"""# TeR-Twin — Digital Twin & Vehicle Dynamics Simulation Suite

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

"""
)

print("[✓] Repositorio TeR-Twin estructurado con éxito.")
