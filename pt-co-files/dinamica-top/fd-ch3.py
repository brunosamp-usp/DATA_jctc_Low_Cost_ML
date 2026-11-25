#!/usr/bin/env python3
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution, Stationary, ZeroRotation
)

from fairchem.core import pretrained_mlip, FAIRChemCalculator

# ---------- entradas ----------
START_XYZ   = "pt-co-dist.xyz"   # mude se quiser outra geometria inicial
BOTTOM_FIX  = 8                # fixa 2 camadas inferiores (índices 0..31)
T_K         = 300.0               # temperatura alvo (K)
DT_fs       = 0.5                 # passo (fs)
FRICTION    = 0.002               # atrito (~1/ps). Ajuste conforme necessário
NSTEPS      = 100               # 0.5 fs × 20000 = 10 ps
TRAJ_FILE   = "pt-co-dist.traj"
FINAL_JSON  = "pt-co-dist.json"

# ---------- carrega sistema ----------
atoms = read(START_XYZ)
atoms.set_constraint(FixAtoms(indices=range(BOTTOM_FIX)))  # cheque se 0..31 batem com seu slab

# ---------- calculadora ML (UMA s-1p1) ----------
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
calc = FAIRChemCalculator(predictor, task_name="oc20")
atoms.calc = calc

# ---------- velocidades iniciais ----------
#MaxwellBoltzmannDistribution(atoms, temperature_K=T_K)
#Stationary(atoms)      # remove movimento do centro de massa
#ZeroRotation(atoms)    # remove rotação global

# ---------- MD Langevin ----------
dyn = Langevin(atoms,
               timestep=DT_fs * units.fs,
               temperature_K=T_K,
               friction=FRICTION)

# ---------- utilitário de impressão ----------
def print_energy(a=atoms):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    temp = ekin / (1.5 * units.kB * len(a))
    print(f"Epot={epot: .6f} eV  Ekin={ekin: .6f} eV  "
          f"T={temp:7.2f} K  Etot={epot+ekin: .6f} eV")

# ---------- anexos: salvar e logar ----------
traj = Trajectory(TRAJ_FILE, "w", atoms)
dyn.attach(traj.write, interval=1)      # grava a cada 50 passos
dyn.attach(print_energy, interval=1)    # imprime a cada 50 passos

# ---------- roda MD ----------
dyn.run(NSTEPS)

# ---------- salva configuração final ----------
write(FINAL_JSON, atoms, format="json")
print(f"[ok] Trajetória: {TRAJ_FILE} | Final: {FINAL_JSON}")

