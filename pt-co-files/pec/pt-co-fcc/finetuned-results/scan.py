#!/usr/bin/env python3
"""PEC vertical CO/fcc em Pt(111) 2x2x4, UMA-s-1p1.
Entrada: estruturas/pt-co-fcc-*.xyz
Saidas: .dat, .png, .pdf, e .traj de frame final com energia e forcas (sem stress)
"""
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from ase.io import read, write
from ase.constraints import FixAtoms, FixedPlane
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

INPUT_DIR = Path('estruturas')
OUTFILE = Path('adsorption_curve_co_uma.dat')
OUT_FIG = Path('pec_co_uma.png')
EV2KCAL = 23.060548
DEVICE = 'cpu'
TASK = 'oc20'
RELAX_PT_XY = True
FMAX = 0.01
STEPS = 200

# Mesma convencao de referencia do calculo anterior do usuario:
E_SLAB_EV = -83.18795112621684
E_CO_REF_EV = -14.486  # C + O atomicos; NAO e CO(g)
EXTRA_CORRECTION_EV = 0.0
E_EXP_KCAL = None  # opcional, somente se comparavel a convencao acima


def parse_dist(path):
    m = re.fullmatch(r'pt-co-fcc-(\d+(?:\.\d+)?)\.xyz', path.name)
    return float(m.group(1)) if m else None


def get_indices(atoms):
    pt = [i for i, a in enumerate(atoms) if a.symbol == 'Pt']
    c = [i for i, a in enumerate(atoms) if a.symbol == 'C']
    o = [i for i, a in enumerate(atoms) if a.symbol == 'O']
    if (len(pt), len(c), len(o)) != (16, 1, 1):
        raise ValueError(f'Esperados 16 Pt, 1 C, 1 O; encontrados {len(pt)} Pt, {len(c)} C, {len(o)} O')
    return pt, c[0], o[0]


def main():
    files = sorted(
        (p for p in INPUT_DIR.glob('pt-co-fcc-*.xyz') if parse_dist(p) is not None),
        key=parse_dist,
    )
    if not files:
        raise FileNotFoundError(f'Nenhum arquivo pt-co-fcc-*.xyz encontrado em {INPUT_DIR}')

    print('Carregando UMA-s-1p1...', flush=True)
    predictor =  load_predict_unit("/home/bruno/doutorado/fairchem/4-pt/resposta/finetuning-com-replay/checkpoint-nivel-dft/checkpoint-umlip/checkpoint-co-fcc/inference_ckpt.pt")
    calc = FAIRChemCalculator(predictor, task_name=TASK)
    records = []
    combined_traj = Trajectory("pt-co-fcc-scan-final.traj", "w")

    for j, path in enumerate(files, 1):
        dist = parse_dist(path)
        print(f'[{j}/{len(files)}] {path.name}', flush=True)
        try:
            atoms = read(path)
            pt, c_idx, o_idx = get_indices(atoms)
            sorted_pt = sorted(pt, key=lambda i: atoms.positions[i, 2])
            bottom = sorted_pt[:8]
            top = sorted_pt[-4:]

            # Impede movimento do CO e dos Pt inferiores; Pt superiores so em x/y.
            if RELAX_PT_XY:
                atoms.set_constraint([
                    FixAtoms(indices=bottom + [c_idx, o_idx]),
                    FixedPlane(indices=[i for i in pt if i not in bottom], direction=(0, 0, 1)),
                ])
            else:
                atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)))))

            atoms.calc = calc
            if RELAX_PT_XY:
                opt = BFGS(atoms, logfile=str(INPUT_DIR / f'{path.stem}.log'))
                opt.run(fmax=FMAX, steps=STEPS)
                if not opt.converged():
                    print('  AVISO: BFGS nao convergiu no limite de passos.', flush=True)

            # Energia e forcas fisicas do modelo, sem zerar forcas por restricoes.
            # A trajetoria de cada ponto tera APENAS o frame final.
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
            final_frame = atoms.copy()
            final_frame.calc = SinglePointCalculator(
                final_frame,
                energy=energy,
                forces=forces,
            )
            traj_path = INPUT_DIR / f'{path.stem}-final.traj'
            with Trajectory(str(traj_path), 'w') as traj:
                traj.write(final_frame)
            combined_traj.write(final_frame)
            rho = float(np.mean(atoms.get_distances(c_idx, top, mic=True)))
            z_top = float(np.mean(atoms.positions[top, 2]))
            height = float(atoms.positions[c_idx, 2] - z_top)
            co_bond = float(atoms.get_distance(c_idx, o_idx, mic=True))
            e_ads = (energy - E_SLAB_EV - E_CO_REF_EV + EXTRA_CORRECTION_EV) * EV2KCAL

            write(INPUT_DIR / f'{path.stem}-final.xyz', atoms, format='extxyz')
            records.append((dist, energy, e_ads, rho, height))
            print(f'  E={energy:.9f} eV | E_ref={e_ads:.4f} kcal/mol | rho={rho:.5f} A | altura={height:.5f} A | C-O={co_bond:.5f} A', flush=True)
        except Exception as exc:
            print(f'  ERRO: {exc}', flush=True)

    combined_traj.close()

    if not records:
        raise RuntimeError('Nenhum ponto do scan foi concluido.')

    data = np.array(sorted(records), dtype=float)
    x, energies, e_ads, rhos, heights = data.T
    with OUTFILE.open('w', encoding='utf-8') as fh:
        fh.write('# altura_C(Angstrom) E_tot(eV) E_ref_CmaisO(kcal/mol) rho_CPt(Angstrom) altura_real(Angstrom)\n')
        for row in data:
            fh.write(f'{row[0]:8.2f} {row[1]:+.10f} {row[2]:+.6f} {row[3]:.7f} {row[4]:.7f}\n')

    plt.figure(figsize=(6.2, 4.6))
    if len(x) >= 3:
        xx = np.linspace(x.min(), x.max(), 600)
        plt.plot(xx, CubicSpline(x, e_ads)(xx), lw=2.2, color='#d95f02', label='UMA-s-1p1')
    else:
        plt.plot(x, e_ads, lw=2.2, color='#d95f02', label='UMA-s-1p1')
    plt.plot(x, e_ads, 'o', ms=4, color='#d95f02')
    if E_EXP_KCAL is not None:
        plt.axhline(E_EXP_KCAL, ls='--', lw=1.5, color='#228b22', label='Experimental')
    plt.xlabel(r'Distance ($\mathrm{\AA}$)', fontsize=22)
    plt.ylabel(r'$E$ (kcal mol$^{-1}$)', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(x.min(), x.max())
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=600, bbox_inches='tight')
    plt.savefig(OUT_FIG.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    imin = int(np.argmin(e_ads))
    print(f'\nFinalizado: {len(x)}/{len(files)} pontos')
    print(f'Minimo amostrado: d={x[imin]:.2f} A, E={e_ads[imin]:.3f} kcal/mol')
    print(f'Dados: {OUTFILE}\nFigura: {OUT_FIG} e {OUT_FIG.with_suffix(".pdf")}')
    print('Trajetorias: estruturas/pt-co-fcc-*-final.traj e pt-co-fcc-scan-final.traj')
    print('ATENCAO: E_CO_REF_EV=-14.486 eV corresponde a C+O atomicos, nao CO gasoso.')


if __name__ == '__main__':
    main()

