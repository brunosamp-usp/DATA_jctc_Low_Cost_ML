#!/usr/bin/env python3
"""PEC rigida CO/Pt(111) top com UMA-s-1p1.

Entrada: pt-co-top.xyz
Coordenada: distancia minima-imagem Pt (indice ASE 15) -> C.
O CO e transladado rigidamente ao longo do vetor Pt->C inicial.
Nao ha relaxacao: slab e CO mantem suas geometrias internas.
Os .traj contem energia e forcas, mas NAO stress.

AVISO: a referencia E_CO_REF_EV=-14.486 eV corresponde a C+O atomicos,
nao ao CO gasoso. Nao comparar a energia relativa diretamente a -29.6
kcal/mol experimental sem calcular a referencia CO(g) compativel.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator
from fairchem.core import pretrained_mlip, FAIRChemCalculator

INPUT = Path('pt-co-top.xyz')
OUTDIR = Path('estruturas-top')
DAT = Path('adsorption_curve_co_top_uma.dat')
FIG = Path('pec_co_top_uma.png')
COMBINED = Path('pt-co-top-scan-final.traj')
PT_REF = 15                 # indice ASE, comeca em 0
DISTANCES = np.sort(
    np.unique(
        np.append(
            np.round(np.arange(0.5, 6.0 + 0.00001, 0.1), 2),
            1.85
        )
    )
)
DEVICE = 'cpu'
TASK = 'oc20'
EV2KCAL = 23.060548
E_SLAB_EV = -83.18795112621684
E_CO_REF_EV = -14.486     # C + O atomicos, NAO CO(g)


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f'Arquivo ausente: {INPUT}')
    base = read(INPUT)
    symbols = base.get_chemical_symbols()
    pt = [i for i, sym in enumerate(symbols) if sym == 'Pt']
    carbon = [i for i, sym in enumerate(symbols) if sym == 'C']
    oxygen = [i for i, sym in enumerate(symbols) if sym == 'O']
    if (len(pt), len(carbon), len(oxygen)) != (16, 1, 1):
        raise ValueError('Esperados exatamente 16 Pt, 1 C e 1 O.')
    if PT_REF not in pt:
        raise ValueError(f'O indice ASE {PT_REF} nao e Pt.')
    ci, oi = carbon[0], oxygen[0]

    # Verifica se o Pt escolhido esta aproximadamente abaixo do C.
    lateral = np.linalg.norm(base.positions[ci, :2] - base.positions[PT_REF, :2])
    if lateral > 0.5:
        raise ValueError(
            f'Pt {PT_REF} nao parece estar abaixo do C: deslocamento xy={lateral:.3f} A. '
            'Confirme o indice e a estrutura.'
        )

    # Direcao inicial Pt->C com convencao de imagem minima.
    axis = np.asarray(base.get_distance(PT_REF, ci, mic=True, vector=True), dtype=float)
    initial_r = float(np.linalg.norm(axis))
    if initial_r < 1e-8:
        raise ValueError('Distancia Pt-C inicial aproximadamente zero.')
    axis /= initial_r
    co_initial = base.get_distance(ci, oi, mic=True)
    print(f'Pt indice ASE = {PT_REF}; C = {ci}; O = {oi}')
    print(f'Distancia inicial Pt-C = {initial_r:.6f} A')
    print(f'Distancia inicial C-O  = {co_initial:.6f} A')
    print(f'Deslocamento lateral C-Pt = {lateral:.6f} A')

    OUTDIR.mkdir(parents=True, exist_ok=True)
    predictor = pretrained_mlip.get_predict_unit('uma-s-1p1', device=DEVICE)
    calc = FAIRChemCalculator(predictor, task_name=TASK)
    rows = []
    with Trajectory(str(COMBINED), 'w') as all_frames:
        for j, distance in enumerate(DISTANCES, 1):
            atoms = base.copy()
            shift = (float(distance) - initial_r) * axis
            atoms.positions[[ci, oi]] += shift
            # Sem otimizacao: uma avaliacao UMA por geometria.
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=np.float32)
            rho = float(atoms.get_distance(PT_REF, ci, mic=True))
            co_bond = float(atoms.get_distance(ci, oi, mic=True))
            e_rel = (energy - E_SLAB_EV - E_CO_REF_EV) * EV2KCAL

            frame = atoms.copy()
            # Calculadora armazenada sem stress e sem free_energy.
            frame.calc = SinglePointCalculator(frame, energy=energy, forces=forces)
            stem = f'pt-co-top-{distance:.2f}'
            with Trajectory(str(OUTDIR / f'{stem}-final.traj'), 'w') as tr:
                tr.write(frame)
            all_frames.write(frame)
            rows.append([distance, energy, e_rel, rho, co_bond])
            print(f'[{j:02d}/{len(DISTANCES)}] r={rho:.3f} A | '
                  f'E={energy:.8f} eV | E_rel={e_rel:+.4f} kcal/mol | '
                  f'C-O={co_bond:.4f} A', flush=True)

    data = np.asarray(rows)
    np.savetxt(DAT, data, fmt=['%.2f', '%.10f', '%.8f', '%.8f', '%.8f'],
               header='PtC_target_A E_total_eV E_rel_CplusO_kcalmol PtC_actual_A CO_bond_A')
    imin = int(np.argmin(data[:, 2]))
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(data[:, 0], data[:, 2], 'o-', lw=1.8, ms=3.8)
    plt.xlabel('Distancia Pt-C (A)', fontsize=15)
    plt.ylabel('Energia relativa a Pt + C + O (kcal/mol)', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG, dpi=300)
    plt.savefig(FIG.with_suffix('.pdf'))
    plt.close()
    print('\nCONCLUIDO')
    print(f'Minimo amostrado: r={data[imin,0]:.2f} A, E={data[imin,2]:+.4f} kcal/mol')
    print(f'Dados: {DAT} | Figura: {FIG} | Trajetorias: {OUTDIR}/ e {COMBINED}')
    print('ATENCAO: referencia energetica C+O atomicos, NAO CO(g).')


if __name__ == '__main__':
    main()
