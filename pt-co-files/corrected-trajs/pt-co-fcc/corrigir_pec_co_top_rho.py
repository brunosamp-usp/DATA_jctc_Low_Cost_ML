#!/usr/bin/env python3
"""Reconstrói rho da PEC top usando os frames salvos no scan UMA.
Nao executa FairChem; nao altera energias. Usa os 4 Pt com maior z,
como determinar_correcao_pt_co_sites.py e aplicar_correcao_pt_co_sites.py.
"""
from pathlib import Path
import argparse
import numpy as np
from ase.io import read


def rho_four_top(a, n=4):
    sy = a.get_chemical_symbols()
    c = [i for i, s in enumerate(sy) if s == 'C']
    pt = [i for i, s in enumerate(sy) if s == 'Pt']
    if len(c) != 1 or len(pt) < n:
        raise ValueError('Estrutura precisa ter exatamente um C e pelo menos n Pt')
    top = sorted(pt, key=lambda i: a.positions[i, 2], reverse=True)[:n]
    return float(np.mean([a.get_distance(i, c[0], mic=True) for i in top]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pec', default='adsorption_curve_co_top_uma.dat')
    ap.add_argument('--traj', default='pt-co-fcc-scan-final.traj')
    ap.add_argument('--out', default='adsorption_curve_co_top_uma_rho4.dat')
    ap.add_argument('--n-top', type=int, default=4)
    args = ap.parse_args()
    data = np.atleast_2d(np.loadtxt(args.pec))
    if data.shape[1] != 5:
        raise ValueError('Esperadas 5 colunas no formato do scan_pt_co_top.py')
    frames = read(args.traj, index=':')
    if len(frames) != len(data):
        raise ValueError(f'{len(frames)} frames, mas {len(data)} linhas na PEC')
    out = np.column_stack([data[:, :4], [rho_four_top(a, args.n_top) for a in frames], data[:, 4]])
    # Validar pareamento por energia total e distancia Pt-C original
    for i, (a, row) in enumerate(zip(frames, data)):
        e = a.get_potential_energy()
        if abs(e - row[1]) > 1e-4:
            raise ValueError(f'Linha {i}: energia nao coincide com frame ({row[1]} vs {e})')
        pt_ref = 15
        c = next(j for j, s in enumerate(a.get_chemical_symbols()) if s == 'C')
        if abs(a.get_distance(pt_ref, c, mic=True) - row[3]) > 1e-3:
            raise ValueError(f'Linha {i}: Pt-C indice 15 nao coincide com frame')
    np.savetxt(args.out, out, fmt='%.10f', header=(
        'PtC_target_A E_total_eV E_rel_CplusO_kcalmol '
        'PtC_actual_A rho_CPt_4top_A CO_bond_A'))
    print(f'Salvo: {args.out} ({len(frames)} pontos)')
    print('ATENCAO: a coluna 3 continua referida a Pt + C + O atomicos, NAO Pt + CO(g).')
    print('Para calibrar com o script atual, gere uma PEC de 4 colunas: x, E_total, E_ads_COgas, rho4.')

if __name__ == '__main__':
    main()
