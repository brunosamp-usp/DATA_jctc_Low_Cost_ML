#!/usr/bin/env python3

import glob
import re
import os

import numpy as np

from ase.constraints import FixAtoms, FixedPlane
from ase.optimize import BFGS
from ase.io import read, write

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


# ==========================================================
# CONSTANTES
# ==========================================================

EV2KCAL = 23.06  # eV -> kcal/mol


# ==========================================================
# FUNÇÃO PARA EXTRAIR A DISTÂNCIA DO NOME
#
# Exemplos:
# pt-o-0.50.xyz -> 0.50
# pt-o-2.00.xyz -> 2.00
# pt-o-6.00.xyz -> 6.00
# ==========================================================

def parse_dist(path: str):

    m = re.search(
        r"pt-o-([0-9]+(?:\.[0-9]+)?)\.xyz$",
        os.path.basename(path)
    )

    return float(m.group(1)) if m else None


# ==========================================================
# p_mean
#
# Para slab Pt(111) 3x3x4:
# 36 Pt + 1 O
#
# primeiros 18 Pt = duas camadas inferiores fixas
# camada superior = índices 27..35
#
# O = índice 36, se a estrutura tiver a ordem esperada.
# ==========================================================

TOP_PT_IDX = list(range(27, 36))


def compute_p_mean(atoms):

    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()

    o_idx = [
        i for i, symbol in enumerate(syms)
        if symbol == "O"
    ]

    if not o_idx:
        return float("nan")

    if len(o_idx) > 1:
        print(
            f"[AVISO] Encontrados {len(o_idx)} átomos de O. "
            "Usando o primeiro."
        )

    rO = pos[o_idx[0]]

    # distância O--Pt para os 9 Pt da camada superior
    d = (
        (pos[TOP_PT_IDX] - rO) ** 2
    ).sum(axis=1) ** 0.5

    return float(d.mean())


# ==========================================================
# MODELO FAIR-CHEM
# ==========================================================

CHECKPOINT = (
    "/home/bruno/doutorado/fairchem/4-pt/resposta/"
    "finetuning-com-replay/checkpoint-nivel-dft/"
    "checkpoint-umlip/treino-proximo/teste-3/inference_ckpt.pt"
)

print("\n========================================")
print("Carregando modelo")
print("========================================")
print(CHECKPOINT)

predictor = load_predict_unit(
    CHECKPOINT,
    device="cpu"
)

calc = FAIRChemCalculator(
    predictor,
    task_name="oc20"
)


# ==========================================================
# SLAB PURO
# ==========================================================

print("\n========================================")
print("Otimizando slab puro")
print("========================================")

slab = read("pt-puro-3.xyz")

# fixa primeiros 18 Pt
slab.set_constraint(
    FixAtoms(indices=range(18))
)

slab.calc = calc

opt_slab = BFGS(
    slab,
    logfile="pt-puro-3-opt.log"
)

opt_slab.run(
    fmax=0.01,
    steps=100
)

slab_e = float(
    slab.get_potential_energy()
)

print(
    f"Energia calculada do slab puro: "
    f"{slab_e:.9f} eV"
)

print(
    "Referência usada no E_ads: "
    "-187.1716267731465 eV"
)


# ==========================================================
# SAÍDA
# ==========================================================

OUTFILE = "adsorption_curve_o_sem_corr.dat"

with open(
    OUTFILE,
    "w",
    encoding="utf-8",
    newline="\n"
) as fout:

    fout.write(
        "# dist(Å)   "
        "E_tot(eV)   "
        "E_ads(kcal/mol)   "
        "p_mean(Å)\n"
    )


    # ======================================================
    # ARQUIVOS pt-o-*.xyz
    # ======================================================

    files = glob.glob(
        "pt-o-*.xyz"
    )

    files = [
        f for f in files
        if parse_dist(f) is not None
    ]

    files = sorted(
        files,
        key=parse_dist
    )

    print("\n========================================")
    print(
        f"Encontrados {len(files)} arquivos "
        "para o scan."
    )
    print("========================================")

    if len(files) == 0:
        raise RuntimeError(
            "Nenhum arquivo pt-o-*.xyz compatível "
            "foi encontrado."
        )


    # ======================================================
    # LOOP DO SCAN
    # ======================================================

    for fname in files:

        dist = parse_dist(fname)

        print("\n========================================")
        print(
            f">>> Rodando {fname} "
            f"(dist={dist:.2f} Å)"
        )
        print("========================================")

        try:

            # --------------------------------------------------
            # lê estrutura
            # --------------------------------------------------

            adslab = read(fname)


            # --------------------------------------------------
            # identifica O e Pt
            # --------------------------------------------------

            o_idx = [
                i
                for i, atom in enumerate(adslab)
                if atom.symbol == "O"
            ]

            pt_idx = [
                i
                for i, atom in enumerate(adslab)
                if atom.symbol == "Pt"
            ]

            if len(o_idx) == 0:
                raise RuntimeError(
                    f"Nenhum átomo O encontrado em {fname}"
                )

            print(
                f"O index: {o_idx}"
            )

            print(
                f"Número de Pt: {len(pt_idx)}"
            )


            # --------------------------------------------------
            # restrições
            #
            # 1) primeiros 18 Pt completamente fixos
            # 2) O completamente fixo durante cada ponto
            # 3) Pt restantes sem movimento em z
            # --------------------------------------------------

            fix_bottom = FixAtoms(
                indices=range(18)
            )

            fix_O = FixAtoms(
                indices=o_idx
            )

            fix_Ptz = FixedPlane(
                indices=pt_idx,
                direction=[0, 0, 1]
            )

            adslab.set_constraint(
                [
                    fix_bottom,
                    fix_O,
                    fix_Ptz
                ]
            )


            # --------------------------------------------------
            # calculator
            # --------------------------------------------------

            adslab.calc = calc


            # --------------------------------------------------
            # otimização
            # --------------------------------------------------

            logfile = fname.replace(
                ".xyz",
                ".log"
            )

            opt = BFGS(
                adslab,
                logfile=logfile
            )

            opt.run(
                fmax=0.01,
                steps=200
            )


            # --------------------------------------------------
            # salva apenas estrutura final
            # --------------------------------------------------

            final_traj = fname.replace(
                ".xyz",
                ".traj"
            )

            write(
                final_traj,
                adslab,
                format="traj"
            )


            # --------------------------------------------------
            # energia total
            # --------------------------------------------------

            E_tot = float(
                adslab.get_potential_energy()
            )


            # --------------------------------------------------
            # energia de adsorção
            #
            # fórmula que você mostrou:
            #
            # (Eadslab - slab_ref - O_atom_ref + correction)
            # * 23.06
            # --------------------------------------------------

            E_ads_kcal = (
                E_tot
                - (-187.1716267731465)
                - (-7.204)
                + (-5.525)
            ) * EV2KCAL


            # --------------------------------------------------
            # p_mean
            # --------------------------------------------------

            pmean = compute_p_mean(
                adslab
            )


            # --------------------------------------------------
            # imprime
            # --------------------------------------------------

            print(
                f"E_tot = {E_tot:+.9f} eV"
            )

            print(
                f"E_ads = {E_ads_kcal:+.3f} "
                "kcal/mol"
            )

            print(
                f"p_mean = {pmean:.6f} Å"
            )


            # --------------------------------------------------
            # escreve .dat
            # --------------------------------------------------

            fout.write(
                f"{dist:7.2f}   "
                f"{E_tot:+.9f}   "
                f"{E_ads_kcal:+.3f}   "
                f"{pmean:7.4f}\n"
            )

            fout.flush()


        except Exception as e:

            print(
                f"[ERRO] {fname}: {e}"
            )


# ==========================================================
# FINAL
# ==========================================================

print("\n========================================")
print("SCAN FINALIZADO")
print("========================================")

print(
    f"Resultados escritos em: {OUTFILE}"
)
