#!/usr/bin/env python3

import glob
import re
import numpy as np
import os

from ase.constraints import FixAtoms, FixedPlane
from ase.optimize import BFGS
from ase.io import read, write

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


# ==========================================================
# CONSTANTES
# ==========================================================

EV2KCAL = 23.0605  # eV -> kcal/mol


# ==========================================================
# FUNÇÃO PARA EXTRAIR A DISTÂNCIA DO NOME DO ARQUIVO
#
# Exemplos:
# pt-h-0.00.xyz -> 0.00
# pt-h-1.50.xyz -> 1.50
# pt-h-6.00.xyz -> 6.00
# ==========================================================

def parse_dist(path: str):
    m = re.search(
        r"pt-h-([0-9]+(?:\.[0-9]+)?)\.xyz$",
        os.path.basename(path)
    )

    return float(m.group(1)) if m else None


# ==========================================================
# p_mean
#
# H em relação aos quatro Pt da camada superior.
# Índices ASE são 0-based.
# ==========================================================

TOP_PT_IDX = [12, 13, 14, 15]


def compute_p_mean(atoms):

    syms = atoms.get_chemical_symbols()
    pos = atoms.get_positions()

    h_idx = [
        i for i, symbol in enumerate(syms)
        if symbol == "H"
    ]

    if not h_idx:
        return float("nan")

    rH = pos[h_idx[0]]

    d = (
        (pos[TOP_PT_IDX] - rH) ** 2
    ).sum(axis=1) ** 0.5

    return float(d.mean())


# ==========================================================
# CARREGA MODELO FINE-TUNED
# ==========================================================

CHECKPOINT = (
    "/home/bruno/doutorado/fairchem/4-pt/resposta/"
    "finetuning-com-replay/checkpoint-nivel-dft/"
    "checkpoint-umlip/treino-proximo/teste-3/inference_ckpt.pt"
)

print("\nCarregando modelo...")
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

slab = read("pt-puro-2.xyz")

# fixa os primeiros 8 Pt
slab.set_constraint(
    FixAtoms(indices=range(8))
)

slab.calc = calc

opt_slab = BFGS(
    slab,
    logfile="pt-puro-opt.log"
)

opt_slab.run(
    fmax=0.01,
    steps=100
)

slab_e = float(
    slab.get_potential_energy()
)

print(
    f"Energia do slab puro (eV): "
    f"{slab_e:.6f}"
)


# ==========================================================
# ARQUIVO DE SAÍDA
# ==========================================================

OUTFILE = "adsorption_curve_h_sem_corr.dat"


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
    # LOCALIZA E ORDENA OS ARQUIVOS
    # ======================================================

    files = glob.glob(
        "pt-h-*.xyz"
    )

    files = [
        f for f in files
        if parse_dist(f) is not None
    ]

    files = sorted(
        files,
        key=parse_dist
    )

    print(
        f"\nEncontrados {len(files)} "
        "arquivos para o scan."
    )

    if len(files) == 0:
        raise RuntimeError(
            "Nenhum arquivo pt-h-*.xyz "
            "compatível foi encontrado."
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

            # ----------------------------------------------
            # lê estrutura
            # ----------------------------------------------

            adslab = read(fname)


            # ----------------------------------------------
            # restrições
            # ----------------------------------------------

            # fixa camada inferior
            fix_bottom = FixAtoms(
                indices=range(8)
            )


            # índice do H
            h_idx = [
                i
                for i, atom in enumerate(adslab)
                if atom.symbol == "H"
            ]


            if len(h_idx) == 0:
                raise RuntimeError(
                    f"Nenhum H encontrado em {fname}"
                )


            # índices dos Pt
            pt_idx = [
                i
                for i, atom in enumerate(adslab)
                if atom.symbol == "Pt"
            ]


            # mantém H fixo
            fix_Hz = FixAtoms(
                indices=h_idx
            )


            # Pt podem relaxar apenas no plano xy
            # deslocamento em z fica bloqueado
            fix_Ptz = FixedPlane(
                indices=pt_idx,
                direction=[0, 0, 1]
            )


            adslab.set_constraint(
                [
                    fix_bottom,
                    fix_Hz,
                    fix_Ptz
                ]
            )


            # ----------------------------------------------
            # calculator
            # ----------------------------------------------

            adslab.calc = calc


            # ----------------------------------------------
            # otimização
            # ----------------------------------------------

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


            # ----------------------------------------------
            # salva APENAS estrutura final
            # ----------------------------------------------

            final_traj = fname.replace(
                ".xyz",
                ".traj"
            )

            write(
                final_traj,
                adslab,
                format="traj"
            )


            # ----------------------------------------------
            # energia total
            # ----------------------------------------------

            E_tot = float(
                adslab.get_potential_energy()
            )


            # ----------------------------------------------
            # energia de adsorção
            #
            # Mantida exatamente a referência que você
            # estava usando no script anterior.
            # ----------------------------------------------

            E_ads_kcal = (
                E_tot
                - (-83.1879506493796)
                - (-3.4770)
                + (-2.239)
            ) * EV2KCAL


            # ----------------------------------------------
            # p_mean
            # ----------------------------------------------

            pmean = compute_p_mean(
                adslab
            )


            # ----------------------------------------------
            # imprime resultado
            # ----------------------------------------------

            print(
                f"E_tot = {E_tot:+.6f} eV"
            )

            print(
                f"E_ads = {E_ads_kcal:+.3f} "
                "kcal/mol"
            )

            print(
                f"p_mean = {pmean:.6f} Å"
            )


            # ----------------------------------------------
            # escreve no .dat
            # ----------------------------------------------

            fout.write(
                f"{dist:7.2f}   "
                f"{E_tot:+.6f}   "
                f"{E_ads_kcal:+.3f}   "
                f"{pmean:7.3f}\n"
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
