#!/usr/bin/env python3

import re
from pathlib import Path

from ase.io import read

from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit


# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

BASE_DIR = Path(".")

CHECKPOINT = (
    "/home/bruno/doutorado/fairchem/4-pt/resposta/finetuning-com-replay/checkpoint-nivel-dft/09-11-resposta/novo-teste-pertub-2-cl/inference_ckpt.pt"
)


OUT_DAT = "pec_o_ft10_rpbe.dat"

TASK_NAME = "oc20"


# ==========================================================
# CONSTANTES PARA Eads
# ==========================================================

E_SLAB = -186.7395999328
E_O = -7.204
F_CORR = -5.525

EV_TO_KCAL = 23.06


# ==========================================================
# CARREGA CHECKPOINT FT10
# ==========================================================

print("=" * 70)
print("Carregando checkpoint de 10 epochs")
print("=" * 70)

predictor = load_predict_unit(
    CHECKPOINT
)

calc = FAIRChemCalculator(
    predictor,
    task_name=TASK_NAME
)

print("Modelo carregado.\n")


# ==========================================================
# EXTRAI DISTÂNCIA
# ==========================================================

def get_distance_from_folder(folder_name):

    m = re.fullmatch(
        r"pt-o-(\d+(?:\.\d+)?)A",
        folder_name
    )

    if m is None:
        return None

    return float(
        m.group(1)
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    folders = []

    for path in BASE_DIR.iterdir():

        if not path.is_dir():
            continue

        dist = get_distance_from_folder(
            path.name
        )

        if dist is not None:

            folders.append(
                (dist, path)
            )

    folders.sort(
        key=lambda x: x[0]
    )

    print(
        f"Pastas encontradas: {len(folders)}"
    )

    print()

    resultados = []


    # ======================================================
    # LOOP NAS DISTÂNCIAS
    # ======================================================

    for dist, folder in folders:

        system_name = folder.name

        qe_traj = (
            folder /
            f"{system_name}-qe.traj"
        )

        print("=" * 70)
        print(
            f"Distância = {dist:.3f} Å"
        )

        print(
            f"Arquivo = {qe_traj}"
        )


        if not qe_traj.exists():

            print(
                "[AVISO] Arquivo não encontrado. Pulando."
            )

            continue


        try:

            # ==============================================
            # ÚLTIMA GEOMETRIA OTIMIZADA PELO QE
            # ==============================================

            atoms_qe = read(
                qe_traj,
                index=-1
            )

            atoms_ft = atoms_qe.copy()

            atoms_ft.set_cell(
                atoms_qe.cell.copy()
            )

            atoms_ft.set_pbc(
                atoms_qe.pbc.copy()
            )


            # ==============================================
            # SINGLE POINT FT10
            # ==============================================

            atoms_ft.calc = calc

            E_tot = float(
                atoms_ft.get_potential_energy()
            )


            # ==============================================
            # ENERGIA DE ADSORÇÃO
            # ==============================================

            E_ads = (
                E_tot
                - E_SLAB
                - E_O
                + F_CORR
            ) * EV_TO_KCAL


            resultados.append(
                (
                    dist,
                    E_tot,
                    E_ads
                )
            )


            print(
                f"E_tot FT10 = "
                f"{E_tot:.10f} eV"
            )

            print(
                f"E_ads FT10 = "
                f"{E_ads:.6f} kcal/mol"
            )


        except Exception as exc:

            print(
                f"[ERRO] {system_name}: {exc}"
            )


    # ======================================================
    # SALVA DAT
    # ======================================================

    resultados.sort(
        key=lambda x: x[0]
    )

    with open(
        OUT_DAT,
        "w",
        encoding="utf-8"
    ) as fout:

        fout.write(
            "# distance_A    "
            "E_FT10_eV    "
            "Eads_FT10_kcal_mol\n"
        )

        for dist, E_tot, E_ads in resultados:

            fout.write(
                f"{dist:10.3f}   "
                f"{E_tot:18.10f}   "
                f"{E_ads:18.8f}\n"
            )


    print()
    print("=" * 70)
    print("FINALIZADO")
    print("=" * 70)

    print(
        f"Pontos calculados: {len(resultados)}"
    )

    print(
        f"Arquivo salvo: {OUT_DAT}"
    )


if __name__ == "__main__":
    main()
