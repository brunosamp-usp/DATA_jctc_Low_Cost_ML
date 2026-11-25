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

OUT_DAT = "pec_h_ft10_rpbe.dat"

TASK_NAME = "oc20"


# ==========================================================
# CONSTANTES PARA Eads
# ==========================================================

E_SLAB = -82.99535688284168
E_H = -3.477
F_CORR = -2.239

EV_TO_KCAL = 23.06


# ==========================================================
# CARREGA CHECKPOINT FT10
# ==========================================================

print("=" * 70)
print("Carregando checkpoint FT10")
print("=" * 70)

print(CHECKPOINT)

predictor = load_predict_unit(
    CHECKPOINT
)

calc = FAIRChemCalculator(
    predictor,
    task_name=TASK_NAME
)

print("Modelo carregado.\n")


# ==========================================================
# RECONHECE PASTAS DE DISTÂNCIA
#
# exemplos:
# 0.2
# 0.5
# 1.00
# 2.60
# 6.00
# ==========================================================

def get_distance_from_folder(folder_name):

    m = re.fullmatch(
        r"\d+(?:\.\d+)?",
        folder_name
    )

    if m is None:
        return None

    return float(folder_name)


# ==========================================================
# MAIN
# ==========================================================

def main():

    folders = []

    # ------------------------------------------------------
    # encontra todas as pastas numéricas
    # ------------------------------------------------------

    for path in BASE_DIR.iterdir():

        if not path.is_dir():
            continue

        dist = get_distance_from_folder(
            path.name
        )

        if dist is not None:

            folders.append(
                (
                    dist,
                    path.name,   # preserva 1.00, 1.10 etc.
                    path
                )
            )

    folders.sort(
        key=lambda x: x[0]
    )

    print(
        f"Pastas de distância encontradas: "
        f"{len(folders)}"
    )

    print()

    resultados = []


    # ======================================================
    # LOOP NAS DISTÂNCIAS
    # ======================================================

    for dist, dist_str, folder in folders:

        # exemplos:
        #
        # 0.2/pt-h-0.2-qe.traj
        # 1.00/pt-h-1.00-qe.traj

        qe_traj = (
            folder /
            f"pt-h-{dist_str}-qe.traj"
        )

        print("=" * 70)

        print(
            f"Distância = {dist:.3f} Å"
        )

        print(
            f"Arquivo = {qe_traj}"
        )


        # --------------------------------------------------
        # confere se existe
        # --------------------------------------------------

        if not qe_traj.exists():

            print(
                "[AVISO] Arquivo não encontrado. "
                "Pulando."
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


            # ==============================================
            # CÓPIA EXATA DA GEOMETRIA QE
            # ==============================================

            atoms_ft = atoms_qe.copy()

            atoms_ft.set_cell(
                atoms_qe.cell.copy()
            )

            atoms_ft.set_pbc(
                atoms_qe.pbc.copy()
            )


            # ==============================================
            # SINGLE POINT COM FT10
            # ==============================================

            atoms_ft.calc = calc

            E_tot = float(
                atoms_ft.get_potential_energy()
            )


            # ==============================================
            # ENERGIA DE ADSORÇÃO
            #
            # Eads =
            # (E_H/Pt - E_Pt - E_H + Fcorr) * 23.06
            # ==============================================

            E_ads = (
                E_tot
                - E_SLAB
                - E_H
                + F_CORR
            ) * EV_TO_KCAL


            # ==============================================
            # GUARDA RESULTADOS
            # ==============================================

            resultados.append(
                (
                    dist,
                    E_tot,
                    E_ads
                )
            )


            # ==============================================
            # PRINT
            # ==============================================

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
                f"[ERRO] {folder.name}: {exc}"
            )


    # ======================================================
    # ORDENA
    # ======================================================

    resultados.sort(
        key=lambda x: x[0]
    )


    # ======================================================
    # SALVA DAT
    # ======================================================

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


    # ======================================================
    # FINAL
    # ======================================================

    print()
    print("=" * 70)
    print("FINALIZADO")
    print("=" * 70)

    print(
        f"Pontos calculados: "
        f"{len(resultados)}"
    )

    print(
        f"Arquivo salvo: "
        f"{OUT_DAT}"
    )


if __name__ == "__main__":
    main()
