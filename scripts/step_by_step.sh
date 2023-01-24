DOWNLOAD_DIR=/home/dozeduck/workspace/database_alphafold/8.alterAlphaFolde_Database
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
sudo bash "${SCRIPT_DIR}/download_pdb_mmcif_uk_mirro.sh" "${DOWNLOAD_DIR}"

