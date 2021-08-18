#!/bin/sh

#SBATCH -p ProdQ
#SBATCH -t 03:30:00
#SBATCH -N 20
#SBATCH -A nuig02
#SBATCH -o output.txt
#SBATCH --mail-user=A.flanagan18@nuigalway.ie
#SBATCH --mail-type=BEGIN,END

cd /ichec/work/nuig02/aaronflano24/Capstone

module load conda/2
source activate /ichec/work/nuig02/aaronflano24/support/py38

python main.py
