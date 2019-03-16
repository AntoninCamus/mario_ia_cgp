# Mario CGP

## Installation
Pour utiliser ce programme il est obligatoire d'avoir les modules python suivants :
```
numpy
opencv-python
gym>=0.12.0
gym-super-mario-bros>=7.1.6
```
Pour faire fonctionner le programme avec MPI, il faut de plus installer le package "openmpi" et le module python "mpi4py".

## Utilisation simple

Voici la documentation d'usage :
```
usage: main.py [-h] [-d] [-r] [-m] [-p]

Cartesian Genetical Program playing Mario Bros :3

optional arguments:
  -h, --help     show this help message and exit
  -d, --debug    Affiche les textes de débogage
  -r, --render   Affiche les parties en cours
  -m, --mpi      Se lance en distribué MPI, incompatible avec le profiling
  -p, --profile  Profile un run, incompatible avec le lancement via mpi
```
Ainsi pour lancer simplement l'application il est possible de faire ```python ./main.py -d```.

## Utilisation avec MPI

Il est conseillé tout d'abord de remplir un hostfile sous la forme :
```
localhost slots=<nb slots>
```
en remplaçant ```<nb slots>``` par votre nombre de coeurs. Vous pouvez ensuite ajouter de la même manière toutes les autres machines distantes en ajoutant des lignes ```<ip> slots=<cores>```.

Ensuite, pour lancer le programme, utilisez la commande ```mpiexec --hostfile ./hostfile -n 4 python ./main.py --mpi --debug```
