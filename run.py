import numpy as np
import numba as nb
from classes import *
import matplotlib.pyplot as plt
import time
starttime = time.time()

Exposant = 12
Taille = 32

#On ouvre le fichier
file = open("data.txt", "w")


A = []
E = []
A_tdc = []
E_tdc = []

grille = ising_aleatoire(1, Taille)

# Faire 1M rep Pour réchauffer
grille.simulation(1000000)
print('grille initialisée')

for k in range(31):

    Aimantation = Observable(Exposant)
    Energie = Observable(Exposant)
    print()
    print("temperature :", grille.temperature)

    for i in range(2**Exposant):
        # Faire 1k rep
        grille.simulation(1000)
        # print(round((time.time() - starttime), 3))
        Aimantation.ajout_mesure(grille.calcule_aimantation(),0)
        Energie.ajout_mesure(grille.calcule_energie(),0)
    grille.temperature += 0.1

    Aimantation_courante = Aimantation.moyenne()/Taille**2 
    Energie_courante = Energie.moyenne()/Taille**2 

    print(k,'/30','--- temps = ',round((time.time() - starttime), 3))
    print("aimantation: ",Aimantation_courante)
    print("energie: ",Energie_courante)
    # print("niveaux : ", random.random())

    file.writelines(str(Aimantation_courante) + '\t' + str(Energie_courante) + "\n")
    A.append(Aimantation_courante)
    E.append(Energie_courante)
    A_tdc.append(Aimantation.temps_correlation())
    E_tdc.append(Energie.temps_correlation())

    print('Temps: ',round((time.time() - starttime), 3))

file.close()

print('Temps final: ',round((time.time() - starttime), 3))

A = np.abs(A)
#affichage
fig, axs = plt.subplots(2)
fig.suptitle('Aimantation, Energie et Erreurs')

axs[0].plot([i for i in range(len(A))],A,label='Aimantation Absolue',linestyle='dotted',marker='+')
axs[0].plot([i for i in range(len(E))],E,label='Energie',linestyle='dotted',marker='+')

axs[1].plot([i for i in range(len(A_tdc))],A_tdc,label='Erreur sur l\'Aimantation',linestyle='dotted',marker='+')
axs[1].plot([i for i in range(len(E_tdc))],E_tdc,label='Erreur sur l\'Energie',linestyle='dotted',marker='+')

#affichage
fig, axs = plt.subplots(2)
fig.suptitle('Aimantation, Energie et Temps de corrélation')

axs[0].plot([1+i/10 for i in range(len(A))],A,label='Aimantation Absolue',linestyle='dotted',marker='+')
axs[0].plot([1+i/10 for i in range(len(E))],E,label='Energie',linestyle='dotted',marker='+')
plt.xlabel('Température')

axs[1].plot([1+i/10 for i in range(len(A_tdc))],A_tdc,label='τ sur l\'Aimantation',linestyle='dotted',marker='+')
axs[1].plot([1+i/10 for i in range(len(E_tdc))],E_tdc,label='τ sur1+ l\'Energie',linestyle='dotted',marker='+')
plt.xlabel('Température')

axs[0].legend()
axs[1].legend()
