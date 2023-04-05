import numpy as np
import numba as nb
import math
import random

@nb.jit(nopython=True)
def ising_aleatoire(temperature, taille):
    """ Génére une grille aléatoire de spins.
    Arguments
    ---------
    temperature : Température du système.
    taille : La grille a une dimension taille x taille.
    """

    spins = np.random.randint(0, 2, (taille, taille))
    spins = 2 * spins - 1
    return Ising(temperature, spins)

@nb.experimental.jitclass([
    ("temperature", nb.float64),
    ("spins", nb.int64[:, :]),
    ("taille", nb.int64),
    ("energie", nb.int64),
    ("aimantation", nb.int64),
])
class Ising:
    """ Modèle de Ising paramagnétique en 2 dimensions.
    Représente une grille de spins classiques avec un couplage J = +1 entre
    les premiers voisins.
    Arguments
    ---------
    temperature : Température du système.
    spins : Tableau carré des valeurs de spins
    """

    def __init__(self, temperature, spins):
        self.temperature = temperature
        self.spins = spins
        self.taille = np.shape(spins)[0]
        self.energie = self.calcule_energie()
        self.aimantation = self.calcule_aimantation()

    def difference_energie(self, x, y):
        """Retourne la différence d'énergie si le spin à la position (x, y)
        était renversé.
        """
        n = self.taille
        delta = 2 * self.spins[x, y] * (
          self.spins[(x - 1) % n, y]
        + self.spins[(x + 1) % n, y]
        + self.spins[x, (y - 1) % n]
        + self.spins[x, (y + 1) % n]   )
        return delta

    def iteration_aleatoire(self,x,y):
        """Renverse un spin aléatoire avec probabilité ~ e^(-ΔE * T).
        Cette fonction met à jour la grille avec la nouvelle valeur de spin
        """
        delta_energie = self.difference_energie(x, y)
        proba = np.exp( - delta_energie / self.temperature)
        if delta_energie <= 0 or np.random.rand() < proba:
            self.spins[x, y] *= -1
            # self.energie += delta_energie
            # self.aimantation += self.spins[x,y]

    def simulation(self, nombre_iterations):
        """Simule le système en effectuant des itérations aléatoires.
        """
        for _ in range(nombre_iterations):
            x = np.random.randint(self.taille)
            y = np.random.randint(self.taille)
            self.iteration_aleatoire(x,y)

        self.aimantation = self.calcule_aimantation()
        self.energie = self.calcule_energie()


    def calcule_energie(self):
        """Retourne l'énergie actuelle de la grille de spins."""
        energie = 0
        n = self.taille
        for x in range(n):
            for y in range(n):
                energie -= self.spins[x, y] * self.spins[(x + 1) % n, y]
                energie -= self.spins[x, y] * self.spins[x, (y + 1) % n]
        return energie
    
    def calcule_aimantation(self):
        """Retourne l'aimantation actuelle de la grille de spins."""
        aimantation = 0
        return np.sum(self.spins)

class Observable:
    """Utilise la méthode du binning pour calculer des statistiques
    pour un observable.
    Arguments
    ---------
    nombre_niveaux : Le nombre de niveaux pour l'algorithme. Le nombre
    de mesures est exponentiel selon le nombre de niveaux.
    """
    
    def __init__(self, nombre_niveaux):
        self.nombre_niveaux = nombre_niveaux

        self.nombre_valeurs = np.zeros(nombre_niveaux + 1, int)
        self.sommes = np.zeros(nombre_niveaux + 1)
        self.sommes_carres = np.zeros(nombre_niveaux + 1)

        self.valeurs_precedentes = np.zeros(nombre_niveaux + 1)

        self.niveau_erreur = self.nombre_niveaux - 6

    def ajout_mesure(self, valeur, niveau=0):
        """Ajoute une mesure.
        Arguments
        ---------
        valeur : Valeur de la mesure.
        niveau : Niveau à lequel ajouter la mesure. Par défaut,
                 le niveau doit toujours être 0. Les autres niveaux
                 sont seulement utilisé pour la récursion.
        """
        self.nombre_valeurs[niveau] += 1
        self.sommes[niveau] += valeur
        self.sommes_carres[niveau] += valeur**2
        # Si un nombre pair de valeurs a été ajouté,
        # on peut faire une simplification.
        if self.nombre_valeurs[niveau] % 2 == 0:
            moyenne = (valeur + self.valeurs_precedentes[niveau]) / 2
            self.ajout_mesure(moyenne, niveau + 1)
        else:
            self.valeurs_precedentes[niveau] = valeur

            
    def est_rempli(self):
        """Retourne vrai si le binnage est complété."""
        return len(self.nombre_valeurs) == self.nombre_niveaux

    def erreur(self):
        """Retourne l'erreur sur la mesure moyenne de l'observable.
        Le dernier niveau doit être rempli avant d'utiliser cette fonction.
        """
        erreurs = np.zeros(self.nombre_niveaux + 1)
        for niveau in range(self.niveau_erreur + 1):
            erreurs[niveau] = np.sqrt(
                (
                    self.sommes_carres[niveau]
                    - self.sommes[niveau]**2 / self.nombre_valeurs[niveau]
                ) / (
                    self.nombre_valeurs[niveau]
                    * (self.nombre_valeurs[niveau] - 1)
                )
            )
        return erreurs[self.niveau_erreur]

    def temps_correlation(self):
        """Retourne le temps de corrélation."""
        variance = np.sqrt(
                self.sommes_carres[0]
                - self.sommes[0] ** 2 / self.nombre_valeurs[0]
            ) / (
                self.nombre_valeurs[0]
                * (self.nombre_valeurs[0] - 1)
            )

        return ((self.erreur()**2/variance**2)-1)/2

    def moyenne(self):
        """Retourne la moyenne des mesures."""
        return self.sommes[self.nombre_niveaux]