#! /usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import math
import os
import csv
import pprint

import numpy
from scipy import signal
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
test_csv = os.path.join(current_dir, "ressources", "ludo_demarche_acce_100_10.csv")


class StepDetector():

    def __init__(self, seuil=None):
        self.time_array = None
        self.x_array = None
        self.y_array = None
        self.z_array = None
        self.total_array = None
        self.signal_filtre = None
        self.seuil = seuil

    def extraction_csv_donnees(self, path_file):
        with open(path_file, 'r') as csv_file:
            csv_data = csv.reader(csv_file, delimiter=',')
            en_tete = next(csv_data)
            temps = []
            rs = []
            for ligne in csv_data:
                temps.append(float(ligne[0]))
                x = (float(ligne[1]))
                y = (float(ligne[2]))
                z = (float(ligne[3]))
                r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
                rs.append(r)

            self.total_array = numpy.array(rs)
            self.time_array = numpy.array(temps)
            plt.plot(self.time_array, self.total_array)
            plt.title("Signal de l'accelerometre")
            plt.xlabel("temps [s]")
            plt.ylabel("acceleration")
            plt.show()

    def filtre_passe_bas(self, ordre, fe, fc):
        # obtention de la frequence de Nyquist
        fnyq = float(fe / 2)
        fc = float(fc / fnyq)
        b, a = signal.butter(ordre, fc, btype='low', analog=False)
        return b, a

    def filtre_signal(self, ordre, fe, fc):
        b, a = self.filtre_passe_bas(ordre, fe, fc)
        self.signal_filtre = signal.filtfilt(b, a, self.total_array)

        plt.plot(self.time_array, self.signal_filtre)
        plt.title("Signal de l'accelerometre filtre")
        plt.xlabel("temps [s]")
        plt.ylabel("acceleration")
        plt.show()

    def detection_pas_seuil(self):
        flag = 1
        pas = 0
        cpt = 0
        pas_infos = []

        if self.seuil is None:
            self.seuil = numpy.mean(self.signal_filtre) * 1.3

        for i,data in enumerate(self.signal_filtre):
            if (data > self.seuil) and (flag == 1):
                pas = pas + 1
                flag = 0
                cpt = 0
                pas_info = {
                    "tps": self.time_array[i],
                    "val": self.signal_filtre[i],
                }
                pas_infos.append(pas_info)
            cpt = cpt + 1
            if cpt == 50:  # On attend 50ms,afin que la courbe de la force G en fonction du temps passe en dessous de 1.25
                cpt = 0
                flag = 1  # On remet le flag a 1, pour pouvoir recompter le nb de pas

            # On affiche le nombre de pas
        print("Nombre de pas: {} par detection par seuil constant".format(pas))

        tps = [pa['tps'] for pa in pas_infos]
        val = [pa['val'] for pa in pas_infos]
        plt.plot(self.time_array, self.signal_filtre, 'b-', linewidth=2)
        plt.plot(tps, val, 'go')
        plt.title(" Detection de pas")
        plt.xlabel('Temps [sec]')
        plt.grid()
        plt.legend()
        plt.show()

        return pas_infos

    def detection_pas_adaptative(self):
        etat_precedent = None
        etat_actuel = None
        pique_precedent = None
        creux_precedent = None
        pique_creux = []

        if self.seuil is None:
            self.seuil = numpy.mean(self.signal_filtre) * 1.3

        print("seuil: {}".format(self.seuil))

        acce_sum = 1
        acce_count = 10

        flag = True
        for i, point in enumerate(self.signal_filtre):

            if point < self.seuil and etat_precedent is not None:
                etat_actuel = 'creux'
                if creux_precedent is None or point < creux_precedent["val"]:
                    creux_precedent = {
                        "tps": self.time_array[i],
                        "val": self.signal_filtre[i],
                        "min_max": "min"
                    }
            elif point > self.seuil:
                etat_actuel = 'pique'
                if pique_precedent is None or point > pique_precedent["val"]:
                    pique_precedent = {
                        "tps": self.time_array[i],
                        "val": self.signal_filtre[i],
                        "min_max": "max"
                    }

            if etat_actuel is not etat_precedent:

                if etat_precedent is 'creux':
                    if pique_precedent:
                        acceleration = pique_precedent['val'] - creux_precedent['val']
                        if flag is True:
                            acceleration_moyenne = acce_sum / acce_count  # acceleration
                            flag = False
                        else:
                            acceleration_moyenne = acce_sum / acce_count

                        if acceleration > acceleration_moyenne * .65:
                            acce_sum += acceleration_moyenne * acceleration
                            acce_count += 1
                            pique_creux.append(creux_precedent)
                    creux_precedent = None

                if etat_precedent is 'pique':
                    pique_precedent = None

            etat_precedent = etat_actuel

        return numpy.array(pique_creux)

    def nombre_de_pas(self):
        pas = self.detection_pas_adaptative()
        tps = [pa['tps'] for pa in pas]
        val = [pa['val'] for pa in pas]
        print("Nombre de pas: {} par detection par seuil variable".format(len(pas)))

        plt.plot(self.time_array, self.signal_filtre, 'b-', linewidth=2)
        plt.plot(tps, val, 'ro')
        plt.title(" Detection de pas")
        plt.xlabel('Temps [sec]')
        plt.grid()
        plt.legend()
        plt.show()


def start_podometer():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-f", "--fichier", dest="fichier",
                            help="mettre le nom du fichier a tester disponible dans ressources ")
    arg_parser.add_argument("-s", "--seuil", dest="seuil", type=float,
                            help="rentrer le seuil voulu")
    arguments = arg_parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_csv = os.path.join(current_dir, "ressources", arguments.fichier)

    if arguments.seuil is None:
        DETECTOR = StepDetector()

    else:
        DETECTOR = StepDetector(arguments.seuil)

    DETECTOR.extraction_csv_donnees(test_csv)
    DETECTOR.filtre_signal(3, 100, 3.6)
    DETECTOR.detection_pas_seuil()
    DETECTOR.nombre_de_pas()


if __name__ == '__main__':
    #start_podometer()

    DETECTOR = StepDetector(10)
    DETECTOR.extraction_csv_donnees(test_csv)
    DETECTOR.filtre_signal(3, 100, 3.6)
    DETECTOR.nombre_de_pas()
    DETECTOR.detection_pas_seuil()
