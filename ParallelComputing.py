import multiprocessing
import time
from multiprocessing import Process, Event, Queue, set_start_method, Manager
import numpy as np
import os

class Parallel:

    def __init__(self):
        self.manager = Manager()
        self.queueResults = self.manager.Queue()
        self.queueMessages = self.manager.Queue()
        self.event = Event()


    def run_select_random_assets_parallel(self, portfolio, data, isin, nbOfSimulation,
                                          percentage, showDensity, isRandom, dateFrom, dateTo, localPortfolio=None):
        if localPortfolio is None:
            localPortfolio = []
        """Exécute la simulation dans plusieurs processus."""
        workers = [Process(target=portfolio.SelectRandomAssets,
                           args=(data, isin, nbOfSimulation, percentage, self.queueResults,
                                 self.queueMessages, self.event, showDensity, isRandom)) for i in range(multiprocessing.cpu_count() - 2)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        results = []
        while not self.queueResults.empty():
            try:
                res = self.queueResults.get()
                if len(res[0]) > 0:
                    results.append(res)
            except:
                pass
        # Calcul du meilleur portefeuille
        try:
            best_portfolio = results[np.argmax([r[4] for r in results])]
        except Exception as e:
            pass
        return best_portfolio


