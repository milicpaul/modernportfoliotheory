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
        cpu_count = multiprocessing.cpu_count()
        """Exécute la simulation dans plusieurs processus."""
        workers = [Process(target=portfolio.SelectRandomAssets, args=(data, isin, nbOfSimulation,
                           percentage, self.queueResults, self.queueMessages, self.event, showDensity, isRandom, dateFrom, dateTo)) for i in range(multiprocessing.cpu_count() - 1)]
        i = 0
        for w in workers:
            self.queueMessages.put(f"[Parallel Computing]Starting Process {os.getpid()}")
            w.start()
        for w in workers:
            w.join()
        results = []
        while not self.queueResults.empty():
            try:
                results.append(self.queueResults.get())
            except:
                a=1
        # Calcul du meilleur portefeuille
        best_portfolio = results[np.argmax([r[4] for r in results])]
        return best_portfolio


