import asyncio
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
from nicegui import ui

class Critical():
    def __init__(self, simulationNumber):
        self.simulationNumber = simulationNumber
        self.lock = Lock()

    def SetSimulationNumber(self, j):
        with self.lock:
            self.simulationNumber = j
            print(f"Critical Set: {self.simulationNumber}, Critical Id: {id(self)}")

    def GetSimulationNumber(self):
        with self.lock:
            print(f"Critical Get: {self.simulationNumber}, Critical Id: {id(self)}")
            return self.simulationNumber

class Parallel():
    def __init__(self, critical):
        self.message_queue = asyncio.Queue()
        self.critical = critical
        # Queue pour la GUI (async)

    async def start_listener(self, one):
        while True:
            print("Listener:", self.critical.GetSimulationNumber())
            if self.critical.GetSimulationNumber() > 0:
                print('on est la')
                await asyncio.create_task(self.message_queue.put(f"Simulation Number"))
            message = await self.message_queue.get()
            if message is None:
                break  # Fin de l'écoute
            else:
                print(message)
            # Mettez à jour l'UI avec le message
            with one:
                ui.notify(message)  # Exemple de mise à jour de texte
                await asyncio.sleep(0.5)  # Pour éviter un 'busy wait'


    @staticmethod
    def compute_volatility(stat, portfolios, data):
        results = []
        volatilities = []
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            chunk = int(len(portfolios) / multiprocessing.cpu_count())
            for i in range(multiprocessing.cpu_count()):
                subPortfolios = portfolios[i * chunk:(i + 1) * chunk]
                volatility = executor.submit(stat.EstimateVolatility, subPortfolios, data)
                volatilities.append(volatility)
            for volatility in as_completed(volatilities):
                results.append(volatility.result())
                print("[INFO] Un thread a terminé son exécution.")
        print("[INFO] Tous les threads ont terminé !")
        return results[np.argmin(list(zip(*results))[0])]

    async def run_select_random_assets_parallel(self, portfolio, data, isin, nbOfSimulation, percentage, showDensity,
                                                isRandom, localPortfolio=[]):
        results = []
        portfolio.SetParallel(self)
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for i in range(multiprocessing.cpu_count()):
                future = executor.submit(portfolio.SelectRandomAssets, data, isin, nbOfSimulation, percentage, i,
                                         showDensity, isRandom, localPortfolio)
                futures.append(future)
                await asyncio.create_task(self.message_queue.put("Thread started !")) # thread-safe

            for future in as_completed(futures):
                results.append(future.result())
                await asyncio.create_task(self.message_queue.put("Thread ended !"))
                print("[INFO] Un thread a terminé son exécution.")

        print("[INFO] Tous les threads ont terminé !")
        return results[np.argmax(list(zip(*results))[4])]
