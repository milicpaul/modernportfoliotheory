import asyncio
import time
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from nicegui import ui


class Parallel():
    def __init__(self):
        self.message_queue = asyncio.Queue()

    async def start_listener(self, gui):
        """Tâche qui affiche les notifications dès qu'il y a un message dans la queue"""
        while True:
            msg = await self.message_queue.get()
            with gui.aggrid:
                ui.notify(msg)
            await asyncio.sleep(0.1)

    @staticmethod
    def compute_volatility(stat, portfolios, data):
        results = []
        volatilities = []
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            chunk = int(len(portfolios)/multiprocessing.cpu_count())
            for i in range(multiprocessing.cpu_count()):
                subPortfolios = portfolios[i * chunk:(i + 1) * chunk]
                volatility = executor.submit(stat.EstimateVolatility, subPortfolios, data)
                volatilities.append(volatility)
            for volatility in as_completed(volatilities):
                results.append(volatility.result())
                print("[INFO] Un thread a terminé son exécution.")
        print("[INFO] Tous les threads ont terminé !")
        return results[np.argmin(list(zip(*results))[0])]

    async def run_select_random_assets_parallel(self, portfolio, data, isin, nbOfSimulation, percentage, showDensity, isRandom, localPortfolio=[]):
        results = []
        # Création d'un ThreadPoolExecutor pour gérer les threads
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:  # 8 threads (modifiable)
            futures = []
            # Lancer plusieurs simulations en parallèle
            for i in range(multiprocessing.cpu_count()):
                future = executor.submit(portfolio.SelectRandomAssets, data, isin, nbOfSimulation, percentage, i, showDensity, isRandom, localPortfolio)
                futures.append(future)
                await self.message_queue.put(f"Thread started !")
                time.sleep(0.1)
            # Récupération des résultats au fur et à mesure
            for future in as_completed(futures):
                results.append(future.result())
                await self.message_queue.put(f"Thread ended !")
                time.sleep(0.1)
                print("[INFO] Un thread a terminé son exécution.")
        print("[INFO] Tous les threads ont terminé !")
        return results[np.argmax(list(zip(*results))[4])]