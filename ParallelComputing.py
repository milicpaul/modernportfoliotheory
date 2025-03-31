import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed


class Parallel():

    @staticmethod
    def run_select_random_assets_parallel(portfolio, data, isin, nbOfSimulation, percentage, showDensity, pUtilities):
        results = []
        # Création d'un ThreadPoolExecutor pour gérer les threads
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:  # 4 threads (modifiable)
            futures = []
            # Lancer plusieurs simulations en parallèle
            for _ in range(multiprocessing.cpu_count()):
                future = executor.submit(portfolio.SelectRandomAssets, data, isin, nbOfSimulation, percentage, showDensity)
                futures.append(future)

            # Récupération des résultats au fur et à mesure
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print("[INFO] Un thread a terminé son exécution.")
        print("[INFO] Tous les threads ont terminé !")
        return results[np.argmax(list(zip(*results))[4])]