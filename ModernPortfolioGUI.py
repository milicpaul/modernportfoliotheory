from anyio import sleep
from nicegui import ui, events
import Portfolio
import  assets
import ParallelComputing
import asyncio
import traceback
import os
from pathlib import Path
import copy

class Gui():
    def __init__(self):
        if os.name != "Windows" and os.name != 'nt':
            self.path = Path("/Users/paul/Documents/Modern Portfolio Theory Data/")
            self.path2 = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            self.path = Path("C:/Users/paul.milic/Modern Portfolio/")

        self.loader = ui.label('Modern Portfolio Theory')
        self.spinner = ui.spinner(size='lg', color='primary')
        self.spinner.visible = False  # Caché tant que pas en traitement
        with ui.row():
            self.nbOfSimulation = ui.input(label='Nb of portfolios simulation:', value='1')
            self.nbOfWeight = ui.input(label='Nb of weight by simulation:', value='10000')
        self.portfolioStructure = []
        self.fileData = []
        self.columnDefs = [{'field': 'File name'}, {'field': 'Number of assets', 'editable': True}]
        files = [f for f in os.listdir(self.path) if f.endswith('.pkl')]
        files.sort()
        for f in files:
            self.fileData.append({"File name": f, "Number of assets": 0})
        self.rowData = []
        with ui.splitter(horizontal=False, reverse=False, value=60,
                         on_change=lambda e: ui.notify(e.value)).style('width: 100%') as splitter:
            with splitter.before:
                self.aggrid = ui.aggrid({
                    'columnDefs': [
                        {"field": "returns"},
                        {"field": "volatility"},
                        {"field": "sharpe", "filter": 'agNumberColumnFilter', 'floatingFilter': True},
                        {"field": "returns lowest vol"},
                        {"field": "lowest volatility"},
                        {"field": "sharpe lowest vol"},
                    ],
                    'rowData': self.rowData
                }).classes('max-h-40')
            with splitter.after:
                self.aggrid2 = ui.aggrid({
                    'columnDefs':
                        self.columnDefs,
                    'rowData':
                        self.fileData
                }).classes('max-h-40')
                self.aggrid2.on('cellValueChanged', self._on_cell_clicked)
        self.a = ui.audio(self.path2 + 'mixkit-air-zoom-vacuum-2608.mp3')
        self.a.visible = False
        ui.switch('Dark', on_change=self.handle_theme_change)
        self.btnSimulate = ui.button('Simulate', on_click=self.Callback)
        ui.run()

    def update_number_of_assets(self, nbAssets, fileName):
        """
        Met à jour les valeurs 'Number of assets' pour les fichiers donnés.
        :param data_dict: Dictionnaire contenant 'rowData'
        :param updates: Dictionnaire { 'File name': 'nouvelle valeur' }
        """
        for row in self.fileData:
            file_name = row.get('File name')
            if file_name == fileName:
                row['Number of assets'] = nbAssets
                break
        return [[row['File name'], row['Number of assets']] for row in self.fileData]

    def _on_cell_clicked(self, event):
        self.portfolioStructure = self.update_number_of_assets(event.args['data']['Number of assets'], event.args['data']['File name'])


    def display_portfolio(self, bestPortfolio):
        self.rowData.append({"returns": round(bestPortfolio[2] * 100, 2), "volatility": round(bestPortfolio[3] * 100, 2),
         "sharpe": round(bestPortfolio[4], 2), "returns lowest vol": round(bestPortfolio[8] * 100, 2),
         "lowest volatility": round(bestPortfolio[7] * 100, 2), "sharpe lowest vol": round(bestPortfolio[9], 2)})
        self.aggrid.update()

    async def Simulate(self, nbOfSimulation, nbOfWeight):
        portfolio = Portfolio.ModernPortfolioTheory(nbOfSimulation, 2, 4)
        portfolio.nbOfSimulatedWeights = nbOfWeight
        data, isin = portfolio.BuilHeterogeneousPortfolio(self.portfolioStructure)
        showDensity = False
        isRandom = True
        bestPortfolios = ParallelComputing.Parallel.run_select_random_assets_parallel(portfolio, data, isin, nbOfSimulation, self.portfolioStructure, showDensity, isRandom, [])
        return bestPortfolios

    async def Callback(self):
        try:
            self.btnSimulate.disable()
            self.loader.text = "Processing"
            await asyncio.sleep(0.3)
            self.spinner.visible = True
            bestPortfolio = await self.Simulate(int(self.nbOfSimulation.value), int(self.nbOfWeight.value))
            self.loader.text = "Terminated"
        except Exception as e:
            print("💥 ERREUR pendant la simulation :", e)
            traceback.print_exc()
            self.loader.text = "Erreur !"
        finally:
            await ui.context.client.connected()
            await asyncio.sleep(2)
            self.btnSimulate.enable()
            self.spinner.visible = False
            self.loader.text = "Simulation terminée"  # Forcer le rafraîchissement en changeant le texte
            self.display_portfolio(bestPortfolio)
            self.a.play()
        return bestPortfolio

    def handle_theme_change(self, e: events.ValueChangeEventArguments):
        self.aggrid.classes(add='ag-theme-balham-dark' if e.value else 'ag-theme-balham',
                     remove='ag-theme-balham ag-theme-balham-dark')

gui = Gui()