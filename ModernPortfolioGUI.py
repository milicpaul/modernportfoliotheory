from anyio import sleep
from nicegui import ui, events
import Portfolio
import PortfolioUtilities
import ParallelComputing
import asyncio
import traceback
import os
from pathlib import Path
import KellyPortfolio

class Gui():
    def __init__(self):
        if os.name != "Windows" and os.name != 'nt':
            self.path = Path("/Users/paul/Documents/Modern Portfolio Theory Data/")
            self.path2 = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            self.path = Path("C:/Users/paul.milic/Modern Portfolio/")
        self.parallelComputing = ParallelComputing.Parallel()
        self.portfolioStructure = []
        self.fileData = []
        self.assetsName = []
        self.fullAssetsList = []
        self.rowData = []
        self.columnDefs = [{'field': 'File name'}, {'field': 'Number of assets', 'editable': True}]
        with ui.tabs().classes('w-full') as tabs:
            self.one = ui.tab('Simulation')
            two = ui.tab('List of assets')
        with ui.tab_panels(tabs, value=two).classes('w-full'):
            with ui.tab_panel(self.one):
                with ui.row():
                    self.nbOfSimulation = ui.input(label='Nb of portfolios simulation:', value='1')
                    self.nbOfWeight = ui.input(label='Nb of weight by simulation:', value='10000')
                    self.spinner = ui.spinner(size='lg', color='primary')
                    self.spinner.visible = False  # Caché tant que pas en traitement
                    self.kelly = ui.switch('Kelly optimization', on_change=self.ChangeValue)
                    self.Robust = ui.switch('Robust optimization')

                files = [f for f in os.listdir(self.path) if f.endswith('.pkl')]
                files.sort()
                for f in files:
                    self.fileData.append({"File name": f, "Number of assets": 0})
                with ui.splitter(horizontal=False, reverse=False, value=40,
                                 on_change=lambda e: ui.notify(e.value)).style('width: 100%') as splitter:
                    with splitter.before:
                        self.aggrid = ui.aggrid({
                            'columnDefs': [
                                {"field": "returns"},
                                {"field": "volatility"},
                                {"field": "sharpe", "filter": 'agTextColumnFilter', 'floatingFilter': True},
                                {"field": "returns lowest vol"},
                                {"field": "lowest volatility"},
                                {"field": "sharpe lowest vol"},
                            ],
                            'rowData': self.rowData
                        }).classes('max-h-40').style('width: 100%')
                        self.finalPortfolio = ui.aggrid({
                            'columnDefs': [
                                {"field": "Asset name"},
                                {"field": "ISIN"},
                                {"field": "Weight"}
                            ],
                            'rowData': self.assetsName
                        }).classes('max-h-40').style('width: 100%')
                        ui.switch('Dark', on_change=self.handle_theme_change)
                        self.btnSimulate = ui.button('Simulate', on_click=self.Callback)
                    with splitter.after:
                        self.aggrid2 = ui.aggrid({
                            'columnDefs':
                                self.columnDefs,
                            'rowData':
                                self.fileData
                        }).classes('max-h-80')
                        self.aggrid2.on('cellValueChanged', self._on_cell_clicked)
                        self.myGraph = ui.matplotlib(figsize=(9, 5))
                        self.ax = self.myGraph.figure.gca()
                        self.ax.set_title("Données issues d'un DataFrame")
                        self.ax.set_xlabel("x")
                        self.ax.set_ylabel("Valeurs")
                self.a = ui.audio(self.path2 + 'mixkit-air-zoom-vacuum-2608.wav')
                self.a.visible = False
            with ui.tab_panel(two):
                self.fullAssets = ui.aggrid({
                    'columnDefs': [
                        {"field": "Name", "filter": 'agTextColumnFilter', 'floatingFilter': False, 'checkboxSelection': True},
                        {"field": "ISIN", "filter": 'agTextColumnFilter', 'floatingFilter': False},
                        {"field": "Volatility"},
                        {"field": "Sum positive"},
                        {"field": "Sum negative"},
                        {"field": "File name"},
                    ],
                    'rowData': self.fullAssetsList
                }).classes('max-h-300').style('width: 100%')
                tabs.set_value(('Simulation'))
        self.GetAllAssetsList()

    def GetAllAssetsList(self):
        portfolioUtilities = PortfolioUtilities.PortfolioUtilities()
        isin = portfolioUtilities.GetAllAssets()
        for i in isin:
            self.fullAssetsList.append({"Name": i[0], "ISIN": i[1], "Volatility": i[2], "Sum positive": i[3], "Sum negative": i[4], "File name": i[5]})
        self.fullAssets.update()

    def ChangeValue(self, event):
        colManager = PortfolioUtilities.ColumnManager(self.finalPortfolio.props['options']['columnDefs'])
        if event.value:
            if not colManager.has('Kelly weight'):
                self.finalPortfolio.props['options']['columnDefs'] = colManager.add_after("Weight", {"field": "Kelly weight"})
        else:
            if colManager.has(('Kelly weight')):
                self.finalPortfolio.props['options']['columnDefs'] = colManager.remove_all("Kelly weight")



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
        self.finalPortfolio.clear()
        self.assetsName.clear()
        portfolioUtilities = PortfolioUtilities.PortfolioUtilities()
        self.rowData.append({"returns": round(bestPortfolio[2] * 100, 2), "volatility": round(bestPortfolio[3] * 100, 2),
         "sharpe": round(bestPortfolio[4], 2), "returns lowest vol": round(bestPortfolio[8] * 100, 2),
         "lowest volatility": round(bestPortfolio[7] * 100, 2), "sharpe lowest vol": round(bestPortfolio[9], 2)})
        assetsDescription = portfolioUtilities.ReturnAssetDescription(bestPortfolio[0])
        i = 0
        for a in assetsDescription:
            if self.kelly.value == True:
                self.assetsName.append({"Asset name": a, "Weight": bestPortfolio[1][i], "ISIN": bestPortfolio[0][i], "Kelly weight": bestPortfolio[-1][i]})
            else:
                self.assetsName.append({"Asset name": a, "ISIN": bestPortfolio[0][i], "Weight": bestPortfolio[1][i]})
            i += 1
        self.ax.clear()
        localDef = bestPortfolio[6][bestPortfolio[0]]
        for column in localDef.columns:
            self.ax.plot(localDef.index, localDef[column], label=column)
        self.ax.figure.canvas.draw()
        self.ax.legend()
        self.myGraph.update()
        self.finalPortfolio.update()
        self.aggrid.update()

    async def Simulate(self, nbOfSimulation, nbOfWeight):
        portfolio = Portfolio.ModernPortfolioTheory(nbOfSimulation, 2, 4)
        portfolio.nbOfSimulatedWeights = nbOfWeight
        data, isin = portfolio.BuilHeterogeneousPortfolio(self.portfolioStructure)
        if len(isin) == 0:
            with self.fullAssets:
                ui.notify("No Assets Class Selected")
                return
        showDensity = False
        isRandom = True
        bestPortfolios = await self.parallelComputing.run_select_random_assets_parallel(portfolio, data, isin, nbOfSimulation, self.portfolioStructure, showDensity, isRandom, [])
        if self.kelly.value == True:
            kelly = KellyPortfolio.KellyCriterion()
            kellyResult =  kelly.SolveKellyCriterion(bestPortfolios[5], len(bestPortfolios[5].columns))
            bestPortfolios.append(list(kellyResult))
        return bestPortfolios

    async def Callback(self):
        try:
            self.btnSimulate.disable()
            self.spinner.visible = True
            await asyncio.sleep(0.3)
            bestPortfolio = await self.Simulate(int(self.nbOfSimulation.value), int(self.nbOfWeight.value))
        except Exception as e:
            print("💥 ERREUR pendant la simulation :", e)
            traceback.print_exc()
        finally:
            await ui.context.client.connected()
            await asyncio.sleep(1)
            self.btnSimulate.enable()
            self.spinner.visible = False
            self.display_portfolio(bestPortfolio)
            self.a.play()
        return bestPortfolio

    def handle_theme_change(self, e: events.ValueChangeEventArguments):
        self.aggrid.classes(add='ag-theme-balham-dark' if e.value else 'ag-theme-balham',
                     remove='ag-theme-balham ag-theme-balham-dark')

gui = Gui()
# ⏳ Démarre le listener dès que possible
ui.timer(0.1, lambda: asyncio.create_task(gui.parallelComputing.start_listener(gui)), once=True)
ui.run()
