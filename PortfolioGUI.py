from tkinter import Event

import pandas as pd
from datetime import datetime
from nicegui import ui, events
import Portfolio
import PortfolioUtilities
import ParallelComputing
import traceback
import os
from pathlib import Path
import KellyPortfolio
from multiprocessing import Process, Event, Queue, set_start_method
import threading
import time
import multiprocessing

class Gui():

    def __init__(self, parallel: ParallelComputing):
        self.parallel = parallel
        if os.name != "Windows" and os.name != 'nt':
            self.path = Path("/Users/paul/Documents/Modern Portfolio Theory Data/")
            self.workingPath = "/Users/paul/Documents/Modern Portfolio Theory Data/Data/"
            self.path2 = "/Users/paul/Documents/Modern Portfolio Theory Data/"
        else:
            self.path = Path("C:/Users/paul.milic/Modern Portfolio/")
            self.workingPath = "C:/Users/paul.milic/Modern Portfolio/Data"
        self.portfolioStructure = []
        self.fileData = []
        self.assetsName = []
        self.fullAssetsList = []
        self.rowData = []
        if os.path.exists(self.workingPath + "results.pkl"):
            self.results = pd.read_pickle(self.workingPath + "results.pkl")

        self.columnDefs = [{'field': 'File name'}, {"field": "Selection", 'editable': True}, {'field': 'Number of assets'}, {'field': 'Size'}]
        with ui.tabs().classes('w-full') as tabs:
            self.one = ui.tab('Simulation')
            two = ui.tab('List of assets')
        with ui.tab_panels(tabs, value=two).classes('w-full'):
            with ui.tab_panel(self.one):
                with ui.row():
                    self.nbOfSimulation = ui.input(label='Nb of portfolios simulation:', value='5')
                    self.nbOfWeight = ui.input(label='Nb of weight by simulation:', value='10000')
                    self.spinner = ui.spinner(size='lg', color='primary')
                    self.spinner.visible = False  # Caché tant que pas en traitement
                    self.kelly = ui.switch('Kelly optimization', on_change=self.ChangeValue)
                    self.Robust = ui.switch('Robust optimization', on_change=self.ChangeValue)
                    self.Sound = ui.switch("Sound", on_change=self.ChangeValue)

                files = [f for f in os.listdir(self.path) if f.endswith('.pkl')]
                files.sort()
                for f in files:
                    df = pd.read_pickle(self.path2 + f)
                    self.fileData.append({"File name": f, "Selection": 0, "Number of assets": len(df.columns), "Size": self.taille_lisible(os.path.getsize(self.path2 + f))})
                with ui.splitter(horizontal=False, reverse=False, value=40,
                                 on_change=lambda e: ui.notify(e.value)).style('width: 100%') as self.splitter:
                   with self.splitter.before:
                        self.aggrid = ui.aggrid({
                            'columnDefs': [
                                {"field": "returns"},
                                {"field": "volatility"},
                                {"field": "sharpe", "filter": 'agTextColumnFilter', 'floatingFilter': True},
                                {"field": "returns lowest vol"},
                                {"field": "lowest volatility"},
                                {"field": "sharpe lowest vol"},
                                {"field": "timestamp"}
                            ],
                            'rowData': self.rowData
                        }).classes('max-h-50').style('width: 100%').on('cellClicked', lambda event: self.FindPortfolio(event))
                        self.finalPortfolio = ui.aggrid({
                            'columnDefs': [
                                {"field": "Asset name"},
                                {"field": "ISIN"},
                                {"field": "Weight"}
                            ],
                            'rowData': self.assetsName
                        }).classes('max-h-50').style('width: 100%')
                        self.EventsTable()
                        ui.switch('Dark', on_change=self.handle_theme_change)
                        self.btnSimulate = ui.button('Simulate', on_click=self.Callback)
                   with self.splitter.after:
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
        if os.path.exists(self.workingPath + "results.pkl"):
            a=1
            self.RefreshRowData()
        #self.GetAllAssetsList()

    def EventsTable(self):
        columns = [
            {'name': 'origin', 'label': 'Origin', 'field': 'origin', 'required': True, 'align': 'left', 'sortable': True},
            {'name': 'event', 'label': 'Event', 'field': 'event', 'sortable': True},
        ]
        rows = [
            {'origin': 'Alice', 'event': 18},
            {'origin': 'Bob', 'event': 21},
            {'origin': 'Carol', 'event': 10},
        ]
        self.eventsTable = ui.table(columns=columns, rows=rows, row_key='name')

    def taille_lisible(self, octets):
        for unite in ['B', 'Ko', 'Mo', 'Go', 'To']:
            if octets < 1024:
                return f"{octets:.2f} {unite}"
            octets /= 1024
        return f"{octets:.2f} Po"  # Si t’es dans l’espace

    def chercher_rec(self, obj, cible):
        if isinstance(obj, list):
            for el in obj:
                if self.chercher_rec(el, cible):
                    return True
        else:
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] == cible
            else:
                return False
        return False

    def FindPortfolio(self, event):
        pu = PortfolioUtilities.PortfolioUtilities()
        self.assetsName.clear()
        mainRow = self.results[self.results["Portfolio"].apply(lambda x: self.chercher_rec(x, event.args['data']['timestamp']))]
        row = list(mainRow.Portfolio)[0]
        mainRow = list(mainRow.Portfolio)
        i = 0
        for isin in row[0]:
            self.assetsName.append({"Asset name": pu.ReturnAssetDescription([isin])[0], "ISIN": isin, "Weight": row[1][i]})
            i += 1
        self.finalPortfolio.update()
        self.displayGraph(row[6])

    def RefreshRowData(self):
        for index, row in self.results.iterrows():
            try:
                p = row.Portfolio
                self.rowData.append(
                    {"returns": round(p[2] * 100, 2), "volatility": round(p[3] * 100, 2),
                     "sharpe": round(p[4], 2), "returns lowest vol": round(p[8] * 100, 2),
                     "lowest volatility": round(p[7] * 100, 2),
                     "sharpe lowest vol": round(p[9], 2), "timestamp": p[11].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]})
            except Exception as a:
                print("RefreshRowData", a)
        self.aggrid.update()

    def GetAllAssetsList(self):
        portfolioUtilities = PortfolioUtilities.PortfolioUtilities()
        isin = portfolioUtilities.GetAllAssets()
        for i in isin:
            self.fullAssetsList.append({"Name": i[0], "ISIN": i[1], "Volatility": i[2], "Sum positive": i[3], "Sum negative": i[4], "File name": i[5]})
        self.fullAssets.update()

    def ChangeValue(self, event):
        if event.sender.text == "Kelly optimization":
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
                row['Selection'] = nbAssets
                break
        return [[row['File name'], row["Selection"], row['Number of assets'], row['Size']] for row in self.fileData]

    def _on_cell_clicked(self, event):
        self.portfolioStructure = self.update_number_of_assets(event.args['data']['Selection'], event.args['data']['File name'])

    def displayGraph(self, localDef):
        self.ax.clear()
        for column in localDef.columns:
            self.ax.plot(localDef.index, localDef[column], label=column)
        self.ax.figure.canvas.draw()
        self.ax.legend()
        self.myGraph.update()
        self.finalPortfolio.update()
        self.aggrid.update()


    def display_portfolio(self, bestPortfolio):
        self.finalPortfolio.clear()
        self.assetsName.clear()
        portfolioUtilities = PortfolioUtilities.PortfolioUtilities()
        self.rowData.append({"returns": round(bestPortfolio[2] * 100, 2), "volatility": round(bestPortfolio[3] * 100, 2),
         "sharpe": round(bestPortfolio[4], 2), "returns lowest vol": round(bestPortfolio[8] * 100, 2),
         "lowest volatility": round(bestPortfolio[7] * 100, 2), "sharpe lowest vol": round(bestPortfolio[9], 2),
         "timestamp": bestPortfolio[-1].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]})
        assetsDescription = portfolioUtilities.ReturnAssetDescription(bestPortfolio[0])
        i = 0
        for a in assetsDescription:
            if self.kelly.value == True:
                self.assetsName.append({"Asset name": a, "Weight": bestPortfolio[1][i], "ISIN": bestPortfolio[0][i], "Kelly weight": bestPortfolio[-2][i]})
            else:
                self.assetsName.append({"Asset name": a, "ISIN": bestPortfolio[0][i], "Weight": bestPortfolio[1][i]})
            i += 1

        self.displayGraph( bestPortfolio[6][bestPortfolio[0]])

    def Simulate(self, nbOfSimulation, nbOfWeight):
        showDensity = False
        isRandom = True

        portfolio = Portfolio.ModernPortfolioTheory(nbOfSimulation, 2, 4, self.parallel)
        portfolio.nbOfSimulatedWeights = nbOfWeight
        data, isin = portfolio.BuilHeterogeneousPortfolio(self.portfolioStructure)
        if len(isin) == 0:
            return []
        bestPortfolios = self.parallel.run_select_random_assets_parallel(portfolio, data, isin, nbOfSimulation,
                                           self.portfolioStructure, showDensity, isRandom, [])
        if self.kelly.value == True:
            kelly = KellyPortfolio.KellyCriterion()
            kellyResult =  kelly.SolveKellyCriterion(bestPortfolios[5], len(bestPortfolios[5].columns))
            bestPortfolios.append(list(kellyResult))
        else:
            bestPortfolios.append([])
        return bestPortfolios

    def Callback(self):
        try:
            bestPortfolio = []
            self.btnSimulate.disable()
            self.spinner.visible = True
            bestPortfolio = self.Simulate(int(self.nbOfSimulation.value), int(self.nbOfWeight.value))
            if len(bestPortfolio) == 0:
                with self.fullAssets:
                    ui.notify("No Assets Class Selected")
                    self.btnSimulate.enable()
                    self.spinner.visible = False
                return
        except Exception as e:
            print("💥 ERREUR pendant la simulation :", e)
            traceback.print_exc()
        finally:
            #self.results = pd.read_pickle(self.workingPath + "results.pkl")
            bestPortfolio.append(datetime.now())
            if not os.path.exists(self.workingPath + "results.pkl"):
                self.results = pd.DataFrame(columns=["Portfolios"], dtype=object)
            self.results = pd.concat([self.results, pd.DataFrame([{"Portfolio": bestPortfolio}])], ignore_index=True)
            if not len(bestPortfolio) == 1:
                self.results.to_pickle(self.workingPath + "results.pkl")
                self.display_portfolio(bestPortfolio)
            self.btnSimulate.enable()
            self.spinner.visible = False
            if self.Sound.value:
                self.a.play()
        return bestPortfolio

    def handle_theme_change(self, e: events.ValueChangeEventArguments):
        self.aggrid.classes(add='ag-theme-balham-dark' if e.value else 'ag-theme-balham',
                     remove='ag-theme-balham ag-theme-balham-dark')

def update_ui(obj, event, queue):
    while True:
        event.wait()
        print("on recoit quelque chose")
        try:
            message = queue.get_nowait()
            obj.notify(message)
        except:
            obj.notify("[Main] Signal received, but queue is empty")
        event.clear()
        time.sleep(1)


@ui.page('/')
async def main_page():
    # on crée la grille (ou l’onglet) utilisé pour notifier
    multiprocessing.set_start_method('spawn', force=True)
    parallel = ParallelComputing.Parallel()
    app_gui = Gui(parallel)
    splitter = app_gui.splitter
    threading.Thread(target=update_ui, args=(splitter, parallel.event, parallel.queueMessages), daemon=True).start()

ui.run()
