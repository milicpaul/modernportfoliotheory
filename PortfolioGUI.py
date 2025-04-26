from tkinter import Event
import psutil
import pandas as pd
from datetime import datetime
from nicegui import ui, events, run
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
        with ui.splitter(value=60).style('width: 100%') as splitterTop:
            with splitterTop.before:
                with ui.row():
                    self.nbOfSimulation = ui.input(label='Portfolios simulation:', value='5')
                    self.nbOfWeight = ui.input(label='Weights by simulation:', value='10000')
                    self.kelly = ui.switch('Kelly ', on_change=self.ChangeValue)
                    self.Robust = ui.switch('Robust', on_change=self.ChangeValue)
                    self.Sound = ui.switch("Sound", on_change=self.ChangeValue)
                    self.ShowLog = ui.switch("Show log")
                with ui.row():
                    mem = psutil.virtual_memory()
                    self.label_total = ui.label(f"Tot: {mem.total / (1024 ** 3):.2f} Go")
                    self.label_available = ui.label(f"Avalaible: {mem.available / (1024 ** 3):.2f} Go")
                    self.label_used = ui.label(f"Used: {mem.used / (1024 ** 3):.2f} Go")
                    self.label_percent = ui.label(f"RAM: {mem.percent}%")
                    self.label_process = ui.label()
                    self.queue_size = ui.label()

            with splitterTop.after:
                with ui.row():
                    with ui.card().style('width: 500px; height: 120px;'):  # <<<<<< ICI, tu ajustes la taille
                        self.process_graph = self.ProcessGraph()
        with ui.splitter(horizontal=False, reverse=False, value=45,
                         on_change=lambda e: ui.notify(e.value)).style('width: 100%') as self.splitter:
            with self.splitter.before:
                files = [f for f in os.listdir(self.path) if f.endswith('.pkl')]
                files.sort()
                for f in files:
                    df = pd.read_pickle(self.path2 + f)
                    self.fileData.append({"File name": f, "Selection": 0, "Number of assets": len(df.columns), "Size": self.taille_lisible(os.path.getsize(self.path2 + f))})
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
                    }).classes('max-h-50').style('width: 98%').on('cellClicked', lambda event: self.FindPortfolio(event))
                self.spinner = ui.spinner(size='lg', color='primary').classes("items-centered")
                self.spinner.visible = False  # Caché tant que pas en traitement
                self.finalPortfolio = ui.aggrid({
                    'columnDefs': [
                        {"field": "Asset name"},
                        {"field": "ISIN"},
                        {"field": "Weight"}
                    ],
                    'rowData': self.assetsName
                }).classes('max-h-50').style('width: 98%')
                ui.switch('Dark', on_change=self.handle_theme_change)
                self.btnSimulate = ui.button('Simulate', on_click=self.Callback)
                ui.separator()
            with self.splitter.after:
                self.aggrid2 = ui.aggrid({
                    'columnDefs':
                        self.columnDefs,
                    'rowData':
                        self.fileData
                }).classes('max-h-80')
                self.log = ui.log(max_lines=10000).classes('w-full h-30').bind_visibility_from(self.ShowLog, 'value')
                self.aggrid2.on('cellValueChanged', self._on_cell_clicked)
                self.myGraph = ui.matplotlib(figsize=(9, 5))
                self.ax = self.myGraph.figure.gca()
                self.ax.set_yscale('log')
                self.ax.set_title("Données issues d'un DataFrame")
                self.ax.set_xlabel("x")
                self.ax.set_ylabel("Valeurs")
                self.a = ui.audio(self.path2 + 'mixkit-air-zoom-vacuum-2608.wav')
                self.a.visible = False

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
        if os.path.exists(self.workingPath + "results.pkl"):
            self.RefreshRowData()
        #self.GetAllAssetsList()


    def color_for_usage(self, usage):
        if usage < 50:
            return '#4caf50'  # Vert
        elif usage < 80:
            return '#ff9800'  # Orange
        else:
            return '#f44336'  # Rouge

    def ProcessGraph(self):
        return ui.echart({
            'xAxis': {'type': 'category', 'data': [f'Cœur {i}' for i in range(psutil.cpu_count())]},
            'yAxis': {'type': 'value', 'max': 100},
            'series': [{
                'type': 'bar',
                'data': [0] * psutil.cpu_count(),
                'itemStyle': {
                    'color': '#4caf50'
                }
            }]
        }).classes('W: full H: full').style('background-color: #121212; color: white;')

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
                self.ShowLog.value = True
                self.log.push(f"[RefreshData] {a}")
        self.aggrid.update()

    def GetAllAssetsList(self):
        portfolioUtilities = PortfolioUtilities.PortfolioUtilities()
        isin = portfolioUtilities.GetAllAssets()
        for i in isin:
            self.fullAssetsList.append({"Name": i[0], "ISIN": i[1], "Volatility": i[2], "Sum positive": i[3], "Sum negative": i[4], "File name": i[5]})
        self.fullAssets.update()

    def ChangeValue(self, event):
        if event.sender.text == "Kelly":
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
        self.log.clear()
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

    async def Callback(self):
        try:
            self.ShowLog.value = True
            bestPortfolio = []
            self.btnSimulate.disable()
            self.spinner.visible = True
            bestPortfolio = await run.io_bound(self.Simulate,int(self.nbOfSimulation.value), int(self.nbOfWeight.value))
            if len(bestPortfolio) == 0:
                with self.fullAssets:
                    ui.notify("No Assets Class Selected")
                    self.btnSimulate.enable()
                    self.spinner.visible = False
                return
        except Exception as e:
            print("💥 ERREUR pendant la simulation :", e)
            self.log.push(e)
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
            self.ShowLog.value = False
        return bestPortfolio

    def handle_theme_change(self, e: events.ValueChangeEventArguments):
        self.aggrid.classes(add='ag-theme-balham-dark' if e.value else 'ag-theme-balham',
                     remove='ag-theme-balham ag-theme-balham-dark')

def update_memory(app):
    process = psutil.Process()
    while True:
        mem = psutil.virtual_memory()
        proc_mem = process.memory_info()
        app.label_total.text = f"Tot: {mem.total / (1024**3):.2f} Go"
        app.label_available.text = f"Available: {mem.available / (1024**3):.2f} Go"
        app.label_used.text = f"Used: {mem.used / (1024**3):.2f} Go"
        app.label_percent.text = f"RAM: {mem.percent}%"
        app.label_process.text = f"Script: {proc_mem.rss / (1024**2):.2f} Mo"
        app.queue_size.text = "Queue size: " + str(app.parallel.queueMessages.qsize())
        time.sleep(0.2)

def update_ui(obj, event, queue):
    p = psutil.Process(os.getpid())
    # Sur macOS, tu peux définir la "nice value"
    while True:
        event.wait()
        try:
            while not queue.empty():
                message = queue.get_nowait()
                time.sleep(0.05)
                try:
                    with obj:
                        obj.push(message)
                except:
                    obj.push(f"[Listener]{a}")
        except:
            time.sleep(0.1)
        event.clear()

def update_chart(app):
    while True:
        usage = psutil.cpu_percent(percpu=True)
        app.process_graph.options['series'][0]['data'] = [
            {
                'value': u,
                'itemStyle': {'color': app.color_for_usage(u)}
            }
            for u in usage
        ]
        app.process_graph.update()
        time.sleep(0.2)


@ui.page('/')
async def main_page():
    # on crée la grille (ou l’onglet) utilisé pour notifier
    multiprocessing.set_start_method('spawn', force=True)
    parallel = ParallelComputing.Parallel()
    app_gui = Gui(parallel)
    threading.Thread(target=update_ui, args=(app_gui.log, parallel.event, parallel.queueMessages), daemon=True).start()
    threading.Thread(target=update_memory, args=(app_gui,), daemon=True).start()
    threading.Thread(target=update_chart, args=(app_gui,), daemon=True).start()

ui.run()
